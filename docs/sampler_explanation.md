# LLM Sampler Deep Dive

This document explains how Ember's sampler converts model logits into the next
output token, and how each option affects behavior.

## 1. Why sampling exists

A model does not directly output "the next word". It outputs a logits vector:

```text
logits = [2.1, -0.5, 5.3, 1.2, ...]  # length = vocab_size
```

Each value is an unnormalized score for one token.
A sampler must:

1. apply optional penalties
2. transform scores into probabilities
3. pick one token

## 2. End-to-end sampling pipeline

Ember applies the following order:

```text
logits
  -> (1) pre-softmax penalties
     - repetition penalty
     - presence penalty
     - frequency penalty
     - optional no-repeat n-gram mask
  -> (2) temperature scaling
  -> (3) softmax
  -> (4) top-k filter
  -> (5) top-p filter
  -> (6) categorical draw
  -> token_id
```

Order matters. For example, top-k/top-p run on probabilities after temperature.

## 3. Softmax: logits to probabilities

Given logits `z_i`, probabilities are:

```text
P(i) = exp(z_i) / sum_j exp(z_j)
```

Properties:

- each `P(i)` is in `(0, 1)`
- all probabilities sum to `1`
- ranking is preserved

### Numerical stability

Direct `exp(z_i)` can overflow for large logits. The stable form is:

```text
P(i) = exp(z_i - z_max) / sum_j exp(z_j - z_max)
```

where `z_max = max(z)`.

## 4. Temperature

Temperature rescales logits before softmax:

```text
z'_i = z_i / T
```

- `T < 1`: sharper distribution, less randomness
- `T = 1`: unchanged
- `T > 1`: flatter distribution, more randomness

Typical range: `0.6 ~ 1.0`.

## 5. Top-K filtering

Keep only the `K` highest-probability tokens. Set all others to zero, then
renormalize.

- small `K` (e.g. `20`): stable, conservative
- larger `K` (e.g. `100`): more diverse

When `K <= 0` or `K >= vocab_size`, top-k is effectively disabled.

## 6. Top-P (nucleus) filtering

Sort tokens by probability descending, keep the smallest prefix whose cumulative
probability reaches `P`.

Example (`top_p = 0.9`):

```text
0.40, 0.25, 0.15, 0.08, 0.06, 0.03, ...
cum:0.40 0.65 0.80 0.88 0.94
```

Keep first 5 tokens (cumulative crosses 0.9), remove the rest, renormalize.

- `top_p` near `1.0`: broader candidate set
- lower `top_p` (e.g. `0.8`): tighter, safer output

## 7. Penalties

Penalties are applied pre-softmax (on logits), using generated history.

## 7.1 Repetition penalty

Intuition: discourage selecting tokens already seen in history.

A common rule is sign-aware scaling:

```text
if logit > 0: logit /= repetition_penalty
else:         logit *= repetition_penalty
```

- `1.0` means disabled
- `> 1.0` increases suppression of repeated tokens

## 7.2 Presence penalty

Subtract a fixed value if token appeared at least once:

```text
logit -= presence_penalty * I[token_seen]
```

This encourages introducing new tokens/topics.

## 7.3 Frequency penalty

Subtract proportionally to occurrence count:

```text
logit -= frequency_penalty * count(token)
```

This strongly suppresses repeatedly used tokens.

## 7.4 No-repeat n-gram

If enabled, any token that would complete an already-seen `n`-gram is masked
(out probability becomes zero).

This is useful when generation tends to loop phrases.

## 8. Categorical sampling

After filtering and renormalization, the sampler draws one token according to
its probability mass.

Pseudo process:

```text
u ~ Uniform(0, 1)
running = 0
for token in candidates:
  running += p[token]
  if running >= u:
    return token
```

This is why fixed seed + fixed runtime path are important for reproducibility.

## 9. Parameter interactions

## 9.1 Greedy-like setup

```text
temp = 0 or very small
(top-k/top-p mostly irrelevant)
```

Highly deterministic, often repetitive.

## 9.2 Common balanced setup

```text
temp = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.05 ~ 1.15
```

Good baseline for interactive text generation.

## 9.3 Creative setup

```text
temp = 0.9 ~ 1.1
top_k = 80 ~ 200
top_p = 0.95 ~ 0.98
```

More diverse output, but less stable.

## 10. Typical failure modes and fixes

1. Repetition loops:
   increase `repetition_penalty`, add `frequency_penalty`, or enable no-repeat
   n-gram.
2. Output is too random:
   reduce `temp`, reduce `top_p`, reduce `top_k`.
3. Output is too rigid / boring:
   increase `temp`, slightly raise `top_p`/`top_k`.
4. Abrupt topic drift:
   lower `temp` and tighten `top_p`.

## 11. Reference pseudo-code

```cpp
int sample_next_token(std::vector<float>& logits,
                      const History& hist,
                      const SamplingParams& p,
                      RNG& rng) {
  // 1) penalties (pre-softmax)
  apply_repetition_penalty(logits, hist, p.repetition_penalty);
  apply_presence_penalty(logits, hist, p.presence_penalty);
  apply_frequency_penalty(logits, hist, p.frequency_penalty);
  apply_no_repeat_ngram_mask(logits, hist, p.no_repeat_ngram_size);

  // 2) temperature
  if (p.temperature > 0.0f && p.temperature != 1.0f) {
    for (float& v : logits) v /= p.temperature;
  }

  // 3) softmax (stable)
  softmax_inplace(logits);

  // 4) top-k
  if (p.top_k > 0) {
    top_k_filter_inplace(logits, p.top_k);
  }

  // 5) top-p
  if (p.top_p > 0.0f && p.top_p < 1.0f) {
    top_p_filter_inplace(logits, p.top_p);
  }

  renormalize(logits);

  // 6) categorical draw
  return sample_from_probs(logits, rng);
}
```

## 12. Practical defaults for Ember

For general chat/instruction generation, start with:

```text
temperature = 0.7
top_p = 0.9
top_k = 40
repetition_penalty = 1.1
presence_penalty = 0.0
frequency_penalty = 0.0
```

Then tune one knob at a time.

## 13. Reproducibility tips

- Fix RNG seed when comparing runs.
- Keep model, prompt, context length, and sampler params identical.
- Avoid changing hardware/runtime path while doing A/B quality checks.
- For debugging, dump logits before and after penalties for a few steps.

## 14. Related files

- `core/sampler.h`: sampler logic and parameters
- `apps/ember_cli/main.cpp`: CLI arguments and default sampler settings
- `docs/testing.md`: correctness and regression workflows

## 15. Chinese version

The original Chinese long-form explanation is preserved at:

- `docs/sampler_explanation.zh.md`
