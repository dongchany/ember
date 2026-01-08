# Testing and Regression

This project uses a layered testing approach.

## 1. Build-only check (always run)

```
cmake --build build -j
```

CI helper:
```
scripts/ci/build.sh
```

## 2. Correctness check (GPU)

Dump outputs:
```
./build/ember -m /path/to/model --check --dump-layer 2 -p "Hello, my name is"
```

Compare logits:
```
python3 scripts/compare_logits.py \
  --model /path/to/model \
  --debug-dir debug/check_models--Qwen--Qwen3-0_6B
```

Compare hidden states:
```
python3 scripts/compare_hidden.py \
  --model /path/to/model \
  --debug-dir debug/check_models--Qwen--Qwen3-0_6B \
  --layer 2
```

## 3. Sampling sanity (GPU)

Greedy (deterministic):
```
./build/ember -m /path/to/model -p "Hello, my name is" --temp 0 --top-k 1 --top-p 1
```

Typical sampling:
```
./build/ember -m /path/to/model -p "Hello, my name is" \
  --temp 0.7 --top-p 0.9 --top-k 40 \
  --repeat-penalty 1.1 --presence-penalty 0.2 --frequency-penalty 0.2 \
  --no-repeat-ngram 3
```

## Expected thresholds

These are guidance values for Qwen3-0.6B:
- `compare_logits.py`: max_abs_diff < 4.0, mean_abs_diff < 1.0
- `compare_hidden.py` (layer 2): max_abs_diff ~1.0, mean_abs_diff ~0.2

Use these as guardrails, not strict guarantees.
