#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_PROMPTS = [
    "Hello, my name is",
    "Explain the difference between RAM and VRAM in one short paragraph.",
    "Write a haiku about winter rain.",
    "List three safe ways to speed up C++ builds.",
    "Translate to French: The meeting starts at nine tomorrow.",
    "Give me a two-step plan to debug NaN in training.",
    "What is the capital of Japan?",
    "Summarize: CUDA kernels benefit from coalesced memory access.",
]


TOKEN_LINE_RE = re.compile(r"Generated token ids:\s*(.*)$")


def die(msg: str) -> int:
    print(f"[greedy-compare] {msg}", file=sys.stderr)
    return 1


def load_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            prompts.append(text)
    return prompts


def parse_token_line(stdout: str) -> List[int]:
    for line in reversed(stdout.splitlines()):
        m = TOKEN_LINE_RE.search(line.strip())
        if not m:
            continue
        payload = m.group(1).strip()
        if not payload:
            return []
        return [int(tok) for tok in payload.split()]
    raise ValueError("failed to parse 'Generated token ids' from decode loop output")


def run_ember_decode_loop(
    ember_bin: Path,
    model_path: Path,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    max_ctx_len: int,
    ember_device: int,
) -> List[int]:
    env = os.environ.copy()
    env["PROMPT_TOKENS"] = ",".join(str(t) for t in prompt_tokens)
    env["MAX_NEW_TOKENS"] = str(max_new_tokens)
    env["MAX_CTX_LEN"] = str(max_ctx_len)
    env["TEMPERATURE"] = "0"
    env["TOP_P"] = "1"
    env["TOP_K"] = "1"
    env["DEVICE_ID"] = str(ember_device)

    proc = subprocess.run(
        [str(ember_bin), str(model_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "decode loop failed\n"
            f"returncode={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return parse_token_line(proc.stdout)


def load_baseline(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    cases = obj.get("cases")
    if not isinstance(cases, list):
        raise ValueError("baseline format error: missing cases list")
    out: List[Dict[str, object]] = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"baseline case #{i} is not an object")
        prompt = case.get("prompt", "")
        prompt_ids = case.get("prompt_token_ids", [])
        expected = case.get("expected_continuation", [])
        if not isinstance(prompt, str):
            raise ValueError(f"baseline case #{i} prompt is not a string")
        if not isinstance(prompt_ids, list) or not all(isinstance(x, int) for x in prompt_ids):
            raise ValueError(f"baseline case #{i} prompt_token_ids is invalid")
        if not isinstance(expected, list) or not all(isinstance(x, int) for x in expected):
            raise ValueError(f"baseline case #{i} expected_continuation is invalid")
        out.append(
            {
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "expected_continuation": expected,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Ember greedy decode token IDs with HF (or saved baseline)."
    )
    parser.add_argument("--model", required=True, help="Model directory (local HF cache/snapshot).")
    parser.add_argument("--build-dir", default="build", help="Build directory for Ember binaries.")
    parser.add_argument(
        "--ember-bin",
        default="",
        help="Path to decode loop binary (default: <build-dir>/ember_example_decode_loop).",
    )
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Optional prompt list file (one prompt per line, '#' comments allowed).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of greedy tokens to decode for each prompt.",
    )
    parser.add_argument(
        "--max-ctx-len",
        type=int,
        default=512,
        help="Minimum decode-loop context length; script auto-raises per prompt if needed.",
    )
    parser.add_argument(
        "--ember-device",
        type=int,
        default=1,
        help="Single GPU device id for ember_example_decode_loop.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Optional GPU list (e.g. '1' or '0,1'); first GPU is used as ember device.",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="Optional baseline JSON (skips HF model inference if provided).",
    )
    parser.add_argument(
        "--write-baseline",
        default="",
        help="Write HF expectations to this JSON file (HF mode only).",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        return die("--max-new-tokens must be > 0")
    if args.max_ctx_len <= 0:
        return die("--max-ctx-len must be > 0")
    ember_device = args.ember_device
    if args.gpus:
        first = args.gpus.split(",", 1)[0].strip()
        if not first:
            return die("--gpus is set but first gpu id is empty")
        try:
            ember_device = int(first)
        except ValueError:
            return die(f"invalid gpu id in --gpus: {first}")
    if ember_device < 0:
        return die("--ember-device/--gpus must be >= 0")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        return die(f"model path not found: {model_path}")

    ember_bin = Path(args.ember_bin).expanduser().resolve() if args.ember_bin else (
        Path(args.build_dir).expanduser().resolve() / "ember_example_decode_loop"
    )
    if not ember_bin.exists():
        return die(f"missing decode-loop binary: {ember_bin}")

    baseline_cases: List[Dict[str, object]] = []
    if args.baseline:
        baseline_path = Path(args.baseline).expanduser().resolve()
        if not baseline_path.exists():
            return die(f"baseline file not found: {baseline_path}")
        try:
            baseline_cases = load_baseline(baseline_path)
        except Exception as exc:
            return die(f"failed to load baseline: {exc}")
        if not baseline_cases:
            return die("baseline has no cases")
    else:
        prompts: List[str]
        if args.prompts_file:
            prompts_path = Path(args.prompts_file).expanduser().resolve()
            if not prompts_path.exists():
                return die(f"prompts file not found: {prompts_path}")
            prompts = load_prompts(prompts_path)
            if not prompts:
                return die("prompts file has no usable prompts")
        else:
            prompts = list(DEFAULT_PROMPTS)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            return die(f"HF compare requires torch+transformers: {exc}")

        print(f"[greedy-compare] Loading tokenizer/model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()

        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"]
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0),
                )
            prompt_ids = input_ids[0].tolist()
            continuation = out[0, input_ids.shape[1]:].tolist()
            baseline_cases.append(
                {
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids,
                    "expected_continuation": continuation,
                }
            )

        if args.write_baseline:
            out_path = Path(args.write_baseline).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_path": str(model_path),
                        "max_new_tokens": args.max_new_tokens,
                        "cases": baseline_cases,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[greedy-compare] Wrote baseline: {out_path}")

    mismatches = 0
    for idx, case in enumerate(baseline_cases):
        prompt = str(case["prompt"])
        prompt_ids = [int(x) for x in case["prompt_token_ids"]]  # type: ignore[index]
        expected = [int(x) for x in case["expected_continuation"]]  # type: ignore[index]
        need_ctx = max(args.max_ctx_len, len(prompt_ids) + args.max_new_tokens + 8)

        try:
            history = run_ember_decode_loop(
                ember_bin=ember_bin,
                model_path=model_path,
                prompt_tokens=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                max_ctx_len=need_ctx,
                ember_device=ember_device,
            )
        except Exception as exc:
            print(f"[Case {idx}] FAIL: runtime error: {exc}")
            mismatches += 1
            continue

        if len(history) < len(prompt_ids):
            print(
                f"[Case {idx}] FAIL: output shorter than prompt "
                f"(history={len(history)}, prompt={len(prompt_ids)})"
            )
            mismatches += 1
            continue

        got = history[len(prompt_ids):]
        if got != expected:
            mismatch_at = -1
            for i in range(min(len(got), len(expected))):
                if got[i] != expected[i]:
                    mismatch_at = i
                    break
            if mismatch_at < 0 and len(got) != len(expected):
                mismatch_at = min(len(got), len(expected))

            print(f"[Case {idx}] FAIL")
            print(f"  prompt: {prompt}")
            print(f"  mismatch_at: {mismatch_at}")
            print(f"  expected(first 16): {expected[:16]}")
            print(f"  got(first 16):      {got[:16]}")
            mismatches += 1
            continue

        print(
            f"[Case {idx}] PASS "
            f"(prompt_tokens={len(prompt_ids)}, generated={len(got)})"
        )

    total = len(baseline_cases)
    passed = total - mismatches
    print(f"[Summary] passed={passed} failed={mismatches} total={total}")
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
