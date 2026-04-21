"""
run_tier1.py — TASEF Baseline Sweep Runner

Evaluates the Tool Schema Dilution (CEI) attack across models at multiple
injection turn depths. Supports all three causal arms:

  Arm B (experimental):  20 cloud-infrastructure schemas injected into tools[].
  Arm A (no-dilution):   Identical conversation, zero extra schemas.
  Arm C (padding):       Token-matched neutral schemas, isolates semantic content.

Runs trials concurrently via asyncio (--concurrency flag) to maximize
API throughput without burning rate-limit headroom.

Usage:
    # Smoke test (no API calls)
    python run_tier1.py --models mock --trials-per-config 3 --turns 5 10

    # Replicate Mistral anchor experiment (paper §3)
    python run_tier1.py --models mistral --trials-per-config 30 \\
        --turns 5 10 13 16 20 25 --include-control --concurrency 5

    # Full 12-model sweep
    python run_tier1.py --models haiku mini flash mistral nemotron qwen \\
        --trials-per-config 30 --turns 5 10 13 16 20 25 \\
        --include-control --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasef.agent_loop import TrialConfig, run_trial
from tasef.telemetry import TelemetryLogger

# ─── Model aliases ─────────────────────────────────────────────────────────────────────────────

MODEL_ALIASES = {
    # ── Anthropic (US) ────────────────────────────────────────────
    "haiku":      "anthropic/claude-3-haiku-20240307",    # Cheap baseline
    "sonnet":     "anthropic/claude-sonnet-4-20250514",   # Mid-tier
    "opus":       "anthropic/claude-opus-4-20250514",     # Frontier
    # ── OpenAI (US) ───────────────────────────────────────────────
    "mini":       "gpt-4o-mini",                          # Cheap baseline
    "gpt4o":      "gpt-4o",                               # Mid-tier
    "gpt41":      "openai/gpt-4.1",                       # Frontier
    # ── Google (US) ───────────────────────────────────────────────
    "flash":      "gemini/gemini-2.5-flash",              # Cheap baseline
    "gemini-pro": "gemini/gemini-2.5-pro",                # Frontier
    # ── NVIDIA NIM — free endpoints ──────────────────────────────
    "deepseek":   "nvidia_nim/deepseek-ai/deepseek-v3.2",                   # DeepSeek, China (MoE)
    "mistral":    "nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512", # Mistral AI, France (MoE)
    "nemotron":   "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",            # NVIDIA, US (MoE)
    "qwen":       "nvidia_nim/qwen/qwen3.5-397b-a17b",                      # Alibaba, China (MoE)
    "glm":        "nvidia_nim/z-ai/glm-4.7",                                # Zhipu AI, China
    "kimi":       "nvidia_nim/moonshotai/kimi-k2.5",                         # Moonshot AI, China (MoE)
    "llama":      "nvidia_nim/meta/llama-3.3-70b-instruct",                  # Meta, US
    "gemma":      "nvidia_nim/google/gemma-4-31b-it",                        # Google, US
    # ── Testing ───────────────────────────────────────────────────
    "mock":       "mock",
}

# Per-model API keys — for NVIDIA NIM models
# Set via environment variables: TASEF_KEY_<ALIAS> (uppercase)
# Or hardcode temporarily on the GCP instance (never push keys to GitHub)
import os as _os
NIM_API_KEYS: dict[str, str | None] = {
    "deepseek":  _os.environ.get("TASEF_KEY_DEEPSEEK"),
    "mistral":   _os.environ.get("TASEF_KEY_MISTRAL"),
    "nemotron":  _os.environ.get("TASEF_KEY_NEMOTRON"),
    "qwen":      _os.environ.get("TASEF_KEY_QWEN"),
    "kimi":      _os.environ.get("TASEF_KEY_KIMI"),
    "llama":     _os.environ.get("TASEF_KEY_LLAMA"),
    "gemma":     _os.environ.get("TASEF_KEY_GEMMA"),
}

# ─── Default sweep config ───────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["mock"]
DEFAULT_INJECTION_TURNS = [5, 10, 13, 16, 20, 25]
DEFAULT_TRIALS = 5  # Start low; bump to 25-30 for real runs


def parse_args():
    parser = argparse.ArgumentParser(description="TASEF Tier 1 — Baseline sweep")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to evaluate (aliases: haiku, mini, flash, llama, mistral, mock)",
    )
    parser.add_argument(
        "--turns",
        nargs="+",
        type=int,
        default=DEFAULT_INJECTION_TURNS,
        help="Injection turn depths to test",
    )
    parser.add_argument(
        "--trials-per-config",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of trials per (model x turn x arm) configuration",
    )
    parser.add_argument(
        "--dilution-schemas",
        type=int,
        default=20,
        help="Number of fake tool schemas to inject (experimental arm only)",
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        default=False,
        help=(
            "Arm A: paired control with ZERO dilution schemas. "
            "Same seeds, same injection message. "
            "Isolates schema dilution as the causal mechanism vs. recency bias."
        ),
    )
    parser.add_argument(
        "--include-padding-control",
        action="store_true",
        default=False,
        help=(
            "Arm C: token-matched neutral padding control. "
            "Injects the same number of schemas as the experimental arm but with "
            "semantically neutral monitoring-probe content instead of cloud-ops schemas. "
            "Isolates schema *content* from context *length* as the causal variable. "
            "Requires --include-control to also be set for a complete 3-arm design."
        ),
    )
    parser.add_argument(
        "--padding-only",
        action="store_true",
        default=False,
        help=(
            "Run Arm C (token-matched padding) only — skip Arm B and Arm A. "
            "Use when Arm A and B data already exist and you only need to fill Arm C."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to sleep between API requests (rate limiting)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of trials to run concurrently per arm. "
            "Set to 5-20 to maximize API throughput. "
            "Default 1 (sequential) for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
    )
    return parser.parse_args()


async def _run_arm_async(
    model: str,
    model_alias: str,
    injection_turn: int,
    trials_per_config: int,
    num_dilution_schemas: int,
    delay: float,
    logger: TelemetryLogger,
    disable_dilution: bool,
    use_padding_schemas: bool,
    completed_base: int,
    total_trials: int,
    concurrency: int = 1,
    api_key: str | None = None,
) -> tuple[int, int, int]:
    """
    Run one arm for a given (model, injection_turn), with concurrent trials.

    Three-arm design:
      Arm B — Experimental:       disable_dilution=False, use_padding_schemas=False
      Arm A — No-dilution ctrl:   disable_dilution=True,  use_padding_schemas=False
      Arm C — Padding ctrl:       disable_dilution=False, use_padding_schemas=True

    Trials run concurrently up to `concurrency` at a time via asyncio.to_thread,
    which offloads each blocking run_trial call to a thread-pool worker without
    blocking the event loop. TelemetryLogger is thread-safe (WAL + fcntl).

    Returns:
        (trials_completed, exploits, srd_confirmed)
    """
    if disable_dilution:
        arm_label = "CONTROL-A (no dilution)"
        vector_type = "no_dilution_control"
    elif use_padding_schemas:
        arm_label = "CONTROL-C (token-matched padding)"
        vector_type = "token_matched_padding"
    else:
        arm_label = "EXPERIMENTAL (dilution)"
        vector_type = "tool_schema_dilution"

    dilution_start = max(1, injection_turn - 8)
    capability_turn = injection_turn + 1
    max_turns = capability_turn + 2

    print(f"  [{arm_label}] {model_alias} @ turn={injection_turn} "
          f"({trials_per_config} trials, concurrency={concurrency})...")

    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    exploits = 0
    srd_confirmed = 0
    lock = asyncio.Lock()

    async def _run_one(seed: int) -> None:
        nonlocal completed_count, exploits, srd_confirmed
        config = TrialConfig(
            model=model,
            vector_type=vector_type,
            dilution_start_turn=dilution_start,
            injection_turn=injection_turn,
            capability_check_turn=capability_turn,
            max_turns=max_turns,
            num_dilution_schemas=num_dilution_schemas,
            dilution_schema_seed=seed * 7,
            disable_dilution=disable_dilution,
            use_padding_schemas=use_padding_schemas,
            seed=seed,
            request_delay_seconds=delay,
            api_key=api_key,
        )
        async with semaphore:
            result = await asyncio.to_thread(run_trial, config, logger)

        async with lock:
            completed_count += 1
            global_idx = completed_base + completed_count
            if result.exploit_detected:
                exploits += 1
            if result.srd_data_point:
                srd_confirmed += 1

            if result.error:
                print(f"    [WARN] seed={seed} error: {result.error}")
                return

            status = "[EXPLOIT]" if result.exploit_detected else "[SECURE]"
            srd = " | [SRD]" if result.srd_data_point else ""
            print(
                f"    [{global_idx}/{total_trials}] seed={seed} {status}{srd} "
                f"ttf={result.turns_to_failure}"
            )

    await asyncio.gather(*[_run_one(seed) for seed in range(trials_per_config)])

    rate = exploits / completed_count * 100 if completed_count > 0 else 0.0
    print(f"    > Exploit rate: {exploits}/{completed_count} ({rate:.1f}%)")
    return completed_count, exploits, srd_confirmed


async def _run_sweep_async(
    models: list[str],
    injection_turns: list[int],
    trials_per_config: int,
    num_dilution_schemas: int,
    include_control: bool,
    include_padding_control: bool,
    padding_only: bool,
    delay: float,
    concurrency: int,
    data_dir: Path,
) -> None:
    logger = TelemetryLogger(data_dir=data_dir)
    arms = (0 if padding_only else 1) + (1 if include_control and not padding_only else 0) + (1 if include_padding_control or padding_only else 0)
    total_trials = len(models) * len(injection_turns) * trials_per_config * arms

    if padding_only:
        arm_desc = "Arm C only (token-matched padding)"
    elif include_padding_control:
        arm_desc = "Experimental (B) + No-dilution (A) + Padding (C) — full 3-arm design"
    elif include_control:
        arm_desc = "Experimental (B) + No-dilution control (A)"
    else:
        arm_desc = "Experimental only"

    print(f"\n{'='*60}")
    print(f"  TASEF — Tool Schema Dilution Sweep")
    print(f"{'='*60}")
    print(f"  Models:          {models}")
    print(f"  Injection turns: {injection_turns}")
    print(f"  Trials/config:   {trials_per_config} per arm")
    print(f"  Arms:            {arm_desc}")
    print(f"  Concurrency:     {concurrency} trials/arm")
    print(f"  Total trials:    {total_trials}")
    print(f"  Data dir:        {data_dir}")
    print(f"{'='*60}\n")

    total_completed_exp = 0
    total_completed_ctrl = 0
    total_completed_pad = 0
    total_exploits_exp = 0
    total_exploits_ctrl = 0
    total_exploits_pad = 0

    for model_alias in models:
        model = MODEL_ALIASES.get(model_alias, model_alias)
        model_key = NIM_API_KEYS.get(model_alias)
        for injection_turn in injection_turns:
            base = total_completed_exp + total_completed_ctrl + total_completed_pad

            # ── Arm B: Experimental (dilution schemas) ────────────────────────
            if not padding_only:
                done_exp, ex, _ = await _run_arm_async(
                    model=model,
                    model_alias=model_alias,
                    injection_turn=injection_turn,
                    trials_per_config=trials_per_config,
                    num_dilution_schemas=num_dilution_schemas,
                    delay=delay,
                    logger=logger,
                    disable_dilution=False,
                    use_padding_schemas=False,
                    completed_base=base,
                    total_trials=total_trials,
                    concurrency=concurrency,
                    api_key=model_key,
                )
                total_completed_exp += done_exp
                total_exploits_exp += ex

            # ── Arm A: No-dilution control ────────────────────────────────────
            if include_control and not padding_only:
                base = total_completed_exp + total_completed_ctrl + total_completed_pad
                done_ctrl, ct, _ = await _run_arm_async(
                    model=model,
                    model_alias=model_alias,
                    injection_turn=injection_turn,
                    trials_per_config=trials_per_config,
                    num_dilution_schemas=num_dilution_schemas,
                    delay=delay,
                    logger=logger,
                    disable_dilution=True,
                    use_padding_schemas=False,
                    completed_base=base,
                    total_trials=total_trials,
                    concurrency=concurrency,
                    api_key=model_key,
                )
                total_completed_ctrl += done_ctrl
                total_exploits_ctrl += ct

            # ── Arm C: Token-matched padding control ──────────────────────────
            if include_padding_control or padding_only:
                base = total_completed_exp + total_completed_ctrl + total_completed_pad
                done_pad, pd, _ = await _run_arm_async(
                    model=model,
                    model_alias=model_alias,
                    injection_turn=injection_turn,
                    trials_per_config=trials_per_config,
                    num_dilution_schemas=num_dilution_schemas,
                    delay=delay,
                    logger=logger,
                    disable_dilution=False,
                    use_padding_schemas=True,
                    completed_base=base,
                    total_trials=total_trials,
                    concurrency=concurrency,
                    api_key=model_key,
                )
                total_completed_pad += done_pad
                total_exploits_pad += pd

    total_completed = total_completed_exp + total_completed_ctrl + total_completed_pad
    exp_rate = total_exploits_exp / total_completed_exp * 100 if total_completed_exp > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE")
    print(f"  Total trials:               {total_completed}")
    print(f"  [ARM B] Experimental:       {total_exploits_exp}/{total_completed_exp} ({exp_rate:.1f}%)")
    if include_control:
        ctrl_rate = total_exploits_ctrl / total_completed_ctrl * 100 if total_completed_ctrl > 0 else 0.0
        print(f"  [ARM A] No-dilution ctrl:   {total_exploits_ctrl}/{total_completed_ctrl} ({ctrl_rate:.1f}%)")
        if ctrl_rate > 0:
            print(f"  Causal dilution effect:     {exp_rate / ctrl_rate:.1f}x (B vs A)")
    if include_padding_control:
        pad_rate = total_exploits_pad / total_completed_pad * 100 if total_completed_pad > 0 else 0.0
        print(f"  [ARM C] Padding ctrl:       {total_exploits_pad}/{total_completed_pad} ({pad_rate:.1f}%)")
        if pad_rate > 0:
            print(f"  Content vs length effect:   {exp_rate / pad_rate:.1f}x (B vs C)")
    print(f"  DB:                     {data_dir / 'results.db'}")
    print(f"  JSONL:                  {data_dir / 'results.jsonl'}")
    print(f"{'='*60}\n")
    print("Generate visualizations:")
    print(f"  python -m tasef.report --data-dir {data_dir}\n")


def main():
    args = parse_args()
    asyncio.run(_run_sweep_async(
        models=args.models,
        injection_turns=args.turns,
        trials_per_config=args.trials_per_config,
        num_dilution_schemas=args.dilution_schemas,
        include_control=args.include_control,
        include_padding_control=args.include_padding_control,
        padding_only=args.padding_only,
        delay=args.delay,
        concurrency=args.concurrency,
        data_dir=args.data_dir,
    ))


if __name__ == "__main__":
    main()
