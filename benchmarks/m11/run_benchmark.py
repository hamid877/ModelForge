"""M11 Phase 2 — SQLite Benchmark Runner.

Executes the full benchmark matrix: scenarios W1–W6 (threads) and P1
(processes) across all configured concurrency levels and repetitions.

Usage
-----
Quick development profile (short durations):
    uv run python benchmarks/m11/run_benchmark.py --profile quick

Full evaluation profile (longer, more accurate):
    uv run python benchmarks/m11/run_benchmark.py --profile full

Single scenario:
    uv run python benchmarks/m11/run_benchmark.py --scenario W3 --threads 10

Custom output directory:
    uv run python benchmarks/m11/run_benchmark.py --output-dir /tmp/bench_results

Outputs
-------
- <output-dir>/results_<timestamp>.json  : Machine-readable full results
- <output-dir>/baseline_report_<timestamp>.md : Human-readable baseline report
- <output-dir>/integrity_<timestamp>.json : Per-run integrity verification
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.m11.harness import (  # noqa: E402
    MEASUREMENT_SECONDS_FULL,
    MEASUREMENT_SECONDS_QUICK,
    PROCESS_CONCURRENCY_LEVELS,
    REPETITIONS,
    THREAD_CONCURRENCY_LEVELS,
    WARMUP_SECONDS,
    RunMetrics,
    _check_correctness,
    _create_temp_db,
    _initialize_db,
    _seed_baseline_data,
    _verify_journal_mode,
    capture_environment,
    run_p1_processes,
    run_w1_threads,
    run_w2_threads,
    run_w3_threads,
    run_w4_threads,
    run_w5_threads,
    run_w6_threads,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("m11.runner")


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

THREAD_SCENARIOS = {
    "W1": run_w1_threads,
    "W2": run_w2_threads,
    "W3": run_w3_threads,
    "W4": run_w4_threads,
    "W5": run_w5_threads,
    "W6": run_w6_threads,
}

PROCESS_SCENARIOS = {
    "P1": run_p1_processes,
}

# W5 needs special seed parameter
W5_SCENARIO = "W5"


# ---------------------------------------------------------------------------
# Single scenario runner (creates its own temp DB per repetition)
# ---------------------------------------------------------------------------


def _run_single(
    scenario: str,
    concurrency: int,
    concurrency_type: str,
    repetition: int,
    warmup_s: float,
    measure_s: float,
) -> tuple[RunMetrics, dict]:
    """Run one scenario × concurrency × repetition. Returns (metrics, correctness)."""
    db_path = _create_temp_db()
    try:
        _initialize_db(db_path)
        journal = _verify_journal_mode(db_path)
        if journal != "delete":
            logger.warning(
                "Journal mode is '%s', expected 'delete'. Proceeding anyway.", journal
            )
        seed = _seed_baseline_data(db_path)

        logger.info(
            "  Running %s concurrency=%d (type=%s) rep=%d ...",
            scenario,
            concurrency,
            concurrency_type,
            repetition,
        )

        if scenario in THREAD_SCENARIOS:
            if scenario == "W5":
                metrics = run_w5_threads(
                    n=concurrency,
                    db_path=db_path,
                    warmup_s=warmup_s,
                    measure_s=measure_s,
                    repetition=repetition,
                    seed=seed,
                )
            else:
                fn = THREAD_SCENARIOS[scenario]
                # W2 needs special handling (seed job required)
                if scenario == "W2":
                    metrics = fn(
                        n=concurrency,
                        db_path=db_path,
                        warmup_s=warmup_s,
                        measure_s=measure_s,
                        repetition=repetition,
                    )
                else:
                    metrics = fn(
                        n=concurrency,
                        db_path=db_path,
                        warmup_s=warmup_s,
                        measure_s=measure_s,
                        repetition=repetition,
                    )
        elif scenario in PROCESS_SCENARIOS:
            fn = PROCESS_SCENARIOS[scenario]
            metrics = fn(
                n=concurrency,
                db_path=db_path,
                warmup_s=warmup_s,
                measure_s=measure_s,
                repetition=repetition,
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        correctness = _check_correctness(db_path)

        if not correctness["ok"]:
            logger.error(
                "CORRECTNESS FAILURE in %s concurrency=%d rep=%d: %s",
                scenario,
                concurrency,
                repetition,
                correctness["details"],
            )

        return metrics, correctness

    finally:
        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Full benchmark matrix
# ---------------------------------------------------------------------------


def run_benchmark_matrix(
    scenarios: list[str],
    thread_levels: tuple[int, ...],
    process_levels: tuple[int, ...],
    repetitions: int,
    warmup_s: float,
    measure_s: float,
) -> tuple[list[RunMetrics], list[dict]]:
    all_metrics: list[RunMetrics] = []
    all_correctness: list[dict] = []

    for scenario in scenarios:
        is_process = scenario in PROCESS_SCENARIOS
        levels = process_levels if is_process else thread_levels
        conc_type = "process" if is_process else "thread"

        logger.info("=== Scenario: %s (%s) ===", scenario, conc_type)

        for concurrency in levels:
            for rep in range(1, repetitions + 1):
                metrics, correctness = _run_single(
                    scenario=scenario,
                    concurrency=concurrency,
                    concurrency_type=conc_type,
                    repetition=rep,
                    warmup_s=warmup_s,
                    measure_s=measure_s,
                )
                all_metrics.append(metrics)
                all_correctness.append(
                    {
                        "scenario": scenario,
                        "concurrency_type": conc_type,
                        "concurrency": concurrency,
                        "repetition": rep,
                        **correctness,
                    }
                )

                # STOP IMMEDIATELY on correctness failure
                if not correctness["ok"] and correctness.get("stop"):
                    logger.error(
                        "Stopping benchmark: Correctness failure detected in "
                        "%s concurrency=%d rep=%d. Details: %s",
                        scenario,
                        concurrency,
                        rep,
                        correctness["details"],
                    )
                    return all_metrics, all_correctness

                _log_progress(metrics)

    return all_metrics, all_correctness


def _log_progress(m: RunMetrics) -> None:
    busy_rate = (m.busy_errors / m.total_ops * 100) if m.total_ops else 0.0
    logger.info(
        "    %s %s×%d rep%d: %d ops in %.1fs = %.1f ops/s | "
        "P50=%.1fms P95=%.1fms P99=%.1fms | "
        "busy=%d (%.3f%%) ownership=%d other=%d | integrity=%s",
        m.scenario,
        m.concurrency_type,
        m.concurrency,
        m.repetition,
        m.total_ops,
        m.duration_s,
        m.ops_per_second,
        m.p50_ms,
        m.p95_ms,
        m.p99_ms,
        m.busy_errors,
        busy_rate,
        m.ownership_rejections,
        m.other_failures,
        "OK" if m.integrity_ok else "FAIL",
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def generate_baseline_report(
    all_metrics: list[RunMetrics],
    all_correctness: list[dict],
    env: dict,
    profile: str,
    output_path: Path,
) -> None:
    """Generate human-readable markdown baseline report."""
    ts = env["timestamp_utc"]

    # Group by scenario + concurrency_type + concurrency
    from collections import defaultdict

    groups: dict[tuple, list[RunMetrics]] = defaultdict(list)
    for m in all_metrics:
        key = (m.scenario, m.concurrency_type, m.concurrency)
        groups[key].append(m)

    lines = [
        "# M11 Phase 2 — SQLite Baseline Benchmark Report",
        "",
        f"> Generated: {ts}  ",
        f"> Profile: **{profile}**  ",
        f"> Journal mode: **{env['journal_mode']}**  ",
        f"> Busy timeout: **{env['busy_timeout_ms']} ms**  ",
        f"> Warm-up: {env['warmup_seconds']} s | Measurement: varies by profile  ",
        f"> Repetitions: {env['repetitions']}  ",
        "",
        "## Environment",
        "",
        "| Key | Value |",
        "|-----|-------|",
        f"| Python | `{env['python_version'].splitlines()[0]}` |",
        f"| SQLite | `{env['sqlite_version']}` |",
        f"| Platform | `{env['platform']}` |",
        f"| CPU cores | {env['cpu_count']} |",
        "",
        "> [!IMPORTANT]",
        "> This is a baseline measurement only. WAL mode and other optimisations",
        "> have NOT been applied. These results characterise DELETE-mode behaviour.",
        "",
        "## Results Summary",
        "",
        "All metrics are averaged across repetitions unless noted.  ",
        "**Busy errors** = `sqlite3.OperationalError: database is locked`.  ",
        (
            "**Ownership rejections** = expected fencing failures "
            "(not counted as errors).  "
        ),
        "",
    ]

    # Scenarios in display order
    scenario_order = ["W1", "W2", "W3", "W4", "W5", "W6", "P1"]
    conc_type_for = {
        "W1": "thread",
        "W2": "thread",
        "W3": "thread",
        "W4": "thread",
        "W5": "thread",
        "W6": "thread",
        "P1": "process",
    }
    scenario_descriptions = {
        "W1": "Job claim contention",
        "W2": "Heartbeat write contention",
        "W3": "Concurrent drift assessments",
        "W4": "Combined write-heavy workload",
        "W5": "Read-heavy metrics queries during writes",
        "W6": "Promotion audit writes",
        "P1": "API/worker-style cross-process contention",
    }

    for scenario in scenario_order:
        ct = conc_type_for[scenario]
        sc_groups = {k: v for k, v in groups.items() if k[0] == scenario and k[1] == ct}
        if not sc_groups:
            continue

        lines.append(f"### {scenario}: {scenario_descriptions.get(scenario, scenario)}")
        lines.append("")
        lines.append(
            "| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | "
            "Busy | Busy% | Ownership | Other | Integrity |"
        )
        lines.append(
            "|-------------|-------|--------|--------|--------|"
            "------|-------|-----------|-------|-----------|"
        )

        sorted_keys = sorted(sc_groups.keys(), key=lambda k: k[2])
        for key in sorted_keys:
            reps = sc_groups[key]
            concurrency = key[2]
            avg_ops = _avg([m.ops_per_second for m in reps])
            avg_p50 = _avg([m.p50_ms for m in reps])
            avg_p95 = _avg([m.p95_ms for m in reps])
            avg_p99 = _avg([m.p99_ms for m in reps])
            total_busy = sum(m.busy_errors for m in reps)
            total_ops = sum(m.total_ops for m in reps)
            busy_pct = (total_busy / total_ops * 100) if total_ops else 0.0
            total_ownership = sum(m.ownership_rejections for m in reps)
            total_other = sum(m.other_failures for m in reps)
            integrity = all(m.integrity_ok for m in reps)
            lines.append(
                f"| {concurrency}× ({ct}) | {avg_ops:.1f} | {avg_p50:.2f} | "
                f"{avg_p95:.2f} | {avg_p99:.2f} | {total_busy} | {busy_pct:.3f}% | "
                f"{total_ownership} | {total_other} | {'✅' if integrity else '❌'} |"
            )
        lines.append("")

    # Integrity summary
    lines.append("## Integrity Verification")
    lines.append("")
    any_fail = any(not c.get("ok") for c in all_correctness)
    if any_fail:
        lines.append("> [!CAUTION]")
        lines.append(
            "> **One or more runs failed integrity verification. See details below.**"
        )
        lines.append("")
    else:
        lines.append("> [!NOTE]")
        lines.append(
            "> All runs passed SQLite integrity_check, foreign key check, "
            "trigger verification, and index verification."
        )
        lines.append("")

    lines.append(
        "| Scenario | Type | Concurrency | Rep | integrity_check | "
        "foreign_keys | triggers | partial_index | Result |"
    )
    lines.append(
        "|----------|------|-------------|-----|-----------------|"
        "--------------|----------|---------------|--------|"
    )
    for c in all_correctness:
        ok = "✅" if c.get("ok") else "❌"
        # Parse key signals from detail
        int_ok = (
            "✅"
            if any("integrity_check: ok" in d for d in c.get("details", []))
            else "❌"
        )
        fk_ok = (
            "✅"
            if any("foreign_key_check: ok" in d for d in c.get("details", []))
            else "❌"
        )
        trig_ok = (
            "✅" if any("triggers: all" in d for d in c.get("details", [])) else "❌"
        )
        idx_ok = (
            "✅" if any("partial_index:" in d for d in c.get("details", [])) else "❌"
        )
        lines.append(
            f"| {c['scenario']} | {c['concurrency_type']} | {c['concurrency']} | "
            f"{c['repetition']} | {int_ok} | {fk_ok} | {trig_ok} | {idx_ok} | {ok} |"
        )
    lines.append("")

    # Known limitations
    lines.append("## Known Limitations and Reproducibility Notes")
    lines.append("")
    lines.append(
        "1. **Synthetic data only**: All benchmark databases use generated data; "
        "production databases are never touched."
    )
    lines.append(
        "2. **Local machine variability**: Results are sensitive to OS scheduler, "
        "CPU load, and I/O subsystem state."
    )
    lines.append(
        "3. **Warm-up phase**: A warm-up period precedes each measurement window, "
        "but OS-level caching effects may persist."
    )
    lines.append(
        "4. **W2 synthetic contention**: Multiple heartbeat writers are acceptable "
        "for stress testing only. The production single-worker invariant is "
        "not changed."
    )
    lines.append(
        "5. **P1 inter-process overhead**: Process spawn overhead on Linux (fork) is "
        "minimal but not zero; results include process setup overhead in warm-up."
    )
    lines.append(
        "6. **Ownership rejections**: Fencing rejections (JobOwnershipError, "
        "ActiveJobExistsError) are expected and correct — they are NOT counted "
        "as errors."
    )
    lines.append(
        "7. **SQLite version**: Results are specific to the SQLite version shown above."
    )
    lines.append(
        "8. **DELETE journal baseline**: This report covers DELETE mode only. "
        "WAL evaluation is planned for a subsequent phase."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# Quick profile (development):")
    lines.append(".venv/bin/python benchmarks/m11/run_benchmark.py --profile quick")
    lines.append("")
    lines.append("# Full evaluation profile:")
    lines.append(".venv/bin/python benchmarks/m11/run_benchmark.py --profile full")
    lines.append("")
    lines.append("# Single scenario:")
    lines.append(
        ".venv/bin/python benchmarks/m11/run_benchmark.py --scenario W3 --threads 10"
    )
    lines.append("```")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Baseline report written to: %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M11 Phase 2 — SQLite benchmark harness for ModelForge"
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help=(
            "'quick' uses short measurement windows; "
            "'full' uses longer ones. (default: quick)"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=["W1", "W2", "W3", "W4", "W5", "W6", "P1", "all"],
        default="all",
        help="Run a single scenario or all. (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Single thread concurrency level to test (overrides defaults).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Single process concurrency level to test (overrides defaults).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
        help=(
            f"Number of repetitions per scenario/concurrency. (default: {REPETITIONS})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/m11/results"),
        help="Directory to write output files. (default: benchmarks/m11/results)",
    )
    args = parser.parse_args()

    measure_s = (
        MEASUREMENT_SECONDS_QUICK
        if args.profile == "quick"
        else MEASUREMENT_SECONDS_FULL
    )
    warmup_s = WARMUP_SECONDS

    thread_levels = (args.threads,) if args.threads else THREAD_CONCURRENCY_LEVELS
    process_levels = (args.processes,) if args.processes else PROCESS_CONCURRENCY_LEVELS

    if args.scenario == "all":
        scenarios = ["W1", "W2", "W3", "W4", "W5", "W6", "P1"]
    else:
        scenarios = [args.scenario]

    output_dir: Path = args.output_dir
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    env = capture_environment()
    env["profile"] = args.profile
    env["measurement_seconds"] = measure_s
    env["scenarios_run"] = scenarios

    logger.info("=== M11 Phase 2 SQLite Benchmark ===")
    logger.info(
        "Profile: %s | Measurement: %.0fs | Warm-up: %.0fs",
        args.profile,
        measure_s,
        warmup_s,
    )
    logger.info("Scenarios: %s", scenarios)
    logger.info(
        "Thread levels: %s | Process levels: %s",
        thread_levels,
        process_levels,
    )
    logger.info("Repetitions: %d", args.repetitions)
    logger.info("Output dir: %s", output_dir)
    logger.info("")

    t_overall = time.perf_counter()
    all_metrics, all_correctness = run_benchmark_matrix(
        scenarios=scenarios,
        thread_levels=thread_levels,
        process_levels=process_levels,
        repetitions=args.repetitions,
        warmup_s=warmup_s,
        measure_s=measure_s,
    )
    total_elapsed = time.perf_counter() - t_overall

    logger.info("")
    logger.info("=== Benchmark complete in %.1fs ===", total_elapsed)

    # --- Machine-readable JSON output ---
    json_path = output_dir / f"results_{ts_tag}.json"
    payload = {
        "environment": env,
        "results": [m.as_dict() for m in all_metrics],
        "integrity": all_correctness,
        "total_elapsed_s": total_elapsed,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("JSON results written to: %s", json_path)

    # --- Integrity JSON ---
    integrity_path = output_dir / f"integrity_{ts_tag}.json"
    integrity_path.write_text(json.dumps(all_correctness, indent=2), encoding="utf-8")
    logger.info("Integrity report written to: %s", integrity_path)

    # --- Human-readable markdown report ---
    report_path = output_dir / f"baseline_report_{ts_tag}.md"
    generate_baseline_report(
        all_metrics=all_metrics,
        all_correctness=all_correctness,
        env=env,
        profile=args.profile,
        output_path=report_path,
    )

    # --- Print summary to stdout ---
    any_correctness_fail = any(not c.get("ok") for c in all_correctness)
    any_busy = any(m.busy_errors > 0 for m in all_metrics)

    print("\n" + "=" * 70)
    print("M11 Phase 2 SQLite Benchmark — Run Summary")
    print("=" * 70)
    print(f"Total elapsed:    {total_elapsed:.1f}s")
    print(f"Scenarios:        {scenarios}")
    print(f"Total runs:       {len(all_metrics)}")
    print(
        f"Integrity:        "
        f"{'ALL PASS' if not any_correctness_fail else '*** FAILURES DETECTED ***'}"
    )
    print(f"SQLITE_BUSY seen: {'YES' if any_busy else 'none'}")
    print(f"JSON results:     {json_path}")
    print(f"Baseline report:  {report_path}")
    print(f"Integrity detail: {integrity_path}")
    print("=" * 70)

    if any_correctness_fail:
        logger.error(
            "One or more correctness failures detected. Review integrity report."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
