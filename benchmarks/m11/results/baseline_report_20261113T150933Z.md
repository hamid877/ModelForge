# M11 Phase 2 — SQLite Baseline Benchmark Report

> Generated: 2026-11-13T15:09:33.059862+00:00  
> Profile: **quick**  
> Journal mode: **DELETE**  
> Busy timeout: **5000 ms**  
> Warm-up: 2.0 s | Measurement: varies by profile  
> Repetitions: 3  

## Environment

| Key | Value |
|-----|-------|
| Python | `3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]` |
| SQLite | `3.45.1` |
| Platform | `Linux-7.0.0-30-generic-x86_64-with-glibc2.39` |
| CPU cores | 6 |

> [!IMPORTANT]
> This is a baseline measurement only. WAL mode and other optimisations
> have NOT been applied. These results characterise DELETE-mode behaviour.

## Results Summary

All metrics are averaged across repetitions unless noted.  
**Busy errors** = `sqlite3.OperationalError: database is locked`.  
**Ownership rejections** = expected fencing failures (not counted as errors).  

### W1: Job claim contention

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (thread) | 9.3 | 63.75 | 114.00 | 163.47 | 0 | 0.000% | 0 | 0 | ✅ |
| 5× (thread) | 11.4 | 63.72 | 504.91 | 1528.98 | 12 | 7.018% | 11 | 0 | ✅ |
| 10× (thread) | 11.4 | 96.03 | 3147.32 | 3984.33 | 23 | 13.450% | 7 | 0 | ✅ |
| 25× (thread) | 19.3 | 75.28 | 3212.64 | 3653.24 | 67 | 23.103% | 51 | 0 | ✅ |

### W2: Heartbeat write contention

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (thread) | 13.2 | 68.93 | 116.04 | 136.52 | 0 | 0.000% | 0 | 0 | ✅ |
| 5× (thread) | 27.2 | 71.65 | 3663.36 | 5013.62 | 0 | 0.000% | 200 | 0 | ✅ |
| 10× (thread) | 20.9 | 74.77 | 5007.12 | 5008.31 | 0 | 0.000% | 86 | 0 | ✅ |
| 25× (thread) | 34.9 | 82.50 | 5009.04 | 5011.21 | 2 | 0.382% | 237 | 0 | ✅ |

### W3: Concurrent drift assessments

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (thread) | 12.7 | 71.59 | 120.38 | 168.16 | 0 | 0.000% | 0 | 0 | ✅ |
| 5× (thread) | 14.5 | 73.51 | 1398.53 | 2329.68 | 12 | 5.505% | 0 | 0 | ✅ |
| 10× (thread) | 16.8 | 72.36 | 2832.87 | 3573.67 | 24 | 9.524% | 0 | 0 | ✅ |
| 25× (thread) | 22.3 | 88.89 | 4099.13 | 4572.13 | 67 | 20.000% | 0 | 0 | ✅ |

### W4: Combined write-heavy workload

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (thread) | 12.9 | 75.10 | 116.75 | 162.74 | 0 | 0.000% | 0 | 0 | ✅ |
| 5× (thread) | 15.3 | 71.09 | 918.37 | 2448.43 | 12 | 5.240% | 0 | 0 | ✅ |
| 10× (thread) | 16.5 | 80.48 | 2477.31 | 3354.87 | 25 | 10.121% | 0 | 0 | ✅ |
| 25× (thread) | 21.4 | 90.90 | 4444.51 | 6163.18 | 58 | 18.069% | 0 | 0 | ✅ |

### W5: Read-heavy metrics queries during writes

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (thread) | 13.1 | 75.05 | 132.66 | 819.39 | 3 | 1.523% | 0 | 0 | ✅ |
| 5× (thread) | 14.0 | 78.53 | 2186.73 | 3964.15 | 7 | 3.333% | 0 | 0 | ✅ |
| 10× (thread) | 15.6 | 86.88 | 3104.69 | 5166.12 | 15 | 6.410% | 0 | 0 | ✅ |
| 25× (thread) | 22.3 | 87.42 | 4389.22 | 5899.48 | 56 | 16.766% | 0 | 0 | ✅ |

### W6: Promotion audit writes

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (thread) | 12.6 | 75.80 | 126.73 | 161.52 | 0 | 0.000% | 0 | 0 | ✅ |
| 5× (thread) | 13.3 | 82.04 | 2971.43 | 4855.16 | 4 | 2.010% | 0 | 0 | ✅ |
| 10× (thread) | 15.5 | 87.15 | 2560.81 | 3111.17 | 26 | 11.159% | 0 | 0 | ✅ |
| 25× (thread) | 20.4 | 92.24 | 5778.80 | 7863.33 | 50 | 16.340% | 0 | 0 | ✅ |

### P1: API/worker-style cross-process contention

| Concurrency | Ops/s | P50 ms | P95 ms | P99 ms | Busy | Busy% | Ownership | Other | Integrity |
|-------------|-------|--------|--------|--------|------|-------|-----------|-------|-----------|
| 1× (process) | 9.5 | 77.40 | 128.96 | 151.81 | 0 | 0.000% | 47 | 0 | ✅ |
| 2× (process) | 10.0 | 73.40 | 103.18 | 110.14 | 2 | 1.333% | 49 | 0 | ✅ |
| 4× (process) | 13.8 | 72.87 | 479.97 | 1199.50 | 6 | 2.899% | 71 | 0 | ✅ |

## Integrity Verification

> [!NOTE]
> All runs passed SQLite integrity_check, foreign key check, trigger verification, and index verification.

| Scenario | Type | Concurrency | Rep | integrity_check | foreign_keys | triggers | partial_index | Result |
|----------|------|-------------|-----|-----------------|--------------|----------|---------------|--------|
| W1 | thread | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 5 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 5 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 5 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 10 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 10 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 10 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 25 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 25 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W1 | thread | 25 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 5 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 5 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 5 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 10 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 10 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 10 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 25 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 25 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W2 | thread | 25 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 5 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 5 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 5 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 10 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 10 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 10 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 25 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 25 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W3 | thread | 25 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 5 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 5 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 5 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 10 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 10 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 10 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 25 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 25 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4 | thread | 25 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 5 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 5 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 5 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 10 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 10 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 10 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 25 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 25 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W5 | thread | 25 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 5 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 5 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 5 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 10 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 10 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 10 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 25 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 25 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| W6 | thread | 25 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 1 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 1 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 1 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 2 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 2 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 2 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 4 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 4 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| P1 | process | 4 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |

## Known Limitations and Reproducibility Notes

1. **Synthetic data only**: All benchmark databases use generated data; production databases are never touched.
2. **Local machine variability**: Results are sensitive to OS scheduler, CPU load, and I/O subsystem state.
3. **Warm-up phase**: A warm-up period precedes each measurement window, but OS-level caching effects may persist.
4. **W2 synthetic contention**: Multiple heartbeat writers are acceptable for stress testing only. The production single-worker invariant is not changed.
5. **P1 inter-process overhead**: Process spawn overhead on Linux (fork) is minimal but not zero; results include process setup overhead in warm-up.
6. **Ownership rejections**: Fencing rejections (JobOwnershipError, ActiveJobExistsError) are expected and correct — they are NOT counted as errors.
7. **SQLite version**: Results are specific to the SQLite version shown above.
8. **DELETE journal baseline**: This report covers DELETE mode only. WAL evaluation is planned for a subsequent phase.

## Reproduction

```bash
# Quick profile (development):
uv run python benchmarks/m11/run_benchmark.py --profile quick

# Full evaluation profile:
uv run python benchmarks/m11/run_benchmark.py --profile full

# Single scenario:
uv run python benchmarks/m11/run_benchmark.py --scenario W3 --threads 10
```