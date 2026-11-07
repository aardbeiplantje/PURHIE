#!/usr/bin/env python3
# rocminfo_wmma.py -- Filter and analyze WMMA kernel traces from rocprofv3 output.
#
#   rocprofv3 --kernel-trace -d <dir> -o run -- <command>
#   rocminfo_wmma.py <dir> [--all] [--epilogue] [--staging]
#
# Options:
#   --all       Show all kernels, not just WMMA
#   --epilogue  Focus on epilogue-related operations
#   --staging   Focus on weight staging/decode operations
#   --json      Output JSON for programmatic analysis
#
# Filters for WMMA-specific kernels and categorizes them:
#   - Weight staging: load/store from global memory
#   - WMMA compute: v_wmma_i32_16x16x16_iu8_w32 launches
#   - Epilogue: scale application, output writing
#   - Synchronization: __syncthreads() barriers

import glob, sqlite3, json, sys, argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Analyze WMMA kernel traces from rocprofv3")
    parser.add_argument("dir", help="rocprofv3 output directory")
    parser.add_argument("--all", action="store_true", help="Show all kernels")
    parser.add_argument("--epilogue", action="store_true", help="Focus on epilogue operations")
    parser.add_argument("--staging", action="store_true", help="Focus on staging/decode operations")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    d = args.dir
    db_files = glob.glob(d + "/**/*.db", recursive=True)
    if not db_files:
        sys.exit(f"no results database under {d}")

    c = sqlite3.connect(db_files[0])

    # Get all kernels
    rows = c.execute(
        "select name, count(*), sum(end - start) / 1e6, avg(end - start) / 1e3, "
        "min(end - start) / 1e3, max(end - start) / 1e3 "
        "from kernels group by name order by 3 desc"
    ).fetchall()

    tot = sum(r[2] for r in rows)
    n_kernels = c.execute("select count(*) from kernels").fetchone()[0]

    # Categorize kernels
    wmma_kernels = []
    staging_kernels = []
    epilogue_kernels = []
    other_kernels = []

    wmma_patterns = [
        "matmul_wmma_",
        "wmma_q80",
        "wmma_f16",
        "wmma_f32",
        "matmul_wmma_id_",
    ]
    staging_patterns = [
        "load_chunk",
        "store_chunk",
        "rocmfp4_fast_quad",
        "rocmfp4_fast_q8_block",
        "wmma_q80",
        "WStageFast",
        "dequant",
        "round_half",
    ]
    epilogue_patterns = [
        "token_scales",
        "mv_rocmfp4_fast",
        "epilogue",
        "scale",
    ]

    for name, cnt, ms, avg, min_us, max_us in rows:
        is_wmma = any(p in name for p in wmma_patterns)
        is_staging = any(p in name for p in staging_patterns)
        is_epilogue = any(p in name for p in epilogue_patterns)

        if args.all:
            entry = {
                "name": name,
                "count": cnt,
                "total_ms": round(ms, 2),
                "pct": round(100 * ms / tot, 2) if tot > 0 else 0,
                "avg_us": round(avg, 2),
                "min_us": round(min_us, 2),
                "max_us": round(max_us, 2),
            }

            if is_wmma:
                entry["category"] = "WMMA"
                wmma_kernels.append(entry)
            elif is_staging:
                entry["category"] = "Staging"
                staging_kernels.append(entry)
            elif is_epilogue:
                entry["category"] = "Epilogue"
                epilogue_kernels.append(entry)
            else:
                entry["category"] = "Other"
                other_kernels.append(entry)
        else:
            if is_wmma:
                entry = {
                    "name": name,
                    "count": cnt,
                    "total_ms": round(ms, 2),
                    "pct": round(100 * ms / tot, 2) if tot > 0 else 0,
                    "avg_us": round(avg, 2),
                    "min_us": round(min_us, 2),
                    "max_us": round(max_us, 2),
                }
                wmma_kernels.append(entry)

    if args.json:
        output = {
            "total_gpu_ms": round(tot, 2),
            "total_kernels": n_kernels,
            "wmma_kernels": wmma_kernels,
            "staging_kernels": staging_kernels,
            "epilogue_kernels": epilogue_kernels,
            "other_kernels": other_kernels,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"=== ROCMin WMMA Analysis (total GPU: {tot:.1f} ms, {n_kernels} kernels) ===\n")

        if wmma_kernels:
            print("--- WMMA Kernels ---")
            print(f"{'Total ms':>10} {'%':>6} {'Count':>8} {'Avg us':>10} {'Min us':>10} {'Max us':>10}  Kernel")
            print("-" * 100)
            for k in sorted(wmma_kernels, key=lambda x: x["total_ms"], reverse=True):
                print(f"{k['total_ms']:10.2f} {k['pct']:5.1f}% {k['count']:8d} {k['avg_us']:10.2f} {k['min_us']:10.2f} {k['max_us']:10.2f}  {k['name']}")
            print()

        if args.all and staging_kernels:
            print("--- Weight Staging / Decode ---")
            print(f"{'Total ms':>10} {'%':>6} {'Count':>8} {'Avg us':>10}  Kernel")
            print("-" * 80)
            for k in sorted(staging_kernels, key=lambda x: x["total_ms"], reverse=True):
                print(f"{k['total_ms']:10.2f} {k['pct']:5.1f}% {k['count']:8d} {k['avg_us']:10.2f}  {k['name']}")
            print()

        if args.all and epilogue_kernels:
            print("--- Epilogue Operations ---")
            print(f"{'Total ms':>10} {'%':>6} {'Count':>8} {'Avg us':>10}  Kernel")
            print("-" * 80)
            for k in sorted(epilogue_kernels, key=lambda x: x["total_ms"], reverse=True):
                print(f"{k['total_ms']:10.2f} {k['pct']:5.1f}% {k['count']:8d} {k['avg_us']:10.2f}  {k['name']}")
            print()

        if args.all and other_kernels:
            print("--- Other Kernels (top 20) ---")
            print(f"{'Total ms':>10} {'%':>6} {'Count':>8} {'Avg us':>10}  Kernel")
            print("-" * 80)
            for k in sorted(other_kernels, key=lambda x: x["total_ms"], reverse=True)[:20]:
                print(f"{k['total_ms']:10.2f} {k['pct']:5.1f}% {k['count']:8d} {k['avg_us']:10.2f}  {k['name']}")

        # WMMA-specific metrics
        if wmma_kernels:
            total_wmma_ms = sum(k["total_ms"] for k in wmma_kernels)
            total_wmma_launches = sum(k["count"] for k in wmma_kernels)
            print(f"\n=== WMMA Summary ===")
            print(f"Total WMMA time: {total_wmma_ms:.2f} ms ({100 * total_wmma_ms / tot:.1f}% of GPU time)")
            print(f"Total WMMA launches: {total_wmma_launches}")
            print(f"Average per launch: {total_wmma_ms / total_wmma_launches * 1000:.2f} us" if total_wmma_launches > 0 else "N/A")


if __name__ == "__main__":
    main()
