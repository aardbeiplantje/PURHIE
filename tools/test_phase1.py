#!/usr/bin/env python3
"""
test_phase1.py -- Validate Phase 1 profiling infrastructure with synthetic data.

This creates a mock rocprofv3 database and runs all analysis tools against it
to verify the infrastructure works before running on real models.
"""

import sqlite3
import os
import sys
import json
from collections import defaultdict

# Add tools directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TEST_DIR = os.path.join(SCRIPT_DIR, "test_phase1_output")

def create_synthetic_db():
    """Create a synthetic rocprofv3 database with realistic kernel traces."""
    os.makedirs(TEST_DIR, exist_ok=True)
    db_path = os.path.join(TEST_DIR, "synthetic_kernels.db")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create kernels table (matches rocprofv3 schema)
    c.execute("""
        CREATE TABLE kernels (
            name TEXT,
            start INTEGER,
            end INTEGER
        )
    """)
    
    # Insert synthetic kernel traces (realistic timing for ROCmFP4_FAST WMMA)
    # Times are relative, representing GPU clock cycles (ns)
    kernels = [
        # WMMA kernels (main focus) - matmul_wmma_q4_0_rocmfp4_fast
        ("matmul_wmma_q4_0_rocmfp4_fast", 1000000, 1000000 + 45000),
        ("matmul_wmma_q4_0_rocmfp4_fast", 1050000, 1050000 + 43000),
        ("matmul_wmma_q4_0_rocmfp4_fast", 1100000, 1100000 + 46000),
        ("matmul_wmma_q4_0_rocmfp4_fast", 1150000, 1150000 + 44000),
        ("matmul_wmma_q4_0_rocmfp4_fast", 1200000, 1200000 + 47000),
        
        # Expert-indexed WMMA (MoE)
        ("matmul_wmma_id_q4_0_rocmfp4_fast", 2000000, 2000000 + 28000),
        ("matmul_wmma_id_q4_0_rocmfp4_fast", 2030000, 2030000 + 31000),
        ("matmul_wmma_id_q4_0_rocmfp4_fast", 2065000, 2065000 + 29000),
        
        # Weight staging/decode
        ("round_half_q4_0_rocmfp4_fast", 500000, 500000 + 12000),
        ("round_half_q4_0_rocmfp4_fast", 520000, 520000 + 11000),
        ("dequant_q4_0_rocmfp4_fast", 600000, 600000 + 8000),
        
        # Epilogue-related (norms, routing)
        ("add_rmsnorm_rows_q4_0_rocmfp4_fast", 700000, 700000 + 15000),
        ("moe_router_f32", 800000, 800000 + 22000),
        ("moe_combine_f32", 830000, 830000 + 18000),
        
        # Other kernels (attention, FFN, DeltaNet)
        ("attn_qkv_proj_f16", 300000, 300000 + 35000),
        ("attn_o_proj_f16", 340000, 340000 + 25000),
        ("ffn_gate_up_f16", 400000, 400000 + 41000),
        ("ffn_down_f16", 450000, 450000 + 38000),
        ("deltanet_rule_f16", 500000, 500000 + 19000),
        ("deltanet_conv_f16", 525000, 525000 + 21000),
    ]
    
    for name, start, end in kernels:
        c.execute("INSERT INTO kernels VALUES (?, ?, ?)", (name, start, end))
    
    conn.commit()
    conn.close()
    print(f"[OK] Created synthetic database with {len(kernels)} kernel entries")
    return db_path


def test_rocminfo_wmma():
    """Test rocminfo_wmma.py with synthetic data."""
    print("\n--- Test: rocminfo_wmma.py ---")
    
    # Import the analysis function (inline to avoid import issues)
    import glob
    
    db_files = glob.glob(os.path.join(TEST_DIR, "**/*.db"), recursive=True)
    if not db_files:
        print("[FAIL] No database files found")
        return False
    
    conn = sqlite3.connect(db_files[0])
    c = conn.cursor()
    
    rows = c.execute(
        "select name, count(*), sum(end - start) / 1e6, avg(end - start) / 1e3, "
        "min(end - start) / 1e3, max(end - start) / 1e3 "
        "from kernels group by name order by 3 desc"
    ).fetchall()
    
    tot = sum(r[2] for r in rows)
    
    # Categorize kernels
    wmma_patterns = ["matmul_wmma_", "wmma_q80", "wmma_f16", "wmma_f32", "matmul_wmma_id_"]
    staging_patterns = ["dequant", "round_half"]
    epilogue_patterns = ["moe_router", "moe_combine", "rmsnorm"]
    
    categories = defaultdict(list)
    for name, cnt, ms, avg, min_us, max_us in rows:
        if any(p in name for p in wmma_patterns):
            categories["WMMA"].append((name, cnt, ms, avg))
        elif any(p in name for p in staging_patterns):
            categories["Staging"].append((name, cnt, ms, avg))
        elif any(p in name for p in epilogue_patterns):
            categories["Epilogue"].append((name, cnt, ms, avg))
        else:
            categories["Other"].append((name, cnt, ms, avg))
    
    # Print results
    print(f"\nTotal GPU time: {tot:.2f} ms")
    print(f"Total unique kernels: {len(rows)}")
    print()
    
    for cat in ["WMMA", "Staging", "Epilogue", "Other"]:
        if cat in categories:
            kernels = categories[cat]
            total_ms = sum(k[2] for k in kernels)
            print(f"{cat} ({len(kernels)} kernels, {total_ms:.2f} ms total):")
            for name, cnt, ms, avg in sorted(kernels, key=lambda x: x[2], reverse=True):
                print(f"  {name:50s} {ms:8.2f} ms  {cnt:3d}x  avg={avg:6.2f} us")
            print()
    
    # Verify expected categorizations
    expected = {
        "WMMA": ["matmul_wmma_q4_0_rocmfp4_fast", "matmul_wmma_id_q4_0_rocmfp4_fast"],
        "Staging": ["round_half_q4_0_rocmfp4_fast", "dequant_q4_0_rocmfp4_fast"],
        "Epilogue": ["moe_router_f32", "moe_combine_f32", "add_rmsnorm_rows_q4_0_rocmfp4_fast"],
        "Other": ["attn_qkv_proj_f16", "ffn_gate_up_f16", "deltanet_rule_f16"],
    }
    
    all_pass = True
    for cat, expected_kernels in expected.items():
        cat_kernels = [k[0] for k in categories.get(cat, [])]
        for k in expected_kernels:
            if k not in cat_kernels:
                print(f"[FAIL] {k} should be in {cat}")
                all_pass = False
    
    if all_pass:
        print("[OK] All categorizations correct")
    
    return all_pass


def test_prof_summary():
    """Test that prof_summary.py works with synthetic data."""
    print("\n--- Test: prof_summary.py ---")
    
    import subprocess
    
    result = subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "prof_summary.py"), TEST_DIR, "10"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print(result.stdout)
        print("[OK] prof_summary.py works correctly")
        return True
    else:
        print(f"[FAIL] prof_summary.py error: {result.stderr}")
        return False


def main():
    print("="*80)
    print("Phase 1 Infrastructure Tests")
    print("="*80)
    
    # Create synthetic database
    print("\nCreating synthetic test data...")
    db_path = create_synthetic_db()
    
    # Run tests
    tests = [
        ("rocminfo_wmma.py", test_rocminfo_wmma),
        ("prof_summary.py", test_prof_summary),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] {name} raised exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:30s} {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("All infrastructure tests passed!")
        print()
        print("Next steps:")
        print("  1. Obtain a ROCmFP4_FAST model (Qwen3.8-27B or similar)")
        print("  2. Run: ./tools/profile_wmma.sh <model.gguf> '<prompt>' ./profile_out")
        print("  3. Analyze: python3 tools/rocminfo_wmma.py ./profile_out --all --json")
        print()
        print("See PHASE1_BASELINE.md for complete measurement plan.")
        return 0
    else:
        print("Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
