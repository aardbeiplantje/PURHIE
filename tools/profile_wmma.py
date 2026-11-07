#!/usr/bin/env python3
# profile_wmma.py -- Run a model through the inference engine with profiling enabled,
#                    collect rocprofv3 kernel traces, and analyze WMMA kernel performance.
#
# Usage:
#   python3 profile_wmma.py <model.gguf> <prompt> [--output <dir>] [--reps N]
#
# This script:
#   1. Runs qwen_demo with QWEN_PROFILE=1 and rocprofv3 --kernel-trace
#   2. Parses the results with rocminfo_wmma.py
#   3. Outputs a structured summary for Phase 1 baseline measurements

import subprocess, sys, os, json, argparse, glob, time
from pathlib import Path

def find_binary(name):
    """Find a binary in PATH or relative to this script."""
    # Check common locations
    for candidate in [
        f"../../build/{name}",
        f"../build/{name}",
        f"./build/{name}",
        name,
    ]:
        if os.path.isfile(candidate):
            return candidate
        # Try with .so extension for shared libraries
        so_path = candidate.replace(name, name + ".1")
        if os.path.isfile(so_path):
            return name  # Return base name, LD_PATH will find .so
    return None

def run_profiling(model_path, prompt, output_dir, reps=3):
    """Run the inference engine with profiling enabled."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the demo binary
    demo = find_binary("qwen_demo")
    if not demo:
        # Try to find it via build directory
        build_dirs = [
            "../../build",
            "../build", 
            "./build",
            "../../../../../build"
        ]
        for bd in build_dirs:
            candidates = glob.glob(f"{bd}/**/qwen_demo", recursive=True)
            if candidates:
                demo = candidates[0]
                break
    
    if not demo or not os.path.isfile(demo):
        print(f"ERROR: Could not find qwen_demo binary")
        print(f"Expected in: {', '.join(['../../build', '../build', './build'])}")
        print(f"Run 'make' in the build directory first.")
        sys.exit(1)
    
    print(f"Using binary: {demo}")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
    print(f"Output dir: {output_dir}")
    print(f"Repetitions: {reps}")
    print()
    
    # Check if ROCm is available
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print("WARNING: rocminfo not available or failed")
    except FileNotFoundError:
        print("WARNING: rocminfo not found - GPU profiling may not work")
    
    # Run profiling
    env = os.environ.copy()
    env["QWEN_PROFILE"] = "1"
    
    # rocprofv3 command
    rocprof_cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-d", output_dir,
        "-o", "run",
    ]
    
    demo_cmd = [demo, model_path, prompt]
    
    full_cmd = rocprof_cmd + demo_cmd
    
    print(f"Running: {' '.join(full_cmd)}")
    print()
    
    results = []
    for i in range(reps):
        print(f"--- Run {i+1}/{reps} ---")
        start = time.time()
        
        try:
            result = subprocess.run(
                full_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per run
                cwd=os.path.dirname(demo)
            )
            
            elapsed = time.time() - start
            
            if result.returncode != 0:
                print(f"ERROR: Return code {result.returncode}")
                if result.stderr:
                    print(f"stderr: {result.stderr[:500]}")
                continue
            
            output = result.stdout
            print(f"Completed in {elapsed:.1f}s")
            
            # Extract profile output
            profile_lines = []
            in_profile = False
            for line in output.split('\n'):
                if '[profile]' in line:
                    in_profile = True
                if in_profile:
                    profile_lines.append(line)
            
            results.append({
                "run": i + 1,
                "elapsed_s": round(elapsed, 2),
                "profile": '\n'.join(profile_lines),
                "success": True
            })
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT after 600s")
            results.append({
                "run": i + 1,
                "elapsed_s": 600,
                "profile": "TIMEOUT",
                "success": False
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "run": i + 1,
                "elapsed_s": 0,
                "profile": str(e),
                "success": False
            })
    
    return results

def analyze_results(output_dir, json_output=False):
    """Analyze profiling results using rocminfo_wmma.py"""
    
    # Run rocminfo_wmma.py
    wmma_script = os.path.join(os.path.dirname(__file__), "rocminfo_wmma.py")
    
    try:
        result = subprocess.run(
            ["python3", wmma_script, output_dir, "--all", "--json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"rocminfo_wmma.py error: {result.stderr}")
            return None
    except FileNotFoundError:
        print(f"ERROR: {wmma_script} not found")
        return None
    except Exception as e:
        print(f"ERROR analyzing results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Profile WMMA kernels with rocprofv3")
    parser.add_argument("model", help="Path to GGUF model")
    parser.add_argument("prompt", help="Prompt to run")
    parser.add_argument("--output", default="./profile_output", help="Output directory")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions")
    parser.add_argument("--analyze", action="store_true", help="Also run analysis")
    args = parser.parse_args()
    
    # Run profiling
    results = run_profiling(args.model, args.prompt, args.output, args.reps)
    
    # Print summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r["success"]]
    if successful:
        avg_time = sum(r["elapsed_s"] for r in successful) / len(successful)
        print(f"Successful runs: {len(successful)}/{len(results)}")
        print(f"Average time: {avg_time:.2f}s")
        print()
        
        # Print profile output from best run
        best = min(successful, key=lambda r: r["elapsed_s"])
        print(f"Best run (#{best['run']}, {best['elapsed_s']}s):")
        print("-" * 80)
        print(best["profile"])
    else:
        print("No successful runs.")
    
    # Analyze if requested
    if args.analyze:
        print("\n" + "="*80)
        print("WMMA KERNEL ANALYSIS")
        print("="*80)
        analysis = analyze_results(args.output)
        if analysis:
            print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()
