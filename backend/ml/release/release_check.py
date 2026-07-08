#!/usr/bin/env python3

import subprocess
import sys
import time


CHECKS = [
    ("Dataset Audit",
     ["python3", "ml/dataset_tools/dataset_manager.py", "audit"]),

    ("Dataset Verify",
     ["python3", "ml/dataset_tools/dataset_manager.py", "verify"]),

    ("Unknown Videos",
     ["python3", "ml/dataset_tools/dataset_manager.py", "unknown"]),

    ("Duplicate Conflicts",
     ["python3", "ml/dataset_tools/dataset_manager.py", "conflicts"]),

    ("Regression Suite",
     ["python3", "../agents/regression_agent.py"]),

    ("Event Benchmark",
     ["bash", "-lc", "cd /Users/josephkamil/Downloads/formcheck_real_inference_updated/backend && source .venv/bin/activate && PYTHONPATH=/Users/josephkamil/Downloads/formcheck_real_inference_updated/backend python3 ml/event_benchmark.py"]),

    ("Python Compile",
     ["python3", "-m", "py_compile", "app/main.py"]),
]


def run(title, cmd):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    start = time.time()

    result = subprocess.run(cmd)

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ FAILED ({elapsed:.1f}s)")
        return False

    print(f"\n✅ PASSED ({elapsed:.1f}s)")
    return True


def main():

    print("\n")
    print("=" * 70)
    print("FORMCHECK AI RELEASE CHECK")
    print("=" * 70)

    passed = 0

    for title, cmd in CHECKS:

        ok = run(title, cmd)

        if not ok:
            print("\nRELEASE BLOCKED")
            sys.exit(1)

        passed += 1

    print()
    print("=" * 70)
    print(f"ALL {passed}/{len(CHECKS)} CHECKS PASSED")
    print("READY FOR TRAINING / DEPLOYMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()