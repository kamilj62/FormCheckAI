import subprocess
import sys

tests = [
    ("Olympic Benchmark", "python3 -m ml.benchmark"),
    ("CoachBench", "python3 -m ml.coachbench"),
]

failed = False

for title, cmd in tests:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        failed = True
        print(f"\nFAILED: {title}")

print("\n" + "=" * 80)

if failed:
    print("QA FAILED")
    sys.exit(1)
else:
    print("ALL QA TESTS PASSED")
