import subprocess

commands = [
    ["python3", "scripts/core_classifier_check.py"],
    ["python3", "/Users/josephkamil/Desktop/Capstone/FormCheck_Phase_Audit/phase_audit.py"],
]

for cmd in commands:
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print("FAILED:")
        print(result.stderr)
        raise SystemExit(result.returncode)

print("\nQA audit complete.")