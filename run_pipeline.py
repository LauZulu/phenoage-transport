"""Reproduce the full study end to end: data -> adaptation -> model -> ONNX.

    python run_pipeline.py
"""
import os
import subprocess
import sys

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

steps = [
    ("Generate cohorts (source + altitude target)", "-m src.data_gen"),
    ("Fit domain-adaptation weights (uLSIF / KLIEP)", "-m src.domain_adapt"),
    ("Estimate biomarker networks (EBICglasso)", "-m src.graphical_model"),
    ("Train, evaluate transportability, export ONNX", "-m src.train"),
]

for i, (label, cmd) in enumerate(steps, 1):
    print(f"\n{'='*64}\n[{i}/{len(steps)}] {label}\n{'='*64}")
    r = subprocess.run([sys.executable, *cmd.split()])
    if r.returncode != 0:
        print(f"Step failed: {label}")
        sys.exit(r.returncode)

print("\nDone. Artifacts in models/ — serve with: vercel dev")
