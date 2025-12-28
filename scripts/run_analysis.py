#!/usr/bin/env python3
"""
Run all notebooks in sequence and generate reports.

This script executes all analysis notebooks in the correct order
and saves the outputs for review.

Usage:
    python scripts/run_analysis.py [--output-dir OUTPUT_DIR]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_notebook(notebook_path: Path, output_dir: Path) -> bool:
    """Execute a Jupyter notebook and save the output."""
    output_path = output_dir / f"{notebook_path.stem}_executed.ipynb"
    
    cmd = [
        "jupyter", "nbconvert", 
        "--to", "notebook",
        "--execute",
        "--output", str(output_path),
        str(notebook_path)
    ]
    
    print(f"🔄 Running {notebook_path.name}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"✅ {notebook_path.name} completed successfully")
            return True
        else:
            print(f"❌ {notebook_path.name} failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {notebook_path.name} timed out")
        return False
    except Exception as e:
        print(f"💥 {notebook_path.name} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all analysis notebooks")
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("outputs"),
        help="Directory to save executed notebooks"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Define notebook execution order
    notebooks = [
        "notebooks/01_eda.ipynb",
        "notebooks/02_regression.ipynb", 
        "notebooks/02b_regression_new_customers.ipynb",
        "notebooks/03_classification.ipynb",
        "notebooks/04_clustering.ipynb",
        "notebooks/05_deep_learning.ipynb"
    ]
    
    print(f"🚀 Starting analysis pipeline at {datetime.now()}")
    print(f"📁 Output directory: {args.output_dir}")
    print("-" * 50)
    
    success_count = 0
    total_count = len(notebooks)
    
    for notebook in notebooks:
        notebook_path = Path(notebook)
        if not notebook_path.exists():
            print(f"⚠️  {notebook} not found, skipping...")
            continue
            
        success = run_notebook(notebook_path, args.output_dir)
        if success:
            success_count += 1
    
    print("-" * 50)
    print(f"📊 Analysis complete: {success_count}/{total_count} notebooks successful")
    
    if success_count == total_count:
        print("🎉 All notebooks executed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some notebooks failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()