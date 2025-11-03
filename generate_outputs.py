#!/usr/bin/env python3
"""
Script to execute the Jupyter notebook and save all outputs (including visualizations)
so they appear when viewing the notebook on GitHub.

Usage:
    python3 generate_outputs.py

This will:
1. Execute all cells in btc_prophet.ipynb
2. Save all outputs (text, images, etc.) to the notebook
3. Update btc_prophet.ipynb with the outputs

After running this, commit and push to see visualizations on GitHub.
"""

import sys
import os
import json
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import nbformat
        import nbconvert
        return True
    except ImportError:
        print("❌ Missing required packages: nbformat and nbconvert")
        print("   Install with: pip install nbconvert nbformat")
        return False

def execute_notebook():
    """Execute the notebook and save outputs."""
    notebook_path = 'btc_prophet.ipynb'

    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    if not os.path.exists('Crypto_historical_data.csv'):
        print("❌ Data file not found: Crypto_historical_data.csv")
        print("   Cannot execute notebook without data")
        return False

    print("🚀 Executing notebook...")
    print("   This may take several minutes depending on your system...")
    print()

    try:
        # Use jupyter nbconvert to execute the notebook
        result = subprocess.run(
            ['jupyter', 'nbconvert',
             '--to', 'notebook',
             '--execute',
             '--inplace',
             notebook_path],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )

        if result.returncode == 0:
            print("✓ Notebook executed successfully!")
            print("✓ All outputs saved to notebook")
            return True
        else:
            print("❌ Error executing notebook:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Execution timed out (>30 minutes)")
        return False
    except FileNotFoundError:
        print("❌ jupyter command not found")
        print("   Make sure Jupyter is installed: pip install jupyter")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Notebook Output Generator for GitHub")
    print("=" * 60)
    print()

    if not check_dependencies():
        sys.exit(1)

    print("✓ All dependencies available")
    print()

    if execute_notebook():
        print()
        print("=" * 60)
        print("✓ SUCCESS!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Review the notebook to verify outputs are saved")
        print("2. Commit the updated notebook:")
        print("   git add btc_prophet.ipynb")
        print("   git commit -m 'Execute notebook and save visualization outputs'")
        print("3. Push to GitHub:")
        print("   git push origin main")
        print()
        print("After pushing, visualizations will appear on GitHub!")
        return True
    else:
        print()
        print("=" * 60)
        print("❌ FAILED to execute notebook")
        print("=" * 60)
        print()
        print("Alternative approach:")
        print("1. Open the notebook in Jupyter:")
        print("   jupyter notebook btc_prophet.ipynb")
        print("2. Run all cells: Kernel → Restart & Run All")
        print("3. Save the notebook (File → Save)")
        print("4. Commit and push the saved notebook")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
