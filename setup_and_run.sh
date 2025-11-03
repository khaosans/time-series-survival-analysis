#!/bin/bash

# Setup and Run Script for BTC Prophet Notebook
# This script sets up the environment and runs the notebook

set -e

echo "=========================================="
echo "BTC Prophet Notebook Setup & Run"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Check if data file exists
if [ ! -f "Crypto_historical_data.csv" ]; then
    echo "❌ ERROR: Crypto_historical_data.csv not found!"
    echo "   Please ensure the data file is in the current directory"
    exit 1
fi

echo "Executing notebook to generate outputs..."
echo "   This will take several minutes..."
echo ""

# Execute the notebook
if command -v jupyter &> /dev/null; then
    jupyter nbconvert --to notebook --execute --inplace btc_prophet.ipynb
    echo ""
    echo "✓ Notebook executed successfully!"
    echo "✓ All outputs saved"
else
    echo "⚠️  Jupyter not found, trying alternative method..."
    python3 generate_outputs.py
fi

echo ""
echo "=========================================="
echo "✓ COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the notebook to verify outputs"
echo "2. Commit and push:"
echo "   git add btc_prophet.ipynb"
echo "   git commit -m 'Execute notebook and save outputs'"
echo "   git push origin main"
echo ""
echo "Visualizations will appear on GitHub!"
