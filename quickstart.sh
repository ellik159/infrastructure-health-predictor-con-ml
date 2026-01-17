#!/bin/bash
# Quick start script for Infrastructure Health Predictor

set -e

echo "🚀 Infrastructure Health Predictor - Quick Start"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Generate sample data if not exists
if [ ! -f "data/raw/sample_metrics.csv" ]; then
    echo "Generating sample training data..."
    python src/data/generate_sample_data.py
fi

# Train model if not exists
if [ ! -f "data/models/lstm_model.h5" ]; then
    echo "Training model (this may take a few minutes)..."
    python src/models/train_model.py \
        --data data/raw/sample_metrics.csv \
        --epochs 30 \
        --batch-size 32
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API server:"
echo "  source venv/bin/activate"
echo "  python src/api/main.py"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Prometheus at: http://localhost:9090"
echo "Grafana at: http://localhost:3000"
