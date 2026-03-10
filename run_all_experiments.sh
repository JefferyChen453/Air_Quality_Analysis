#!/bin/bash

# Ensure we are in the project root
export PYTHONPATH=$PYTHONPATH:.

echo "--- Starting Air Quality Analysis Experiments ---"

echo "1. Training Multi-Output LSTM (Joint Pollutant Prediction)..."
uv run python models/train_lstm.py --epochs 100 --lookback 14

echo "2. Evaluating LSTM on Test Set (2024)..."
uv run python models/eval_lstm.py

echo "3. Generating Visualizations..."
uv run python visualize_results.py

echo "--- All experiments completed. Artifacts are in the models/ directory. ---"
