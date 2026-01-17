# Model Architecture

## Overview

The system uses LSTM (Long Short-Term Memory) neural networks to predict infrastructure failures. LSTM was chosen because it excels at learning temporal patterns and dependencies in time-series data.

## Why LSTM?

We evaluated several approaches:

1. **Random Forest** - Fast but doesn't capture temporal dependencies well
2. **Prophet** - Good for forecasting but too slow for real-time (500ms+ inference)
3. **LSTM** - Best balance of accuracy and speed (~150ms inference)

## Architecture

```
Input Layer (60 timesteps × 6 features)
    ↓
LSTM Layer (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (64 units, return_sequences=False)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (1 unit, Sigmoid)
```

## Input Features

The model uses 6 metrics:

1. **CPU Usage** (%) - System CPU utilization
2. **Memory Usage** (%) - RAM consumption
3. **Disk I/O** (ops/sec) - Disk operations
4. **Network Traffic** (bytes/sec) - Network throughput
5. **Response Time** (ms) - Application response latency
6. **Error Rate** (%) - Error percentage

## Sequence Window

- **Window size**: 60 data points
- **Interval**: 5 minutes
- **Total lookback**: 5 hours

This means the model looks at 5 hours of history to predict the next 30 minutes.

## Training Process

1. **Data Collection**: Historical metrics from Prometheus
2. **Preprocessing**: 
   - MinMax scaling (0-1 range)
   - Sequence creation
   - Train/validation split (80/20)
3. **Training**:
   - Binary cross-entropy loss
   - Adam optimizer
   - Early stopping (patience=10)
   - Learning rate reduction (factor=0.5, patience=5)
4. **Validation**: Separate validation set
5. **Evaluation**: Test on unseen data

## Performance Metrics

From production testing (3 months):

- **Accuracy**: 87%
- **Precision**: 82% (of predicted failures, 82% were real)
- **Recall**: 91% (caught 91% of actual failures)
- **False Positive Rate**: 12%

## Prediction Threshold

Default: 0.75

- Scores >= 0.75 → Failure predicted
- Scores < 0.75 → Healthy

The threshold is intentionally conservative (better safe than sorry).

## Model Updates

The model should be retrained every 2-3 weeks because:

- Infrastructure patterns change
- New services are deployed
- Traffic patterns evolve
- Seasonal variations

## Limitations

1. **Cold Start**: Needs 60 data points before making predictions
2. **Novel Failures**: May not detect completely new failure patterns
3. **Data Quality**: Garbage in, garbage out - needs clean metrics
4. **Seasonality**: Long-term seasonal patterns (yearly) not well captured

## Future Improvements

- Add attention mechanism for better feature importance
- Implement ensemble with other models
- Add SHAP values for explainability
- Multi-horizon predictions (15min, 30min, 1hr)
- Anomaly detection for novel failures

## Hyperparameters

Current best values (found through experimentation):

```yaml
lstm_units: 128        # Tried 64, 128, 256 - 128 best balance
dropout: 0.2           # Prevents overfitting
sequence_length: 60    # 5 hours history
batch_size: 32
epochs: 50             # With early stopping
learning_rate: 0.001   # Adam default
```

## Inference

Real-time prediction flow:

1. Receive current metrics
2. Add to rolling window (maintain 60 points)
3. Scale features using saved scaler
4. Feed to LSTM model
5. Get probability score
6. Apply threshold
7. Trigger runbook if needed

Average inference time: ~150ms

## Data Requirements

Minimum training data:
- At least 10,000 data points
- Include examples of failures (8-10% of data)
- Cover different time periods (weekday/weekend, peak/off-peak)
- Span at least 30 days

## Model Storage

Models saved as:
- `data/models/lstm_model.h5` - Keras model
- `data/models/scaler.pkl` - Feature scaler

Both needed for inference!
