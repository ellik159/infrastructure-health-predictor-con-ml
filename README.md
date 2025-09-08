# Infrastructure Health Predictor 🚀

Predict infrastructure issues before they happen using ML. This started as a side project to learn LSTM networks and turned into something actually useful.

## What it does

Monitors Kubernetes metrics, system logs, and application metrics to predict potential failures. Uses LSTM networks to find patterns in time-series data that humans might miss.

- Collects metrics from Prometheus
- Trains LSTM models on historical failure data
- Predicts potential issues 5-60 minutes before they happen
- Sends alerts to Slack/Teams
- Has a basic web UI to visualize predictions

## Quick Start ⚡

```bash
# Create virtual environment (python 3.9+)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment (copy example)
cp .env.example .env
# Edit .env with your settings

# Run the development server
uvicorn src.api.main:app --reload
```

The API will be available at http://localhost:8000. There's also a basic web UI at /dashboard.

## Features

- Real-time metric collection from Prometheus 📊
- LSTM-based anomaly detection
- Multi-tenant support (different teams/clusters)
- REST API with FastAPI
- Basic web dashboard
- Slack/Teams integration for alerts
- Model retraining pipeline
- Docker support for deployment

## Architecture

The system has a few main components:

1. **Collector**: Pulls metrics from Prometheus API
2. **Preprocessor**: Cleans and normalizes time-series data
3. **Predictor**: LSTM model that makes predictions
4. **Alert Manager**: Sends notifications when issues predicted
5. **API Layer**: FastAPI app for everything

```
[Prometheus] -> [Collector] -> [Preprocessor] -> [Predictor] -> [Alert Manager]
       ↑                                           ↓
    [Metrics]                              [Predictions] -> [API] -> [Dashboard]
```

## TODO / Known Issues

- [ ] Need to add more test coverage (currently ~60%)
- [ ] The LSTM model sometimes overfits on small datasets
- [ ] Web UI is pretty basic, needs more polish
- [ ] Authentication is minimal, should add proper OAuth
- [ ] Database migrations are manual right now
- [ ] Memory usage can get high with large clusters

## Notes

This was harder than I expected. Time-series forecasting with LSTMs is tricky - getting the right window size and features took a lot of trial and error.

The Prometheus integration was straightforward but handling missing data points was annoying. Ended up using linear interpolation for gaps.

The web dashboard uses Chart.js because it's simple. Could switch to something fancier later but it works for now.

Deployment to Kubernetes was a pain initially - had issues with resource limits and the model loading at startup.

## License

MIT - use it however you want. If you find it useful, let me know!