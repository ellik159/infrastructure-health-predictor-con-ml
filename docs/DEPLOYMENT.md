# Deployment Guide

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- 4GB RAM minimum
- Prometheus instance (or use included docker-compose)

## Local Development Setup

1. **Clone repository**
```bash
git clone <repo-url>
cd infrastructure-health-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate sample data**
```bash
python src/data/generate_sample_data.py
```

5. **Train model**
```bash
python src/models/train_model.py --data data/raw/sample_metrics.csv --epochs 30
```

6. **Start API**
```bash
python src/api/main.py
```

API will be available at http://localhost:8000

## Docker Deployment

The easiest way to deploy everything:

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop services
docker-compose down
```

Services:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Production Deployment

### 1. Environment Setup

Create `.env` file:
```
PROMETHEUS_URL=http://your-prometheus:9090
AUTO_EXECUTE_RUNBOOKS=false
LOG_LEVEL=INFO
```

### 2. Model Training

Train on your actual data:
```bash
# Collect metrics from Prometheus
python src/data/collect_metrics.py --duration 60 --output data/raw/prod_metrics.csv

# Train model
python src/models/train_model.py \
  --data data/raw/prod_metrics.csv \
  --epochs 50 \
  --batch-size 64
```

### 3. Deploy with Docker

```bash
docker build -t infrastructure-health-predictor -f docker/Dockerfile .

docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  --name health-predictor \
  infrastructure-health-predictor
```

### 4. Configure Prometheus

Add to your `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'health-predictor'
    static_configs:
      - targets: ['health-predictor:8000']
    metrics_path: '/metrics'
```

### 5. Setup Grafana

1. Login to Grafana (admin/admin)
2. Add Prometheus datasource
3. Import dashboards from `grafana/dashboards/`

## API Usage

### Make prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_usage": 85.5,
    "memory_usage": 78.2,
    "disk_io": 250.0,
    "network_traffic": 2500.0,
    "response_time": 150.0,
    "error_rate": 3.5
  }'
```

### Check health:
```bash
curl http://localhost:8000/health
```

## Monitoring

View Prometheus metrics:
```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `predictions_total` - Total predictions made
- `failure_predictions_total` - Failures predicted
- `prediction_duration_seconds` - Prediction latency

## Troubleshooting

**Model not loading:**
- Check that model file exists in `data/models/lstm_model.h5`
- Retrain if necessary

**High memory usage:**
- Reduce batch size in config
- Limit sequence length
- Use smaller LSTM units

**Slow predictions:**
- Check model inference time in logs
- Consider model optimization (quantization)
- Use GPU if available

## Configuration

Edit `config/config.yaml` for:
- Model parameters (LSTM units, sequence length)
- API settings (port, workers)
- Prometheus connection
- Runbook settings

## Runbook Development

Create custom runbooks in `src/runbooks/`:

```python
def execute(context):
    """Your remediation logic"""
    metrics = context['metrics']
    
    # Do something based on metrics
    
    return {
        "status": "success",
        "actions": ["action1", "action2"]
    }
```

Enable auto-execution in config:
```yaml
runbooks:
  enabled: true
  auto_execute: true  # Use with caution!
```

## Updating the Model

Retrain periodically (recommended every 2-3 weeks):

```bash
# Collect fresh data
python src/data/collect_metrics.py --duration 30

# Retrain
python src/models/train_model.py --data data/raw/metrics.csv

# Restart API to load new model
docker-compose restart api
```

## Security Notes

- Don't commit `.env` files
- Rotate Grafana admin password
- Use authentication for production API
- Review runbooks before enabling auto-execution
- Keep model files secure (they contain info about your infrastructure)
