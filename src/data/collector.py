"""
Metric collector for Prometheus data.
Pulls metrics from Prometheus API and caches them.
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
import json

# TODO: add redis for caching
# import redis

class MetricCollector:
    """Collects metrics from Prometheus and other sources."""
    
    def __init__(self, prometheus_url: Optional[str] = None):
        self.prometheus_url = prometheus_url or os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.cache = {}  # simple in-memory cache for now
        self.cache_ttl = 300  # 5 minutes in seconds
        
        # Mock data for development when Prometheus isn't available
        self.use_mock_data = os.getenv('USE_MOCK_DATA', 'false').lower() == 'true'
        
        # Track available clusters (would come from config or discovery)
        self.known_clusters = ['production', 'staging', 'development']
        
        # API client session
        self.session = requests.Session()
        self.session.timeout = 10
        
        print(f"MetricCollector initialized with Prometheus URL: {self.prometheus_url}")
        if self.use_mock_data:
            print("Warning: Using mock data mode")
    
    def collect_metrics(self, cluster_id: str, time_range: str = "1h") -> List[Dict]:
        """
        Collect metrics for a specific cluster.
        
        Args:
            cluster_id: ID of the cluster to collect metrics for
            time_range: Time range to collect (e.g., "1h", "6h", "24h")
            
        Returns:
            List of metric dictionaries
        """
        cache_key = f"{cluster_id}_{time_range}"
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            print(f"Using cached metrics for {cluster_id}")
            return cached
        
        if self.use_mock_data:
            metrics = self._generate_mock_metrics(cluster_id, time_range)
        else:
            try:
                metrics = self._collect_from_prometheus(cluster_id, time_range)
            except Exception as e:
                print(f"Error collecting from Prometheus: {e}")
                # Fall back to mock data
                metrics = self._generate_mock_metrics(cluster_id, time_range)
        
        # Cache the results
        self._add_to_cache(cache_key, metrics)
        
        return metrics
    
    def _collect_from_prometheus(self, cluster_id: str, time_range: str) -> List[Dict]:
        """Collect actual metrics from Prometheus API."""
        # TODO: implement actual Prometheus queries
        # For now, return mock data
        print(f"Would query Prometheus for cluster {cluster_id}, range {time_range}")
        return self._generate_mock_metrics(cluster_id, time_range)
    
    def _generate_mock_metrics(self, cluster_id: str, time_range: str) -> List[Dict]:
        """Generate mock metrics for development/testing."""
        # Parse time range
        hours = 1
        if time_range.endswith('h'):
            try:
                hours = int(time_range[:-1])
            except:
                pass
        
        # Generate time series data
        metrics = []
        now = datetime.now()
        
        # Different patterns based on cluster
        if cluster_id == 'production':
            # Production tends to have more stable but higher load
            base_cpu = 0.6
            base_memory = 0.7
            variation = 0.2
        elif cluster_id == 'staging':
            # Staging has more variation
            base_cpu = 0.4
            base_memory = 0.5
            variation = 0.3
        else:
            # Development or unknown
            base_cpu = 0.3
            base_memory = 0.4
            variation = 0.4
        
        # Generate data points (one per minute for the time range)
        for i in range(hours * 60):
            timestamp = now - timedelta(minutes=i)
            
            # Add some periodic patterns
            time_of_day = timestamp.hour
            sin_pattern = 0.1 * (1 + np.sin(i / 30))  # 30-minute cycle
            
            # CPU usage with some spikes
            cpu_usage = base_cpu + (variation * np.random.random()) + sin_pattern
            if np.random.random() < 0.05:  # 5% chance of spike
                cpu_usage = min(0.95, cpu_usage + 0.3)
            
            # Memory usage (more stable than CPU)
            memory_usage = base_memory + (variation * 0.5 * np.random.random())
            
            # Network traffic
            network_rx = 1000 + 500 * np.random.random()
            network_tx = 500 + 300 * np.random.random()
            
            # Other metrics
            disk_io = 50 + 30 * np.random.random()
            pod_count = 10 + int(5 * np.random.random())
            error_rate = 0.01 + 0.02 * np.random.random()
            latency = 100 + 50 * np.random.random()
            request_rate = 1000 + 500 * np.random.random()
            queue_length = 5 + 3 * np.random.random()
            
            metrics.append({
                'timestamp': timestamp.isoformat(),
                'cluster_id': cluster_id,
                'cpu_usage': round(cpu_usage, 3),
                'memory_usage': round(memory_usage, 3),
                'network_rx': round(network_rx, 1),
                'network_tx': round(network_tx, 1),
                'disk_io': round(disk_io, 1),
                'pod_count': pod_count,
                'error_rate': round(error_rate, 4),
                'latency': round(latency, 1),
                'request_rate': round(request_rate, 1),
                'queue_length': queue_length
            })
        
        # Reverse to have chronological order (oldest first)
        metrics.reverse()
        return metrics
    
    def get_available_clusters(self) -> List[str]:
        """Get list of available clusters."""
        # TODO: discover clusters dynamically
        # For now, return known clusters
        return self.known_clusters
    
    def get_recent_metrics(self, cluster_id: str, limit: int = 100) -> List[Dict]:
        """Get recent metrics for a cluster (for debugging)."""
        metrics = self.collect_metrics(cluster_id, "1h")
        return metrics[-limit:] if metrics else []
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict]]:
        """Get data from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: List[Dict]):
        """Add data to cache."""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Simple cache eviction if too large
        if len(self.cache) > 100:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def test_prometheus_connection(self) -> bool:
        """Test connection to Prometheus."""
        try:
            response = self.session.get(f"{self.prometheus_url}/api/v1/query?query=up")
            return response.status_code == 200
        except:
            return False
    
    def get_prometheus_metrics(self) -> List[str]:
        """Get list of available metrics from Prometheus."""
        # TODO: implement actual metric discovery
        return [
            'container_cpu_usage_seconds_total',
            'container_memory_usage_bytes',
            'container_network_receive_bytes_total',
            'container_network_transmit_bytes_total',
            'container_fs_reads_bytes_total',
            'container_fs_writes_bytes_total',
            'kube_pod_container_status_restarts_total',
            'kube_pod_status_phase',
            'node_cpu_seconds_total',
            'node_memory_MemFree_bytes'
        ]

# Helper function for numpy operations
try:
    import numpy as np
except ImportError:
    # Mock numpy for development
    print("Warning: numpy not installed, using simple random")
    class MockNumpy:
        def random(self):
            import random
            return random.random()
        
        def sin(self, x):
            import math
            return math.sin(x)
    
    np = MockNumpy()