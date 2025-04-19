import time
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy metric', ['dataset_id', 'model_id', 'metric_type'])
MODEL_PRECISION = Gauge('model_precision', 'Model precision metric', ['dataset_id', 'model_id'])
MODEL_RECALL = Gauge('model_recall', 'Model recall metric', ['dataset_id', 'model_id'])
MODEL_MAP50 = Gauge('model_map50', 'Model mAP50 metric', ['dataset_id', 'model_id'])
MODEL_MAP = Gauge('model_map', 'Model mAP50-95 metric', ['dataset_id', 'model_id'])

TRAINING_DURATION = Histogram('training_duration_seconds', 'Time taken for model training',
                             ['dataset_id', 'model_id'], buckets=(30, 60, 120, 300, 600, 1800, 3600, 7200))
TESTING_DURATION = Histogram('testing_duration_seconds', 'Time taken for model testing',
                            ['dataset_id', 'model_id'], buckets=(10, 30, 60, 120, 300, 600))
INFERENCE_SPEED = Gauge('inference_speed_ms', 'Inference speed in milliseconds', ['dataset_id', 'model_id'])

MODELS_TRAINED_TOTAL = Counter('models_trained_total', 'Total number of models trained', ['dataset_id'])
MODELS_TESTED_TOTAL = Counter('models_tested_total', 'Total number of models tested', ['dataset_id'])
MODEL_TRAINING_STATUS = Gauge('model_training_status', 'Status of model training', 
                             ['dataset_id', 'model_id', 'status'])

class MetricsExporter:
    """Class to export ML metrics to Prometheus"""
    
    def __init__(self, port=8001):
        """Initialize the metrics exporter"""
        self.port = port
        self.server_started = False
        
    def start_server(self):
        """Start the metrics server"""
        if not self.server_started:
            start_http_server(self.port)
            self.server_started = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def record_training_completion(self, dataset_id, model_id, results, duration_seconds):
        """Record metrics after training completion"""
        # Increment counter
        MODELS_TRAINED_TOTAL.labels(dataset_id=dataset_id).inc()
        
        # Record metrics from results
        if results:
            if 'metrics/mAP50(B)' in results:
                MODEL_MAP50.labels(dataset_id=dataset_id, model_id=model_id).set(results['metrics/mAP50(B)'])
                MODEL_ACCURACY.labels(dataset_id=dataset_id, model_id=model_id, metric_type='map50').set(results['metrics/mAP50(B)'])
            
            if 'metrics/mAP50-95(B)' in results:
                MODEL_MAP.labels(dataset_id=dataset_id, model_id=model_id).set(results['metrics/mAP50-95(B)'])
                MODEL_ACCURACY.labels(dataset_id=dataset_id, model_id=model_id, metric_type='map').set(results['metrics/mAP50-95(B)'])
            
            if 'metrics/precision(B)' in results:
                MODEL_PRECISION.labels(dataset_id=dataset_id, model_id=model_id).set(results['metrics/precision(B)'])
                MODEL_ACCURACY.labels(dataset_id=dataset_id, model_id=model_id, metric_type='precision').set(results['metrics/precision(B)'])
            
            if 'metrics/recall(B)' in results:
                MODEL_RECALL.labels(dataset_id=dataset_id, model_id=model_id).set(results['metrics/recall(B)'])
                MODEL_ACCURACY.labels(dataset_id=dataset_id, model_id=model_id, metric_type='recall').set(results['metrics/recall(B)'])
        
        # Record training duration
        TRAINING_DURATION.labels(dataset_id=dataset_id, model_id=model_id).observe(duration_seconds)
        
        # Update status
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='completed').set(1)
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='in_progress').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='failed').set(0)
    
    def record_training_failure(self, dataset_id, model_id):
        """Record metrics after training failure"""
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='completed').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='in_progress').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='failed').set(1)
    
    def record_training_start(self, dataset_id, model_id):
        """Record metrics at training start"""
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='completed').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='in_progress').set(1)
        MODEL_TRAINING_STATUS.labels(dataset_id=dataset_id, model_id=model_id, status='failed').set(0)
    
    def record_test_completion(self, dataset_id, model_id, results, duration_seconds):
        """Record metrics after testing completion"""
        # Increment counter
        MODELS_TESTED_TOTAL.labels(dataset_id=dataset_id).inc()
        
        # Record test metrics
        if results:
            if 'map50' in results:
                MODEL_MAP50.labels(dataset_id=dataset_id, model_id=model_id).set(results['map50'])
            
            if 'map' in results:
                MODEL_MAP.labels(dataset_id=dataset_id, model_id=model_id).set(results['map'])
                
            if 'precision' in results:
                MODEL_PRECISION.labels(dataset_id=dataset_id, model_id=model_id).set(results['precision'])
                
            if 'recall' in results:
                MODEL_RECALL.labels(dataset_id=dataset_id, model_id=model_id).set(results['recall'])
                
            if 'inference_speed' in results:
                INFERENCE_SPEED.labels(dataset_id=dataset_id, model_id=model_id).set(results['inference_speed'])
        
        # Record testing duration
        TESTING_DURATION.labels(dataset_id=dataset_id, model_id=model_id).observe(duration_seconds)


# Create a singleton instance
metrics_exporter = MetricsExporter() 