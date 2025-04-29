import logging
import os
from prometheus_client import (
    Gauge, Counter, Histogram, REGISTRY, start_http_server
)

# Configure logging
logger = logging.getLogger(__name__)

# Define metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy value', ['dataset_id', 'task_id', 'metric_type'])
MODEL_PRECISION = Gauge('model_precision', 'Model precision', ['dataset_id', 'task_id'])
MODEL_RECALL = Gauge('model_recall', 'Model recall', ['dataset_id', 'task_id'])
MODEL_MAP50 = Gauge('model_map50', 'Mean Average Precision @ IoU=0.5', ['dataset_id', 'task_id'])
MODEL_MAP = Gauge('model_map', 'Mean Average Precision [0.5:0.95]', ['dataset_id', 'task_id'])

TRAINING_DURATION = Histogram(
    'training_duration_seconds', 'Training duration',
    ['dataset_id', 'task_id'], buckets=(30, 60, 120, 300, 600, 1800, 3600, 7200)
)
TESTING_DURATION = Histogram(
    'testing_duration_seconds', 'Testing duration',
    ['dataset_id', 'task_id'], buckets=(10, 30, 60, 120, 300, 600)
)
INFERENCE_SPEED = Gauge('inference_speed_ms', 'Inference speed per image in ms', ['dataset_id', 'task_id'])

MODELS_TRAINED_TOTAL = Counter('models_trained_total', 'Number of models trained', ['dataset_id'])
MODELS_TESTED_TOTAL = Counter('models_tested_total', 'Number of models tested', ['dataset_id'])

MODEL_TRAINING_STATUS = Gauge('model_training_status', 'Training status of model',
                             ['dataset_id', 'task_id', 'status'])

class MetricsExporter:
    def __init__(self, port=9090):
        """Initialize metrics exporter with HTTP server"""
        self.port = port

    def start_server(self):
        """Start HTTP server for metrics"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            raise

    def _safe_metric_update(self, metric, value, labels=None):
        """Safely update a metric with error handling"""
        try:
            if labels:
                metric.labels(*labels).set(value)
            else:
                metric.set(value)
        except Exception as e:
            logger.warning(f"Failed to update metric {metric._name}: {str(e)}")

    def record_training_start(self, dataset_id, task_id):
        """Record training start status"""
        self._safe_metric_update(MODEL_TRAINING_STATUS, 1, [dataset_id, task_id, 'in_progress'])
        self._safe_metric_update(MODEL_TRAINING_STATUS, 0, [dataset_id, task_id, 'completed'])
        self._safe_metric_update(MODEL_TRAINING_STATUS, 0, [dataset_id, task_id, 'failed'])

    def record_training_failure(self, dataset_id, task_id):
        """Record training failure status"""
        self._safe_metric_update(MODEL_TRAINING_STATUS, 0, [dataset_id, task_id, 'in_progress'])
        self._safe_metric_update(MODEL_TRAINING_STATUS, 0, [dataset_id, task_id, 'completed'])
        self._safe_metric_update(MODEL_TRAINING_STATUS, 1, [dataset_id, task_id, 'failed'])

    def record_training_completion(self, dataset_id, task_id, results, duration_seconds):
        """Record training completion with metrics"""
        try:
            MODELS_TRAINED_TOTAL.labels(dataset_id=dataset_id).inc()

            if results:
                metrics_map = {
                    'metrics/mAP50(B)': (MODEL_MAP50, 'map50'),
                    'metrics/mAP50-95(B)': (MODEL_MAP, 'map'),
                    'metrics/precision(B)': (MODEL_PRECISION, None),
                    'metrics/recall(B)': (MODEL_RECALL, None)
                }

                for result_key, (metric, acc_type) in metrics_map.items():
                    if result_key in results:
                        value = results[result_key]
                        self._safe_metric_update(metric, value, [dataset_id, task_id])
                        if acc_type:
                            self._safe_metric_update(MODEL_ACCURACY, value, 
                                                   [dataset_id, task_id, acc_type])

            self._safe_metric_update(TRAINING_DURATION, duration_seconds, [dataset_id, task_id])
            
            # Update status
            self._safe_metric_update(MODEL_TRAINING_STATUS, 1, [dataset_id, task_id, 'completed'])
            self._safe_metric_update(MODEL_TRAINING_STATUS, 0, [dataset_id, task_id, 'in_progress'])
            self._safe_metric_update(MODEL_TRAINING_STATUS, 0, [dataset_id, task_id, 'failed'])

        except Exception as e:
            logger.error(f"Error recording training completion: {str(e)}")

    def record_epoch_metrics(self, dataset_id, task_id, epoch, metrics):
        """Record metrics for a single training epoch"""
        try:
            metric_mapping = {
                'mAP50(B)': (MODEL_MAP50, 'map50'),
                'mAP50-95(B)': (MODEL_MAP, 'map'),
                'precision(B)': (MODEL_PRECISION, None),
                'recall(B)': (MODEL_RECALL, None)
            }

            for metric_name, value in metrics.items():
                if metric_name.startswith('metrics/'):
                    key = metric_name.split('/')[1] 
                    if key in metric_mapping:
                        metric, acc_type = metric_mapping[key]
                        self._safe_metric_update(metric, value, [dataset_id, task_id])
                        if acc_type:
                            self._safe_metric_update(MODEL_ACCURACY, value, 
                                                [dataset_id, task_id, acc_type])
        except Exception as e:
            logger.error(f"Error recording epoch metrics: {str(e)}")

    def record_test_completion(self, dataset_id, task_id, results, duration_seconds):
        """Record testing completion with metrics"""
        try:
            MODELS_TESTED_TOTAL.labels(dataset_id=dataset_id).inc()

            metric_mapping = {
                'map50': MODEL_MAP50,
                'map': MODEL_MAP,
                'precision': MODEL_PRECISION,
                'recall': MODEL_RECALL,
                'inference_speed': INFERENCE_SPEED
            }

            for key, metric in metric_mapping.items():
                if key in results:
                    self._safe_metric_update(metric, results[key], [dataset_id, task_id])

            self._safe_metric_update(TESTING_DURATION, duration_seconds, [dataset_id, task_id])

        except Exception as e:
            logger.error(f"Error recording test completion: {str(e)}")

# Singleton instance (metrics will be available at http://localhost:9090/metrics)
metrics_exporter = MetricsExporter(port=9090)