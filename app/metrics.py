import logging
import os
from prometheus_client import (
    Gauge, Counter, Histogram, REGISTRY, push_to_gateway, start_http_server
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
                              ['dataset_id', 'task_id', 'status'])  # status = in_progress, completed, failed

PUSHGATEWAY_URL = os.getenv('PUSHGATEWAY_URL', 'pushgateway:9091')

class MetricsExporter:
    def __init__(self):
        logger.info("MetricsExporter initialized")

    def start_server(self, port=8000):
        """
        Start the Prometheus metrics HTTP server.
        """
        logger.info(f"Starting Prometheus metrics server on port {port}")
        start_http_server(port)

    def record_training_start(self, dataset_id, task_id):
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'in_progress').set(1)
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'completed').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'failed').set(0)

    def record_training_failure(self, dataset_id, task_id):
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'in_progress').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'completed').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'failed').set(1)

    def record_training_completion(self, dataset_id, task_id, results, duration_seconds):
        MODELS_TRAINED_TOTAL.labels(dataset_id=dataset_id).inc()

        if results:
            if 'metrics/mAP50(B)' in results:
                value = results['metrics/mAP50(B)']
                MODEL_MAP50.labels(dataset_id, task_id).set(value)
                MODEL_ACCURACY.labels(dataset_id, task_id, 'map50').set(value)

            if 'metrics/mAP50-95(B)' in results:
                value = results['metrics/mAP50-95(B)']
                MODEL_MAP.labels(dataset_id, task_id).set(value)
                MODEL_ACCURACY.labels(dataset_id, task_id, 'map').set(value)

            if 'metrics/precision(B)' in results:
                value = results['metrics/precision(B)']
                MODEL_PRECISION.labels(dataset_id, task_id).set(value)
                MODEL_ACCURACY.labels(dataset_id, task_id, 'precision').set(value)

            if 'metrics/recall(B)' in results:
                value = results['metrics/recall(B)']
                MODEL_RECALL.labels(dataset_id, task_id).set(value)
                MODEL_ACCURACY.labels(dataset_id, task_id, 'recall').set(value)

        TRAINING_DURATION.labels(dataset_id, task_id).observe(duration_seconds)

        # Set training status
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'completed').set(1)
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'in_progress').set(0)
        MODEL_TRAINING_STATUS.labels(dataset_id, task_id, 'failed').set(0)

        try:
            push_to_gateway(PUSHGATEWAY_URL, job=f'train_{task_id}', registry=REGISTRY)
            logger.info(f"Metrics pushed to gateway for task {task_id}")
        except Exception as e:
            logger.warning(f"Could not push metrics: {str(e)}")

    def record_test_completion(self, dataset_id, task_id, results, duration_seconds):
        MODELS_TESTED_TOTAL.labels(dataset_id=dataset_id).inc()

        if 'map50' in results:
            MODEL_MAP50.labels(dataset_id, task_id).set(results['map50'])

        if 'map' in results:
            MODEL_MAP.labels(dataset_id, task_id).set(results['map'])

        if 'precision' in results:
            MODEL_PRECISION.labels(dataset_id, task_id).set(results['precision'])

        if 'recall' in results:
            MODEL_RECALL.labels(dataset_id, task_id).set(results['recall'])

        if 'inference_speed' in results:
            INFERENCE_SPEED.labels(dataset_id, task_id).set(results['inference_speed'])

        TESTING_DURATION.labels(dataset_id, task_id).observe(duration_seconds)


# Singleton instance
metrics_exporter = MetricsExporter()