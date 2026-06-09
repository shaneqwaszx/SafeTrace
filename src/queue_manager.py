"""System-wide queue manager for SafeTrace.

Implements a FIFO queue to balance the load and prevent concurrent
GPU/RAM resource exhaustion during batch uploads or multiple user sessions.
"""
import concurrent.futures

# max_workers=1 ensures strict sequential processing across the system.
# This prevents Out-Of-Memory (OOM) errors by ensuring only one heavy 
# ML operation (Ingestion/YOLO) runs at any given time.
pipeline_queue = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def submit_task(func, *args, **kwargs):
    """Submit a heavy ML task to the internal load-balancing queue."""
    return pipeline_queue.submit(func, *args, **kwargs)