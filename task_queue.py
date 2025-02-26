import asyncio
import threading
import queue
import time
from functools import wraps
import logging
from typing import Dict, Any, Callable, Coroutine, Optional, List

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("TaskQueue")

class TaskQueue:
    """A queue system for handling long-running tasks asynchronously"""
    
    def __init__(self, max_workers: int = 3):
        self.task_queue = queue.Queue()
        self.results = {}
        self.max_workers = max_workers
        self.workers = []
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start the worker threads"""
        if self.running:
            return
        
        self.running = True
        
        # Create worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} worker threads")
    
    def stop(self):
        """Stop the worker threads"""
        self.running = False
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        logger.info("All worker threads stopped")
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop that processes tasks from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get a task from the queue with a timeout so we can check if we should stop
                task_id, task_func, args, kwargs = self.task_queue.get(timeout=1.0)
                
                logger.info(f"Worker {worker_id} processing task {task_id}")
                
                try:
                    # Execute the task
                    result = task_func(*args, **kwargs)
                    
                    # Store the result
                    with self.lock:
                        self.results[task_id] = {
                            "status": "completed",
                            "result": result,
                            "error": None,
                            "completed_at": time.time()
                        }
                except Exception as e:
                    logger.error(f"Error in task {task_id}: {str(e)}")
                    
                    # Store the error
                    with self.lock:
                        self.results[task_id] = {
                            "status": "failed",
                            "result": None,
                            "error": str(e),
                            "completed_at": time.time()
                        }
                
                # Mark the task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks in the queue, just continue
                pass
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker_id}: {str(e)}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def add_task(self, task_id: str, task_func: Callable, *args, **kwargs) -> str:
        """Add a task to the queue and return its ID"""
        with self.lock:
            self.results[task_id] = {
                "status": "pending",
                "result": None,
                "error": None,
                "created_at": time.time()
            }
        
        # Add the task to the queue
        self.task_queue.put((task_id, task_func, args, kwargs))
        
        logger.info(f"Added task {task_id} to the queue")
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task"""
        with self.lock:
            if task_id in self.results:
                return self.results[task_id]
            else:
                return {"status": "not_found"}
    
    def clear_completed_tasks(self, max_age: float = 3600.0):
        """Clear completed tasks older than max_age seconds"""
        now = time.time()
        with self.lock:
            to_remove = []
            for task_id, task_data in self.results.items():
                if task_data["status"] in ["completed", "failed"]:
                    completed_at = task_data.get("completed_at", 0)
                    if now - completed_at > max_age:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.results[task_id]
            
            if to_remove:
                logger.info(f"Cleared {len(to_remove)} completed tasks")

# Global task queue instance
task_queue = TaskQueue(max_workers=3)

# Start the queue when the module is imported
task_queue.start()

# Background task that periodically cleans up old tasks
async def cleanup_task():
    """Periodically clean up old tasks"""
    while True:
        task_queue.clear_completed_tasks()
        await asyncio.sleep(3600)  # Run every hour

# Function to start the cleanup task
def start_cleanup():
    """Start the cleanup task"""
    loop = asyncio.get_event_loop()
    loop.create_task(cleanup_task())

# Decorator for creating asynchronous tasks
def async_task(func):
    """Decorator to run a function as an asynchronous task"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate a unique task ID
        task_id = f"{func.__name__}_{time.time()}"
        
        # Add the task to the queue
        task_queue.add_task(task_id, func, *args, **kwargs)
        
        return task_id
    return wrapper