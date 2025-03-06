import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

class TaskQueue:
    def __init__(self, max_workers: int = 1, max_queue_size: int = 20):
        self.video_queue = asyncio.Queue(maxsize=max_queue_size)
        self.results = {}
        self.max_workers = max_workers
        self.workers = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.video_tasks = set()
        self.max_queue_size = max_queue_size
        self.queue_positions = {}  # Maps task_id to position in queue

    async def start(self):
        if self.running:
            return
        self.running = True
        for _ in range(self.max_workers):
            worker = asyncio.create_task(self._worker())
            self.workers.append(worker)

    async def stop(self):
        self.running = False
        for worker in self.workers:
            worker.cancel()
        self.workers.clear()
        await self.video_queue.join()

    async def _worker(self):
        while self.running:
            try:
                task_id, coro, args, kwargs = await self.video_queue.get()
                # Remove from queue positions since it's now being processed
                if task_id in self.queue_positions:
                    del self.queue_positions[task_id]
                # Update positions for remaining tasks
                self._update_queue_positions()
                
                try:
                    result = await coro(*args, **kwargs)
                    self.results[task_id] = {
                        "status": "completed",
                        "result": result,
                        "error": None,
                        "completed_at": time.time()
                    }
                except Exception as e:
                    self.results[task_id] = {
                        "status": "failed",
                        "result": None,
                        "error": str(e),
                        "completed_at": time.time()
                    }
                finally:
                    self.video_queue.task_done()
                    if task_id in self.video_tasks:
                        self.video_tasks.remove(task_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    def _update_queue_positions(self):
        """Update position numbers for all tasks in queue"""
        queue_size = self.video_queue.qsize()
        position = 1
        # Get all tasks from the queue (this is a simplified approach)
        for task_id in list(self.queue_positions.keys()):
            if task_id in self.video_tasks:
                self.queue_positions[task_id] = position
                position += 1

    async def add_task(self, task_id: str, coro, *args, **kwargs):
        # Check if queue is full
        if self.video_queue.qsize() >= self.max_queue_size:
            raise ValueError(f"Queue is full (max {self.max_queue_size} tasks)")
            
        # Add to tracking sets
        self.video_tasks.add(task_id)
        # Assign initial queue position
        position = self.video_queue.qsize() + 1
        self.queue_positions[task_id] = position
        
        # Add to actual queue
        await self.video_queue.put((task_id, coro, args, kwargs))
        return task_id, position

    def get_task_status(self, task_id: str):
        # If task is in active processing
        if task_id in self.video_tasks and task_id not in self.queue_positions:
            return {"status": "processing"}
        
        # If task is in queue
        if task_id in self.queue_positions:
            return {
                "status": "queued",
                "position": self.queue_positions[task_id],
                "total_in_queue": self.video_queue.qsize()
            }
            
        # If task has results
        return self.results.get(task_id, {"status": "not_found"})

    def get_queue_info(self):
        """Get current queue statistics"""
        return {
            "current_size": self.video_queue.qsize(),
            "max_size": self.max_queue_size,
            "active_workers": self.max_workers
        }

# Global task queue instance
task_queue = TaskQueue(max_workers=2, max_queue_size=20)


def async_task(func):
    async def wrapper(*args, **kwargs):
        # Remove task_id from kwargs to avoid duplicate passing
        task_id = kwargs.pop('task_id', f"{func.__name__}_{time.time()}")
        try:
            result = await task_queue.add_task(task_id, func, *args, **kwargs)
            return result  # Now returns (task_id, position)
        except ValueError as e:
            # Queue full error
            raise ValueError(str(e))
    return wrapper


async def cleanup_old_results():
    while True:
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1 hour
            for task_id in list(task_queue.results.keys()):
                result = task_queue.results[task_id]
                if result.get('completed_at', 0) < cutoff_time:
                    del task_queue.results[task_id]
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
            await asyncio.sleep(60)

def start_cleanup(loop=None):
    """Start the cleanup task with an optional event loop"""
    if loop is None:
        loop = asyncio.get_event_loop()
    loop.create_task(cleanup_old_results())

# Export the cleanup function
__all__ = ['task_queue', 'start_cleanup', 'async_task']