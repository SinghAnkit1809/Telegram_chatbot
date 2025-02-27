import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

class TaskQueue:
    def __init__(self, max_workers: int = 2):
        self.video_queue = asyncio.Queue()
        self.results = {}
        self.max_workers = max_workers
        self.workers = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.video_tasks = set()

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

    async def add_task(self, task_id: str, coro, *args, **kwargs):
        self.video_tasks.add(task_id)
        await self.video_queue.put((task_id, coro, args, kwargs))
        return task_id

    def get_task_status(self, task_id: str):
        if task_id in self.video_tasks:
            return {"status": "processing"}
        return self.results.get(task_id, {"status": "not_found"})

# Global task queue instance
task_queue = TaskQueue(max_workers=2)

# def async_task(func):   
#     async def wrapper(*args, **kwargs):
#         task_id = kwargs.get('task_id') or f"{func.__name__}_{time.time()}"
#         await task_queue.add_task(task_id, func, *args, **kwargs)
#         return task_id
#     return wrapper
def async_task(func):
    async def wrapper(*args, **kwargs):
        # Remove task_id from kwargs to avoid duplicate passing
        task_id = kwargs.pop('task_id', f"{func.__name__}_{time.time()}")
        await task_queue.add_task(task_id, func, *args, **kwargs)
        return task_id
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