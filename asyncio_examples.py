
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import requests
import time

def unit_task(data):
    """
    For example, this function can be the requests.get for http response.
    """
    t = random.random() * 5
    time.sleep(t)
    print(f"slept {t} seconds")
    return data * t

async def batch_task(data, thread_pool):
    """
    Since the unit_task is an ordinary function, executing it can occupy
    the event loop thread, block other asyncio tasks, and hence concurrency 
    goes down to 1.
    In order to have some concurrency, use a thread pool to run the unit_task
    function.
    """
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(thread_pool, unit_task, unit_data)
        for unit_data in data
    ]
    results = await asyncio.gather(*futures)
    return results

if __name__ == "__main__":
    # concurrency is roughly the pool size
    thread_pool = ThreadPoolExecutor(10)
    results = []
    batch = []
    for x in range(100):
        batch.append(x)
        if len(batch) > 20:
            results.extend(asyncio.run(batch_task(batch, thread_pool)))
            batch.clear()
    print(results) 
