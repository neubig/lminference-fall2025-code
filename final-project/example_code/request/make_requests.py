# import requests
import json
# import os
# import numpy as np
from typing import List, Dict, Any, Tuple

import asyncio
import aiohttp
import time
from datetime import datetime
from pathlib import Path

# Modal will give you a URL after deployment, you will need to edit yourModalID to your modal username
url = "https://csna--csna-1-model-completions.modal.run"

async def send_batch_at_scheduled_time(
    session: aiohttp.ClientSession,
    batch: Dict[str, Any],
    url: str,
    start_time: float,
    semaphore: asyncio.Semaphore,
    output_dir: str = "batch_results"
) -> Dict[str, Any]:
    """
    Send a single batch request at its scheduled arrival time.
    
    Args:
        session: aiohttp client session
        batch: batch dictionary with arrival_time, prompts, etc.
        url: API endpoint URL
        start_time: simulation start time (from time.time())
        semaphore: semaphore to limit concurrent requests
        output_dir: directory to save results
        
    Returns:
        Dictionary with batch info and results
    """
    # Calculate how long to wait until this batch's arrival time
    current_elapsed = time.time() - start_time
    wait_time = max(0, batch['arrival_time'] - current_elapsed)
    
    if wait_time > 0:
        await asyncio.sleep(wait_time)
    
    # Wait for semaphore availability (capacity) AFTER sleeping for schedule
    async with semaphore:
        # Record actual send time
        actual_send_time = time.time() - start_time
        
        print(f"[t={actual_send_time:.2f}s] Sending batch {batch['batch_id']} "
              f"(scheduled: {batch['arrival_time']:.2f}s, size: {batch['batch_size']})")
        
        # Send the request
        request_start = time.time()
        try:
            async with session.post(
                url,
                json={
                    "prompt": batch['prompts'],
                    "max_tokens": batch['max_length'],
                },
                timeout=aiohttp.ClientTimeout(total=600)  # 10 minute timeout # NOTE: you may want to set a manual timeout=600 in your @app.cls() decorator as well
            ) as response:
                response_data = await response.json()
                request_duration = time.time() - request_start
                
                result = {
                    "batch_id": batch['batch_id'],
                    "batch_size": batch['batch_size'],
                    "scheduled_arrival_time": batch['arrival_time'],
                    "actual_send_time": actual_send_time,
                    "request_duration": request_duration,
                    "completion_time": time.time() - start_time,
                    "status_code": response.status,
                    "prompt_idxs": batch.get('prompt_idxs', []),
                    "response": response_data if response.status == 200 else None,
                    "error": None if response.status == 200 else f"HTTP {response.status}"
                }
                
                # Write result immediately
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                
                filename = output_path / f"batch_{batch['batch_id']:04d}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"[t={result['completion_time']:.2f}s] Completed batch {batch['batch_id']} "
                      f"(duration: {request_duration:.2f}s, status: {response.status})")
                
                return result
                
        except asyncio.TimeoutError:
            request_duration = time.time() - request_start
            result = {
                "batch_id": batch['batch_id'],
                "batch_size": batch['batch_size'],
                "scheduled_arrival_time": batch['arrival_time'],
                "actual_send_time": actual_send_time,
                "request_duration": request_duration,
                "completion_time": time.time() - start_time,
                "status_code": None,
                "prompt_idxs": batch.get('prompt_idxs', []),
                "response": None,
                "error": "Timeout"
            }
            print(f"[t={result['completion_time']:.2f}s] TIMEOUT batch {batch['batch_id']}")
            return result
            
        except Exception as e:
            request_duration = time.time() - request_start
            result = {
                "batch_id": batch['batch_id'],
                "batch_size": batch['batch_size'],
                "scheduled_arrival_time": batch['arrival_time'],
                "actual_send_time": actual_send_time,
                "request_duration": request_duration,
                "completion_time": time.time() - start_time,
                "status_code": None,
                "prompt_idxs": batch.get('prompt_idxs', []),
                "response": None,
                "error": str(e)
            }
            print(f"[t={result['completion_time']:.2f}s] ERROR batch {batch['batch_id']}: {e}")
            return result


async def run_batch_simulation(
    batches: List[Dict[str, Any]],
    url: str,
    output_dir: str = "batch_results",
    max_concurrent_requests: int = 50
) -> List[Dict[str, Any]]:
    """
    Run the full batch arrival simulation, sending requests at scheduled times
    and collecting results asynchronously.
    
    Args:
        batches: List of batch dictionaries from simulate_poisson_batch_arrivals_exhaustive
        url: API endpoint URL
        output_dir: Directory to save individual batch results
        max_concurrent_requests: Maximum number of concurrent requests
        
    Returns:
        List of result dictionaries for all batches
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Starting simulation with {len(batches)} batches")
    print(f"Total prompts: {sum(b['batch_size'] for b in batches)}")
    print(f"Arrival simulation duration: {batches[-1]['arrival_time']:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"Max concurrent requests: {max_concurrent_requests}\n")
    
    start_time = time.time()
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    # Configure connector to allow enough concurrent connections
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests + 10)

    # Create aiohttp session and send all batches
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_batch_at_scheduled_time(
                session, batch, url, start_time, semaphore, output_dir
            )
            for batch in batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Process results and handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Exception in batch {i}: {result}")
            processed_results.append({
                "batch_id": batches[i]['batch_id'],
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    # Save summary
    summary = {
        "total_batches": len(batches),
        "total_prompts": sum(b['batch_size'] for b in batches),
        "scheduled_duration": batches[-1]['arrival_time'],
        "actual_duration": total_time,
        "successful_batches": sum(1 for r in processed_results if r.get('status_code') == 200),
        "failed_batches": sum(1 for r in processed_results if r.get('status_code') != 200),
        "results": processed_results
    }
    
    summary_path = output_path / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Simulation complete!")
    print(f"Total time: {total_time:.2f}s (scheduled: {batches[-1]['arrival_time']:.2f}s)")
    print(f"Successful: {summary['successful_batches']}/{len(batches)}")
    print(f"Failed: {summary['failed_batches']}/{len(batches)}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    return processed_results

async def main():
    with open("batch_arrivals_sample.json") as f:
        batches_sample = json.load(f)

    results = await run_batch_simulation(
        batches=batches_sample,
        url=url,
        output_dir="batch_results_sample_py",
        max_concurrent_requests=300,
    )

if __name__ == "__main__":
    asyncio.run(main())
