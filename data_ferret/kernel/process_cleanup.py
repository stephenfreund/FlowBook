"""Process cleanup utilities for managing child processes and loky workers."""

import asyncio
import os
import time

import psutil


async def stop_loky_and_all_children(timeout: float = 3.0, verbose: bool = False, max_passes: int = 2):
    """
    Clean up loky/joblib workers and all child processes.

    1) Ask loky/joblib to shut down cleanly (cancel futures, wait for exit).
    2) Force-reset the global reusable executor so the next cell gets a fresh one.
    3) Kill *all* remaining child processes.

    Args:
        timeout: Seconds to wait for processes to terminate
        verbose: If True, print debug information
        max_passes: Number of cleanup passes to attempt
    """
    me = psutil.Process(os.getpid())

    async def _shutdown_and_reset_loky():
        try:
            from joblib.externals.loky import reusable_executor
        except Exception:
            if verbose:
                print("Stopping loky and all children: no reusable executor found")
            return

        def _do():
            try:
                ex = reusable_executor.get_reusable_executor()
                if ex._executor_manager_thread is not None:
                    if verbose:
                        print("Killing loky workers")
                    ex._executor_manager_thread.kill_workers(
                        "executor shutting down in kernel"
                    )
                try:
                    if verbose:
                        print("Shutting down loky workers")
                    ex.shutdown(wait=True, kill_workers=True)
                except TypeError:
                    if verbose:
                        print("Waiting for loky workers to shut down")
                    ex.shutdown(wait=True)
                time.sleep(0.2)
            except Exception:
                pass

            # Hard reset: drop the singleton so next use creates a brand-new executor
            try:
                reusable_executor._executor = None
            except Exception:
                pass
            try:
                reusable_executor._executor_args = None
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _do)

    # Clean shutdown + singleton reset
    if "joblib" in globals():
        await _shutdown_and_reset_loky()

    # Reap any leftover children
    for _ in range(max_passes):
        try:
            kids = me.children(recursive=True)
        except Exception:
            kids = []
        if not kids:
            break

        if verbose:
            try:
                print("Terminating:", [(p.pid, p.name()) for p in kids])
            except Exception:
                print("Terminating:", [p.pid for p in kids])

        for p in kids:
            try:
                p.terminate()
            except Exception:
                pass

        _, alive = psutil.wait_procs(kids, timeout=timeout)

        for p in alive:
            if verbose:
                try:
                    print("Killing stubborn:", p.pid, p.name())
                except Exception:
                    print("Killing stubborn:", p.pid)
            try:
                p.kill()
            except Exception:
                pass

        psutil.wait_procs(alive, timeout=timeout)
