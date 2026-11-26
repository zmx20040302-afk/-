"""Run independent geometry-constrained searches for one shared full-plane target."""
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from multiprocessing import get_context
import os

FAMILIES = ("random", "cluster", "edge", "stripe", "patch")


@lru_cache(maxsize=1)
def worker_pool():
    return ProcessPoolExecutor(max_workers=min(len(FAMILIES), os.cpu_count() or 1),
                               mp_context=get_context("spawn"))


def solve_one(mask, target, family, mode="protected", scope="plane", point=None):
    from family_design import design_family

    return design_family(mask, family, mode, target=target, fixed=True, parallel=True, scope=scope, point=point)


def design_all(mask, target, mode="protected", scope="plane", point=None):
    pool = worker_pool()
    jobs = {}
    try:
        for family in FAMILIES:
            jobs[pool.submit(solve_one, mask, target, family, mode, scope, point)] = family
        for future in as_completed(jobs):
            family = jobs[future]
            try:
                yield future.result()
            except BrokenProcessPool:
                raise
            except Exception as exc:
                yield dict(family=family, status="error", reason=str(exc), mask=None)
    except BrokenProcessPool:
        pool.shutdown(wait=True, cancel_futures=True)
        worker_pool.cache_clear()
        raise
    finally:
        # A closed client must not release the solver slot while its workers are active.
        for future in jobs:
            future.cancel()
        wait(jobs)
