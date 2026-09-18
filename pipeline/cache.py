# pipeline/cache.py
"""
Redis-backed cache for Nsight Compute metrics, keyed by a hash of the kernel
source.

The cache is a pure OPTIMIZATION and must never be load-bearing. Previously the
client was constructed at import time and `get_cached_metrics` let
redis.ConnectionError propagate, which aborted the whole of pre_flight() -- so a
machine without Redis running lost its profiler metrics entirely, the bottleneck
degraded to "unknown", and the entire profiler-guided premise of the system
silently evaporated. Now every Redis interaction fails soft: a dead cache means
a cache miss, and pre_flight goes on to compile and profile as normal.
"""
import json
import os

import redis

REDIS_HOST = os.getenv("KARMA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("KARMA_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("KARMA_REDIS_DB", "0"))
CACHE_TTL_SECONDS = 604800  # 7 days

# Connecting is deferred until first use, and a failure is remembered so we
# don't pay a TCP timeout on every single lookup of a long run.
_client: redis.Redis | None = None
_unavailable = False
_warned = False


def _warn_once(exc: Exception) -> None:
    global _warned
    if not _warned:
        print(f"  [cache] Redis unavailable ({exc.__class__.__name__}) — "
              f"continuing without metric caching")
        _warned = True


def _get_client() -> redis.Redis | None:
    """Lazily connect. Returns None if Redis is unreachable."""
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is None:
        try:
            client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            client.ping()  # force the connection now, not on first get()
            _client = client
        except Exception as e:
            _unavailable = True
            _warn_once(e)
            return None
    return _client


def get_cached_metrics(code_hash: str) -> dict | None:
    """Cached metrics, or None on miss OR on any cache failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        cached = client.get(f"karma_metrics:{code_hash}")
    except Exception as e:
        _warn_once(e)
        return None

    if not cached:
        return None
    try:
        data = json.loads(cached)
    except (ValueError, TypeError):
        return None  # poisoned entry -> treat as a miss

    print("  [cache] hit — bypassing compiler and profiler")
    return data


def save_to_cache(code_hash: str, metrics: dict) -> bool:
    """Best-effort write. Returns True if stored, False if the cache is down."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.setex(f"karma_metrics:{code_hash}", CACHE_TTL_SECONDS, json.dumps(metrics))
        return True
    except Exception as e:
        _warn_once(e)
        return False
