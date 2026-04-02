import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def get_cached_metrics(code_hash: str)->dict|None:
    cached_data = redis_client.get(f"karma_metrics:{code_hash}")
    if cached_data:
        print("⚡ Cache Hit! Bypassing compiler and profiler.")
        return json.loads(cached_data)
    
    return None

def save_to_cache(code_hash:str, metrics:dict):
    cached_data = redis_client.setex(f"karma_metrics:{code_hash}", 604800, json.dumps(metrics))