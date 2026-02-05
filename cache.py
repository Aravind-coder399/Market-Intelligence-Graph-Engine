"""
Redis Cache Layer for Market Intelligence
Caches embeddings, predictions, and market data for faster repeated queries.
"""
import os
import json
import logging
import hashlib
from typing import Any, Optional

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis-based caching layer for faster repeated queries."""
    
    def __init__(self):
        self.client = None
        self.enabled = False
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        try:
            import redis
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            self.enabled = True
            logger.info(f"Redis cache connected: {redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) - caching disabled")
    
    def _hash_key(self, prefix: str, data: str) -> str:
        """Generate a cache key from data."""
        return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"
    
    def get_embedding(self, headline: str) -> Optional[list]:
        """Get cached embedding for a headline."""
        if not self.enabled:
            return None
        
        try:
            key = self._hash_key("emb", headline)
            cached = self.client.get(key)
            if cached:
                logger.debug(f"Cache HIT: {key[:20]}...")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set_embedding(self, headline: str, embedding: list, ttl: int = 86400) -> bool:
        """Cache an embedding with TTL (default 24 hours)."""
        if not self.enabled:
            return False
        
        try:
            key = self._hash_key("emb", headline)
            self.client.setex(key, ttl, json.dumps(embedding))
            return True
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return False
    
    def get_prediction(self, headline: str, ticker: str = None) -> Optional[dict]:
        """Get cached prediction result."""
        if not self.enabled:
            return None
        
        try:
            cache_input = f"{headline}:{ticker or 'none'}"
            key = self._hash_key("pred", cache_input)
            cached = self.client.get(key)
            if cached:
                logger.debug(f"Prediction Cache HIT: {key[:20]}...")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set_prediction(self, headline: str, prediction: dict, ticker: str = None, ttl: int = 3600) -> bool:
        """Cache a prediction result with TTL (default 1 hour)."""
        if not self.enabled:
            return False
        
        try:
            cache_input = f"{headline}:{ticker or 'none'}"
            key = self._hash_key("pred", cache_input)
            self.client.setex(key, ttl, json.dumps(prediction))
            return True
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return False
    
    def get_market_data(self, ticker: str) -> Optional[dict]:
        """Get cached market data for a ticker."""
        if not self.enabled:
            return None
        
        try:
            key = f"market:{ticker}"
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set_market_data(self, ticker: str, data: dict, ttl: int = 300) -> bool:
        """Cache market data with TTL (default 5 minutes)."""
        if not self.enabled:
            return False
        
        try:
            key = f"market:{ticker}"
            self.client.setex(key, ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return False
    
    def clear_all(self) -> bool:
        """Clear all cached data."""
        if not self.enabled:
            return False
        
        try:
            self.client.flushdb()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
        
        return False
    
    def stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.client.info("memory")
            keys = self.client.dbsize()
            return {
                "enabled": True,
                "keys": keys,
                "memory_used": info.get("used_memory_human", "N/A")
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}


# Global cache instance
_cache = None

def get_cache() -> RedisCache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
