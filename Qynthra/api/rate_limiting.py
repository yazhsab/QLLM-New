"""
Rate limiting module for Quantum LLM API.
Provides enterprise-grade rate limiting capabilities to prevent abuse and ensure fair resource allocation.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qynthra.api.rate_limiting")

class RateLimiter:
    """
    Rate limiter for API requests.
    Implements a token bucket algorithm for flexible rate limiting.
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.rate = requests_per_minute / 60.0  # Requests per second
        self.burst_size = burst_size or requests_per_minute
        
        # User buckets: user_id -> (tokens, last_refill_time)
        self.buckets = {}
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if request is allowed
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        async with self.lock:
            # Get or create bucket
            if user_id not in self.buckets:
                self.buckets[user_id] = (self.burst_size, time.time())
            
            tokens, last_refill = self.buckets[user_id]
            
            # Calculate token refill
            now = time.time()
            time_passed = now - last_refill
            refill = time_passed * self.rate
            
            # Refill tokens
            tokens = min(self.burst_size, tokens + refill)
            
            # Check if request can be allowed
            if tokens < 1:
                # Calculate wait time
                wait_time = (1 - tokens) / self.rate
                
                logger.warning(f"Rate limit exceeded for user {user_id}. Wait time: {wait_time:.2f}s")
                
                # Return 429 Too Many Requests
                retry_after = int(wait_time) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Consume token
            tokens -= 1
            
            # Update bucket
            self.buckets[user_id] = (tokens, now)
            
            return True
    
    def get_remaining_tokens(self, user_id: str) -> float:
        """
        Get remaining tokens for user.
        
        Args:
            user_id: User ID
            
        Returns:
            float: Remaining tokens
        """
        if user_id not in self.buckets:
            return self.burst_size
        
        tokens, last_refill = self.buckets[user_id]
        
        # Calculate token refill
        now = time.time()
        time_passed = now - last_refill
        refill = time_passed * self.rate
        
        # Refill tokens
        tokens = min(self.burst_size, tokens + refill)
        
        return tokens

class TieredRateLimiter(RateLimiter):
    """
    Tiered rate limiter for different user tiers.
    """
    
    def __init__(self, tier_limits: Dict[str, int]):
        """
        Initialize tiered rate limiter.
        
        Args:
            tier_limits: Dictionary mapping tier names to requests per minute
        """
        # Default to lowest tier
        default_tier = min(tier_limits.values())
        super().__init__(requests_per_minute=default_tier)
        
        self.tier_limits = tier_limits
        self.user_tiers = {}  # user_id -> tier_name
    
    def set_user_tier(self, user_id: str, tier: str):
        """
        Set user tier.
        
        Args:
            user_id: User ID
            tier: Tier name
        """
        if tier not in self.tier_limits:
            raise ValueError(f"Invalid tier: {tier}")
        
        self.user_tiers[user_id] = tier
        
        # Reset bucket with new limit
        rate = self.tier_limits[tier] / 60.0
        burst_size = self.tier_limits[tier]
        
        # Update bucket with new rate and full tokens
        self.buckets[user_id] = (burst_size, time.time())
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if request is within rate limit based on user tier.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if request is allowed
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Get user tier
        tier = self.user_tiers.get(user_id, "default")
        
        # Get tier limit
        tier_limit = self.tier_limits.get(tier, min(self.tier_limits.values()))
        
        # Update rate if different from current
        current_rate = self.rate * 60.0
        if current_rate != tier_limit:
            self.rate = tier_limit / 60.0
            self.burst_size = tier_limit
        
        # Check rate limit
        return await super().check_rate_limit(user_id)

class ResourceBasedRateLimiter:
    """
    Resource-based rate limiter for quantum resources.
    Limits usage based on quantum resource consumption rather than request count.
    """
    
    def __init__(self, 
                 max_qubits_per_minute: int = 1000, 
                 max_circuits_per_minute: int = 100,
                 max_shots_per_minute: int = 10000):
        """
        Initialize resource-based rate limiter.
        
        Args:
            max_qubits_per_minute: Maximum number of qubits per minute
            max_circuits_per_minute: Maximum number of circuits per minute
            max_shots_per_minute: Maximum number of shots per minute
        """
        self.max_qubits = max_qubits_per_minute
        self.max_circuits = max_circuits_per_minute
        self.max_shots = max_shots_per_minute
        
        # User resource usage: user_id -> {resource: (usage, window_start)}
        self.usage = {}
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def check_resource_limit(self, 
                                  user_id: str, 
                                  qubits: int, 
                                  circuits: int, 
                                  shots: int) -> bool:
        """
        Check if resource usage is within limits.
        
        Args:
            user_id: User ID
            qubits: Number of qubits
            circuits: Number of circuits
            shots: Number of shots
            
        Returns:
            bool: True if usage is allowed
            
        Raises:
            HTTPException: If resource limit is exceeded
        """
        async with self.lock:
            now = time.time()
            window_size = 60  # 1 minute window
            
            # Get or create user usage
            if user_id not in self.usage:
                self.usage[user_id] = {
                    "qubits": (0, now),
                    "circuits": (0, now),
                    "shots": (0, now)
                }
            
            user_usage = self.usage[user_id]
            
            # Check and update each resource
            for resource, (usage, window_start) in user_usage.items():
                # Reset window if needed
                if now - window_start > window_size:
                    usage = 0
                    window_start = now
                
                # Get resource limit
                if resource == "qubits":
                    limit = self.max_qubits
                    new_usage = usage + qubits
                elif resource == "circuits":
                    limit = self.max_circuits
                    new_usage = usage + circuits
                elif resource == "shots":
                    limit = self.max_shots
                    new_usage = usage + shots
                else:
                    continue
                
                # Check if limit is exceeded
                if new_usage > limit:
                    logger.warning(f"Resource limit exceeded for user {user_id}. Resource: {resource}")
                    
                    # Calculate wait time (time until window resets)
                    wait_time = window_size - (now - window_start)
                    retry_after = int(wait_time) + 1
                    
                    raise HTTPException(
                        status_code=429,
                        detail=f"Resource limit exceeded for {resource}. Try again in {retry_after} seconds.",
                        headers={"Retry-After": str(retry_after)}
                    )
                
                # Update usage
                user_usage[resource] = (new_usage, window_start)
            
            # Update user usage
            self.usage[user_id] = user_usage
            
            return True
    
    def get_resource_usage(self, user_id: str) -> Dict[str, float]:
        """
        Get resource usage for user.
        
        Args:
            user_id: User ID
            
        Returns:
            dict: Resource usage
        """
        if user_id not in self.usage:
            return {
                "qubits": 0,
                "circuits": 0,
                "shots": 0
            }
        
        now = time.time()
        window_size = 60  # 1 minute window
        
        result = {}
        for resource, (usage, window_start) in self.usage[user_id].items():
            # Reset window if needed
            if now - window_start > window_size:
                result[resource] = 0
            else:
                result[resource] = usage
        
        return result

# Create default rate limiters
default_rate_limiter = RateLimiter(requests_per_minute=60)

# Create tiered rate limiter
tiered_rate_limiter = TieredRateLimiter({
    "free": 60,       # 60 requests per minute
    "basic": 300,     # 300 requests per minute
    "premium": 1000,  # 1000 requests per minute
    "enterprise": 5000  # 5000 requests per minute
})

# Create resource-based rate limiter
resource_rate_limiter = ResourceBasedRateLimiter(
    max_qubits_per_minute=1000,
    max_circuits_per_minute=100,
    max_shots_per_minute=10000
)