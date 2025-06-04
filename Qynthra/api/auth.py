"""
Authentication and authorization module for Quantum LLM API.
Provides enterprise-grade security features including JWT authentication,
role-based access control, and API key management.
"""

import os
import time
import uuid
import logging
from typing import Dict, List, Optional, Union, Any

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qynthra.api.auth")

# Security
security = HTTPBearer()

# In production, use a secure database for storing API keys and user information
# This is a simplified in-memory implementation
api_keys = {}
user_db = {
    # Example users - in production, store securely with hashed passwords
    "admin_user_id": {
        "username": "admin",
        "roles": ["admin"],
        "api_keys": []
    },
    "regular_user_id": {
        "username": "user",
        "roles": ["user"],
        "api_keys": []
    }
}

def verify_token(token: str) -> Dict:
    """
    Verify JWT token and return payload.
    
    Args:
        token: JWT token
        
    Returns:
        dict: Token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        # In production, use a secure secret key stored in environment variables
        secret_key = os.getenv("JWT_SECRET_KEY", "quantum_secure_secret_key")
        
        # Decode and verify token
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        
        # Check if token is expired
        if payload.get("exp", 0) < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """
    Get current user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        dict: User information
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)
        
        # In production, validate user against database
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        # Add additional user information if needed
        return payload
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

class RoleChecker:
    """
    Role-based access control checker.
    """
    
    def __init__(self, allowed_roles: List[str]):
        """
        Initialize role checker.
        
        Args:
            allowed_roles: List of allowed roles
        """
        self.allowed_roles = allowed_roles
    
    async def __call__(self, user: Dict = Depends(get_current_user)) -> bool:
        """
        Check if user has required role.
        
        Args:
            user: User information
            
        Returns:
            bool: True if user has required role
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        user_roles = user.get("roles", [])
        
        # Check if user has any of the allowed roles
        for role in user_roles:
            if role in self.allowed_roles:
                return True
        
        logger.warning(f"Unauthorized access attempt: User {user.get('sub')} with roles {user_roles} tried to access resource requiring {self.allowed_roles}")
        raise HTTPException(status_code=403, detail="Not authorized")

class APIKeyManager:
    """
    API key management for enterprise applications.
    """
    
    @staticmethod
    def generate_api_key(user_id: str, name: str, expires_in: int = 0) -> Dict:
        """
        Generate a new API key for a user.
        
        Args:
            user_id: User ID
            name: Key name
            expires_in: Expiration time in seconds (0 for no expiration)
            
        Returns:
            dict: API key information
        """
        # Check if user exists
        if user_id not in user_db:
            raise ValueError("User not found")
        
        # Generate API key
        api_key = str(uuid.uuid4())
        
        # Create API key record
        key_info = {
            "key": api_key,
            "name": name,
            "user_id": user_id,
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + expires_in if expires_in > 0 else 0,
            "last_used": 0,
            "enabled": True
        }
        
        # Store API key
        api_keys[api_key] = key_info
        user_db[user_id]["api_keys"].append(api_key)
        
        return key_info
    
    @staticmethod
    def validate_api_key(api_key: str) -> Dict:
        """
        Validate API key and return user information.
        
        Args:
            api_key: API key
            
        Returns:
            dict: User information
            
        Raises:
            ValueError: If API key is invalid
        """
        # Check if API key exists
        if api_key not in api_keys:
            raise ValueError("Invalid API key")
        
        key_info = api_keys[api_key]
        
        # Check if API key is enabled
        if not key_info["enabled"]:
            raise ValueError("API key is disabled")
        
        # Check if API key is expired
        if key_info["expires_at"] > 0 and key_info["expires_at"] < time.time():
            raise ValueError("API key is expired")
        
        # Update last used timestamp
        key_info["last_used"] = int(time.time())
        
        # Get user information
        user_id = key_info["user_id"]
        user_info = user_db.get(user_id)
        
        if not user_info:
            raise ValueError("User not found")
        
        return {
            "user_id": user_id,
            "username": user_info["username"],
            "roles": user_info["roles"]
        }
    
    @staticmethod
    def revoke_api_key(api_key: str) -> bool:
        """
        Revoke API key.
        
        Args:
            api_key: API key
            
        Returns:
            bool: True if API key was revoked
        """
        # Check if API key exists
        if api_key not in api_keys:
            return False
        
        # Disable API key
        api_keys[api_key]["enabled"] = False
        
        return True
    
    @staticmethod
    def list_user_api_keys(user_id: str) -> List[Dict]:
        """
        List API keys for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            list: List of API key information
        """
        # Check if user exists
        if user_id not in user_db:
            return []
        
        # Get API keys for user
        user_api_keys = user_db[user_id]["api_keys"]
        
        # Get API key information
        key_info_list = []
        for api_key in user_api_keys:
            if api_key in api_keys:
                key_info = api_keys[api_key].copy()
                # Don't expose the actual key in listings
                key_info["key"] = key_info["key"][:8] + "..." + key_info["key"][-4:]
                key_info_list.append(key_info)
        
        return key_info_list

# Helper functions for token management
def create_access_token(user_id: str, roles: List[str], expires_in: int = 3600) -> str:
    """
    Create JWT access token.
    
    Args:
        user_id: User ID
        roles: User roles
        expires_in: Expiration time in seconds
        
    Returns:
        str: JWT token
    """
    # Create payload
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": int(time.time()) + expires_in,
        "iat": int(time.time())
    }
    
    # In production, use a secure secret key stored in environment variables
    secret_key = os.getenv("JWT_SECRET_KEY", "quantum_secure_secret_key")
    
    # Create token
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    
    return token

def create_refresh_token(user_id: str, expires_in: int = 86400 * 7) -> str:
    """
    Create JWT refresh token.
    
    Args:
        user_id: User ID
        expires_in: Expiration time in seconds
        
    Returns:
        str: JWT token
    """
    # Create payload
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": int(time.time()) + expires_in,
        "iat": int(time.time())
    }
    
    # In production, use a secure secret key stored in environment variables
    secret_key = os.getenv("JWT_SECRET_KEY", "quantum_secure_secret_key")
    
    # Create token
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    
    return token