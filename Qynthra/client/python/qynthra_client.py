"""
Python client SDK for Qynthra API.
Provides enterprise-grade client capabilities for interacting with the Qynthra API.
"""

import os
import time
import json
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime

import requests
import websockets
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qynthra.client")

class QynthraClient:
    """
    Client for interacting with the Qynthra API.
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 token: Optional[str] = None,
                 timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            api_url: API URL
            api_key: API key (optional)
            username: Username (optional)
            password: Password (optional)
            token: Authentication token (optional)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key or os.environ.get("QYNTHRA_API_KEY")
        self.username = username or os.environ.get("QYNTHRA_USERNAME")
        self.password = password or os.environ.get("QYNTHRA_PASSWORD")
        self.token = token or os.environ.get("QYNTHRA_TOKEN")
        self.timeout = timeout
        
        # Authentication state
        self.authenticated = False
        self.token_expiry = 0
        self.user_id = None
        self.roles = []
        
        # Websocket connections
        self.ws_connections = {}
        
        # Auto-authenticate if credentials are provided
        if self.token:
            self._validate_token()
        elif self.api_key:
            self.authenticate_with_api_key(self.api_key)
        elif self.username and self.password:
            self.authenticate(self.username, self.password)
    
    def _validate_token(self):
        """Validate the token and extract user information."""
        try:
            # Decode token without verification to extract expiry
            decoded = jwt.decode(self.token, options={"verify_signature": False})
            
            # Check if token is expired
            if "exp" in decoded and decoded["exp"] < time.time():
                logger.warning("Token is expired")
                self.authenticated = False
                return False
            
            # Extract user information
            self.user_id = decoded.get("sub")
            self.roles = decoded.get("roles", [])
            self.token_expiry = decoded.get("exp", 0)
            
            self.authenticated = True
            return True
        except:
            logger.warning("Invalid token")
            self.authenticated = False
            return False
    
    def _get_headers(self):
        """Get request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "QynthraClient/1.0"
        }
        
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        
        return headers
    
    def _check_auth(self):
        """Check if client is authenticated."""
        if not self.authenticated:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        # Check if token is expired
        if self.token_expiry > 0 and self.token_expiry < time.time():
            # Try to refresh token
            if self.username and self.password:
                logger.info("Token expired. Refreshing...")
                self.authenticate(self.username, self.password)
            else:
                raise ValueError("Token expired. Please authenticate again.")
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            bool: True if authentication was successful
        """
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/token",
                json={"username": username, "password": password},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.token_expiry = int(time.time()) + data["expires_in"]
                self.user_id = data["user_id"]
                self.roles = data["roles"]
                self.authenticated = True
                
                # Save credentials for auto-refresh
                self.username = username
                self.password = password
                
                return True
            else:
                logger.error(f"Authentication failed: {response.text}")
                self.authenticated = False
                return False
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            self.authenticated = False
            return False
    
    def authenticate_with_api_key(self, api_key: str) -> bool:
        """
        Authenticate with API key.
        
        Args:
            api_key: API key
            
        Returns:
            bool: True if authentication was successful
        """
        self.api_key = api_key
        
        try:
            # Validate API key by making a test request
            response = requests.get(
                f"{self.api_url}/api/v1/models",
                headers={"X-API-Key": api_key},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.authenticated = True
                return True
            else:
                logger.error(f"API key validation failed: {response.text}")
                self.authenticated = False
                return False
        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            self.authenticated = False
            return False
    
    def list_models(self) -> List[Dict]:
        """
        List available models.
        
        Returns:
            list: List of models
        """
        self._check_auth()
        
        response = requests.get(
            f"{self.api_url}/api/v1/models",
            headers=self._get_headers(),
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            return response.json()["models"]
        else:
            raise Exception(f"Error listing models: {response.text}")
    
    def generate_text(self,
                     prompt: str,
                     model_id: str = "qynthra-base-4q",
                     max_length: int = 50,
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None) -> Dict:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            model_id: Model ID
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            dict: Generated text response
        """
        self._check_auth()
        
        request_data = {
            "prompt": prompt,
            "model_id": model_id,
            "max_length": max_length,
            "temperature": temperature,
            "stream": False
        }
        
        if top_k is not None:
            request_data["top_k"] = top_k
        
        if top_p is not None:
            request_data["top_p"] = top_p
        
        response = requests.post(
            f"{self.api_url}/api/v1/generate",
            json=request_data,
            headers=self._get_headers(),
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error generating text: {response.text}")
    
    async def generate_text_stream(self,
                                  prompt: str,
                                  model_id: str = "qynthra-base-4q",
                                  max_length: int = 50,
                                  temperature: float = 1.0,
                                  top_k: Optional[int] = None,
                                  top_p: Optional[float] = None,
                                  callback: Optional[Callable[[str, str], None]] = None) -> Dict:
        """
        Generate text from the model with streaming.
        
        Args:
            prompt: Input prompt
            model_id: Model ID
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            callback: Callback function for token events
            
        Returns:
            dict: Generated text response
        """
        self._check_auth()
        
        # Create request data
        request_data = {
            "prompt": prompt,
            "model_id": model_id,
            "max_length": max_length,
            "temperature": temperature
        }
        
        if top_k is not None:
            request_data["top_k"] = top_k
        
        if top_p is not None:
            request_data["top_p"] = top_p
        
        # Connect to WebSocket
        uri = f"ws://{self.api_url.replace('http://', '')}/api/v1/generate/stream"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate
            await websocket.send(json.dumps({"token": self.token}))
            
            # Send request
            await websocket.send(json.dumps(request_data))
            
            # Process responses
            generated_text = ""
            response_data = {}
            
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if "error" in data:
                    raise Exception(f"Error in streaming: {data['error']}")
                
                if data["event"] == "start":
                    response_data = {
                        "id": data["id"],
                        "created_at": data["created_at"],
                        "generated_text": ""
                    }
                elif data["event"] == "token":
                    token = data["token"]
                    generated_text += token
                    
                    if callback:
                        callback("token", token)
                elif data["event"] == "complete":
                    response_data["generated_text"] = data["generated_text"]
                    response_data["processing_time"] = data["processing_time"]
                    
                    if callback:
                        callback("complete", data["generated_text"])
                    
                    break
        
        return response_data
    
    def get_model_info(self, model_id: str) -> Dict:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            
        Returns:
            dict: Model information
        """
        self._check_auth()
        
        models = self.list_models()
        
        for model in models:
            if model["id"] == model_id:
                return model
        
        raise ValueError(f"Model not found: {model_id}")

class AsyncQynthraClient(QynthraClient):
    """
    Asynchronous client for interacting with the Qynthra API.
    """
    
    async def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            bool: True if authentication was successful
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/v1/token",
                    json={"username": username, "password": password},
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.token = data["access_token"]
                        self.token_expiry = int(time.time()) + data["expires_in"]
                        self.user_id = data["user_id"]
                        self.roles = data["roles"]
                        self.authenticated = True
                        
                        # Save credentials for auto-refresh
                        self.username = username
                        self.password = password
                        
                        return True
                    else:
                        text = await response.text()
                        logger.error(f"Authentication failed: {text}")
                        self.authenticated = False
                        return False
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            self.authenticated = False
            return False
    
    async def list_models(self) -> List[Dict]:
        """
        List available models.
        
        Returns:
            list: List of models
        """
        import aiohttp
        
        self._check_auth()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_url}/api/v1/models",
                headers=self._get_headers(),
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["models"]
                else:
                    text = await response.text()
                    raise Exception(f"Error listing models: {text}")
    
    async def generate_text(self,
                           prompt: str,
                           model_id: str = "qynthra-base-4q",
                           max_length: int = 50,
                           temperature: float = 1.0,
                           top_k: Optional[int] = None,
                           top_p: Optional[float] = None) -> Dict:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            model_id: Model ID
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            dict: Generated text response
        """
        import aiohttp
        
        self._check_auth()
        
        request_data = {
            "prompt": prompt,
            "model_id": model_id,
            "max_length": max_length,
            "temperature": temperature,
            "stream": False
        }
        
        if top_k is not None:
            request_data["top_k"] = top_k
        
        if top_p is not None:
            request_data["top_p"] = top_p
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/api/v1/generate",
                json=request_data,
                headers=self._get_headers(),
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"Error generating text: {text}")

# Example usage
if __name__ == "__main__":
    # Create client
    client = QynthraClient(
        api_url="http://localhost:8000",
        username="user",
        password="qynthra_user_password"
    )
    
    # List models
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Generate text
    response = client.generate_text(
        prompt="Hello, quantum world!",
        model_id="qllm-base-4q",
        max_length=50,
        temperature=0.7
    )
    
    print(f"Generated text: {response['generated_text']}")
    
    # Async example
    async def async_example():
        # Create async client
        async_client = AsyncQuantumLLMClient(
            api_url="http://localhost:8000",
            username="user",
            password="quantum_user_password"
        )
        
        # Generate text
        response = await async_client.generate_text(
            prompt="Hello, quantum world!",
            model_id="qllm-base-4q",
            max_length=50,
            temperature=0.7
        )
        
        print(f"Async generated text: {response['generated_text']}")
    
    # Run async example
    asyncio.run(async_example())