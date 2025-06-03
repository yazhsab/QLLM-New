"""
REST API implementation for Quantum LLM.
Provides enterprise-grade API endpoints for model inference and management.
"""

import os
import time
import uuid
import json
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any

import jwt
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
import numpy as np

from quantum_llm.qllm_base import QLLMBase
from quantum_llm.qllm_advanced import QLLMAdvanced, QLLMWithKVCache
from quantum_llm.api.auth import verify_token, get_current_user, RoleChecker
from quantum_llm.api.monitoring import log_request, log_response, monitor_performance
from quantum_llm.api.rate_limiting import RateLimiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quantum_llm_api.log")
    ]
)
logger = logging.getLogger("quantum_llm.api")

# Create FastAPI app
app = FastAPI(
    title="Quantum LLM API",
    description="Enterprise-grade API for Quantum Large Language Model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
admin_role = RoleChecker(["admin"])
user_role = RoleChecker(["user", "admin"])

# Rate limiting
rate_limiter = RateLimiter(requests_per_minute=60)

# Model cache
model_cache = {}

# WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Request/Response Models
class TokenRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: str
    roles: List[str]

class GenerateRequest(BaseModel):
    prompt: str
    model_id: str
    max_length: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False

class GenerateResponse(BaseModel):
    id: str
    model_id: str
    generated_text: str
    prompt: str
    created_at: int
    processing_time: float

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    created_at: int
    quantum_device: str
    circuit_type: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

# API Routes
@app.post("/api/v1/token", response_model=TokenResponse)
async def get_token(request: TokenRequest):
    """Get authentication token"""
    try:
        # In production, validate against secure user database
        # This is a simplified example
        if request.username == "admin" and request.password == "quantum_secure_password":
            roles = ["admin"]
        elif request.username == "user" and request.password == "quantum_user_password":
            roles = ["user"]
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_id = str(uuid.uuid4())
        expires_in = 3600  # 1 hour
        
        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": int(time.time()) + expires_in
        }
        
        # In production, use a secure secret key stored in environment variables
        secret_key = os.getenv("JWT_SECRET_KEY", "quantum_secure_secret_key")
        access_token = jwt.encode(payload, secret_key, algorithm="HS256")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "user_id": user_id,
            "roles": roles
        }
    except Exception as e:
        logger.error(f"Token generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/models", response_model=ModelsResponse)
async def list_models(credentials: HTTPAuthorizationCredentials = Depends(security),
                     user: Dict = Depends(get_current_user),
                     _: bool = Depends(user_role)):
    """List available models"""
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit(user["sub"])
        
        # Log request
        log_request("list_models", user["sub"])
        
        # In production, fetch from database
        models = [
            {
                "id": "qllm-base-4q",
                "name": "Quantum LLM Base (4 qubits)",
                "description": "Basic quantum language model with 4 qubits",
                "parameters": {
                    "n_qubits": 4,
                    "n_layers": 2,
                    "embedding_dim": 16,
                    "vocab_size": 1000
                },
                "created_at": int(time.time()) - 86400,
                "quantum_device": "default.qubit",
                "circuit_type": "quantum_transformer"
            },
            {
                "id": "qllm-advanced-8q",
                "name": "Quantum LLM Advanced (8 qubits)",
                "description": "Advanced quantum language model with 8 qubits",
                "parameters": {
                    "n_qubits": 8,
                    "n_layers": 4,
                    "n_heads": 4,
                    "embedding_dim": 32,
                    "vocab_size": 5000
                },
                "created_at": int(time.time()) - 43200,
                "quantum_device": "default.qubit",
                "circuit_type": "quantum_transformer"
            }
        ]
        
        # Log response
        log_response("list_models", user["sub"], 200)
        
        return {"models": models}
    except Exception as e:
        logger.error(f"List models error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest,
                       background_tasks: BackgroundTasks,
                       credentials: HTTPAuthorizationCredentials = Depends(security),
                       user: Dict = Depends(get_current_user),
                       _: bool = Depends(user_role)):
    """Generate text from the model"""
    try:
        # Rate limiting
        await rate_limiter.check_rate_limit(user["sub"])
        
        # Start performance monitoring
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Log request
        log_request("generate_text", user["sub"], request_dict=request.dict())
        
        # Check if streaming is requested
        if request.stream:
            raise HTTPException(status_code=400, detail="For streaming, use the WebSocket endpoint")
        
        # Get or load model
        model = await get_model(request.model_id)
        
        # Tokenize input (simplified)
        # In production, use a proper tokenizer
        input_ids = torch.tensor([[i for i in range(min(10, len(request.prompt)))]])
        
        # Generate text
        with monitor_performance("text_generation", user["sub"], request_id):
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
        
        # Detokenize (simplified)
        # In production, use a proper tokenizer
        generated_text = "Generated quantum text: " + request.prompt
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log response
        log_response("generate_text", user["sub"], 200, processing_time=processing_time)
        
        # Schedule background task for analytics
        background_tasks.add_task(
            record_generation_analytics,
            user_id=user["sub"],
            model_id=request.model_id,
            prompt_length=len(request.prompt),
            generated_length=len(generated_text),
            processing_time=processing_time
        )
        
        return {
            "id": request_id,
            "model_id": request.model_id,
            "generated_text": generated_text,
            "prompt": request.prompt,
            "created_at": int(time.time()),
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Text generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/api/v1/generate/stream")
async def generate_stream(websocket: WebSocket):
    """Stream text generation via WebSocket"""
    await websocket.accept()
    
    try:
        # Authenticate
        auth_message = await websocket.receive_text()
        auth_data = json.loads(auth_message)
        
        if "token" not in auth_data:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close()
            return
        
        try:
            user = verify_token(auth_data["token"])
        except:
            await websocket.send_json({"error": "Invalid token"})
            await websocket.close()
            return
        
        # Store connection
        connection_id = str(uuid.uuid4())
        active_connections[connection_id] = websocket
        
        # Process requests
        while True:
            # Receive request
            request_message = await websocket.receive_text()
            request_data = json.loads(request_message)
            
            # Validate request
            if not all(k in request_data for k in ["prompt", "model_id"]):
                await websocket.send_json({"error": "Invalid request format"})
                continue
            
            # Rate limiting
            try:
                await rate_limiter.check_rate_limit(user["sub"])
            except:
                await websocket.send_json({"error": "Rate limit exceeded"})
                continue
            
            # Get model
            try:
                model = await get_model(request_data["model_id"])
            except:
                await websocket.send_json({"error": "Model not found"})
                continue
            
            # Generate text with streaming
            request_id = str(uuid.uuid4())
            
            # Send initial response
            await websocket.send_json({
                "id": request_id,
                "event": "start",
                "created_at": int(time.time())
            })
            
            # Simulate streaming generation
            # In production, implement actual token-by-token generation
            prompt = request_data["prompt"]
            max_length = request_data.get("max_length", 50)
            
            # Stream tokens
            for i in range(min(10, max_length)):
                await asyncio.sleep(0.2)  # Simulate generation time
                token = f"token_{i}"
                
                await websocket.send_json({
                    "id": request_id,
                    "event": "token",
                    "token": token,
                    "index": i
                })
            
            # Send completion
            await websocket.send_json({
                "id": request_id,
                "event": "complete",
                "generated_text": f"Generated quantum text: {prompt}",
                "processing_time": 2.0
            })
            
    except WebSocketDisconnect:
        # Clean up on disconnect
        if connection_id in active_connections:
            del active_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"error": "Internal server error"})
            await websocket.close()
        except:
            pass

# Admin routes
@app.post("/api/v1/admin/models", status_code=201)
async def create_model(request: Request,
                      credentials: HTTPAuthorizationCredentials = Depends(security),
                      user: Dict = Depends(get_current_user),
                      _: bool = Depends(admin_role)):
    """Create a new model (admin only)"""
    try:
        # Implementation for model creation
        # This would involve training or loading a new model
        
        return {"status": "Model creation initiated"}
    except Exception as e:
        logger.error(f"Model creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper functions
async def get_model(model_id: str):
    """Get or load model from cache"""
    if model_id in model_cache:
        return model_cache[model_id]
    
    # In production, load from model storage
    if model_id == "qllm-base-4q":
        model = QLLMBase(
            vocab_size=1000,
            embedding_dim=16,
            n_qubits=4,
            n_layers=2,
            device="default.qubit"
        )
    elif model_id == "qllm-advanced-8q":
        model = QLLMAdvanced(
            vocab_size=5000,
            embedding_dim=32,
            n_qubits=8,
            n_layers=4,
            n_heads=4,
            device="default.qubit",
            use_quantum_attention=True
        )
    else:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Cache the model
    model_cache[model_id] = model
    return model

async def record_generation_analytics(user_id: str, model_id: str, prompt_length: int, 
                                     generated_length: int, processing_time: float):
    """Record analytics for text generation"""
    # In production, store in database
    analytics = {
        "user_id": user_id,
        "model_id": model_id,
        "prompt_length": prompt_length,
        "generated_length": generated_length,
        "processing_time": processing_time,
        "timestamp": int(time.time())
    }
    
    logger.info(f"Generation analytics: {json.dumps(analytics)}")

# Run the API server
def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api_server()