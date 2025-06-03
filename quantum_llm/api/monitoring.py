"""
Monitoring and logging module for Quantum LLM API.
Provides enterprise-grade monitoring, logging, and alerting capabilities.
"""

import os
import time
import json
import logging
import threading
import contextlib
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantum_llm.api.monitoring")

# In production, use a proper time series database or monitoring service
# This is a simplified in-memory implementation
performance_metrics = {
    "requests": {
        "total": 0,
        "success": 0,
        "error": 0,
        "by_endpoint": {}
    },
    "latency": {
        "avg": 0,
        "p50": 0,
        "p95": 0,
        "p99": 0
    },
    "resources": {
        "cpu": [],
        "memory": [],
        "quantum_resources": []
    },
    "users": {
        "active": set(),
        "requests_by_user": {}
    }
}

# Latency measurements for percentile calculations
latency_measurements = []

# Lock for thread-safe updates
metrics_lock = threading.Lock()

# Alert thresholds
alert_thresholds = {
    "error_rate": 0.05,  # 5% error rate
    "p95_latency": 1000,  # 1000ms
    "memory_usage": 0.9,  # 90% memory usage
    "cpu_usage": 0.8      # 80% CPU usage
}

# Alert callbacks
alert_callbacks = []

def register_alert_callback(callback: Callable[[str, Any], None]):
    """
    Register a callback for alerts.
    
    Args:
        callback: Callback function that takes alert type and data
    """
    alert_callbacks.append(callback)

def trigger_alert(alert_type: str, data: Any):
    """
    Trigger an alert.
    
    Args:
        alert_type: Type of alert
        data: Alert data
    """
    logger.warning(f"ALERT: {alert_type} - {data}")
    
    # Call all registered callbacks
    for callback in alert_callbacks:
        try:
            callback(alert_type, data)
        except Exception as e:
            logger.error(f"Error in alert callback: {str(e)}")

def log_request(endpoint: str, user_id: str, request_dict: Optional[Dict] = None):
    """
    Log API request.
    
    Args:
        endpoint: API endpoint
        user_id: User ID
        request_dict: Request data (optional)
    """
    with metrics_lock:
        # Update request metrics
        performance_metrics["requests"]["total"] += 1
        
        if endpoint not in performance_metrics["requests"]["by_endpoint"]:
            performance_metrics["requests"]["by_endpoint"][endpoint] = {
                "total": 0,
                "success": 0,
                "error": 0,
                "latency_sum": 0,
                "latency_count": 0
            }
        
        performance_metrics["requests"]["by_endpoint"][endpoint]["total"] += 1
        
        # Update user metrics
        performance_metrics["users"]["active"].add(user_id)
        
        if user_id not in performance_metrics["users"]["requests_by_user"]:
            performance_metrics["users"]["requests_by_user"][user_id] = 0
        
        performance_metrics["users"]["requests_by_user"][user_id] += 1
    
    # Log request
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "request",
        "endpoint": endpoint,
        "user_id": user_id
    }
    
    if request_dict:
        # Remove sensitive data
        if "password" in request_dict:
            request_dict = request_dict.copy()
            request_dict["password"] = "********"
        
        log_data["request"] = request_dict
    
    logger.info(json.dumps(log_data))

def log_response(endpoint: str, user_id: str, status_code: int, processing_time: Optional[float] = None):
    """
    Log API response.
    
    Args:
        endpoint: API endpoint
        user_id: User ID
        status_code: HTTP status code
        processing_time: Processing time in seconds (optional)
    """
    with metrics_lock:
        # Update response metrics
        if status_code < 400:
            performance_metrics["requests"]["success"] += 1
            if endpoint in performance_metrics["requests"]["by_endpoint"]:
                performance_metrics["requests"]["by_endpoint"][endpoint]["success"] += 1
        else:
            performance_metrics["requests"]["error"] += 1
            if endpoint in performance_metrics["requests"]["by_endpoint"]:
                performance_metrics["requests"]["by_endpoint"][endpoint]["error"] += 1
            
            # Check error rate threshold
            error_rate = performance_metrics["requests"]["error"] / performance_metrics["requests"]["total"]
            if error_rate > alert_thresholds["error_rate"]:
                trigger_alert("high_error_rate", {
                    "error_rate": error_rate,
                    "threshold": alert_thresholds["error_rate"],
                    "endpoint": endpoint
                })
        
        # Update latency metrics
        if processing_time is not None:
            latency_ms = processing_time * 1000  # Convert to milliseconds
            
            latency_measurements.append(latency_ms)
            
            if endpoint in performance_metrics["requests"]["by_endpoint"]:
                endpoint_metrics = performance_metrics["requests"]["by_endpoint"][endpoint]
                endpoint_metrics["latency_sum"] += latency_ms
                endpoint_metrics["latency_count"] += 1
            
            # Check latency threshold
            if len(latency_measurements) >= 100:
                # Calculate percentiles
                sorted_latencies = sorted(latency_measurements)
                p50_idx = int(len(sorted_latencies) * 0.5)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)
                
                performance_metrics["latency"]["avg"] = sum(sorted_latencies) / len(sorted_latencies)
                performance_metrics["latency"]["p50"] = sorted_latencies[p50_idx]
                performance_metrics["latency"]["p95"] = sorted_latencies[p95_idx]
                performance_metrics["latency"]["p99"] = sorted_latencies[p99_idx]
                
                # Check p95 latency threshold
                if performance_metrics["latency"]["p95"] > alert_thresholds["p95_latency"]:
                    trigger_alert("high_latency", {
                        "p95_latency": performance_metrics["latency"]["p95"],
                        "threshold": alert_thresholds["p95_latency"],
                        "endpoint": endpoint
                    })
    
    # Log response
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "response",
        "endpoint": endpoint,
        "user_id": user_id,
        "status_code": status_code
    }
    
    if processing_time is not None:
        log_data["processing_time_ms"] = processing_time * 1000
    
    logger.info(json.dumps(log_data))

def log_error(endpoint: str, user_id: str, error: Exception, request_dict: Optional[Dict] = None):
    """
    Log API error.
    
    Args:
        endpoint: API endpoint
        user_id: User ID
        error: Exception
        request_dict: Request data (optional)
    """
    # Log error
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "error",
        "endpoint": endpoint,
        "user_id": user_id,
        "error": str(error),
        "error_type": type(error).__name__
    }
    
    if request_dict:
        # Remove sensitive data
        if "password" in request_dict:
            request_dict = request_dict.copy()
            request_dict["password"] = "********"
        
        log_data["request"] = request_dict
    
    logger.error(json.dumps(log_data))

def update_resource_metrics(cpu_usage: float, memory_usage: float, quantum_resources: Dict[str, Any]):
    """
    Update resource metrics.
    
    Args:
        cpu_usage: CPU usage (0-1)
        memory_usage: Memory usage (0-1)
        quantum_resources: Quantum resource usage
    """
    with metrics_lock:
        # Update resource metrics
        timestamp = int(time.time())
        
        performance_metrics["resources"]["cpu"].append((timestamp, cpu_usage))
        performance_metrics["resources"]["memory"].append((timestamp, memory_usage))
        performance_metrics["resources"]["quantum_resources"].append((timestamp, quantum_resources))
        
        # Limit history
        max_history = 1000
        if len(performance_metrics["resources"]["cpu"]) > max_history:
            performance_metrics["resources"]["cpu"] = performance_metrics["resources"]["cpu"][-max_history:]
        if len(performance_metrics["resources"]["memory"]) > max_history:
            performance_metrics["resources"]["memory"] = performance_metrics["resources"]["memory"][-max_history:]
        if len(performance_metrics["resources"]["quantum_resources"]) > max_history:
            performance_metrics["resources"]["quantum_resources"] = performance_metrics["resources"]["quantum_resources"][-max_history:]
        
        # Check resource thresholds
        if cpu_usage > alert_thresholds["cpu_usage"]:
            trigger_alert("high_cpu_usage", {
                "cpu_usage": cpu_usage,
                "threshold": alert_thresholds["cpu_usage"]
            })
        
        if memory_usage > alert_thresholds["memory_usage"]:
            trigger_alert("high_memory_usage", {
                "memory_usage": memory_usage,
                "threshold": alert_thresholds["memory_usage"]
            })

@contextlib.contextmanager
def monitor_performance(operation: str, user_id: str, request_id: str):
    """
    Context manager for monitoring performance.
    
    Args:
        operation: Operation name
        user_id: User ID
        request_id: Request ID
    """
    start_time = time.time()
    
    # Log operation start
    logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "type": "operation_start",
        "operation": operation,
        "user_id": user_id,
        "request_id": request_id
    }))
    
    try:
        yield
    except Exception as e:
        # Log operation error
        logger.error(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "type": "operation_error",
            "operation": operation,
            "user_id": user_id,
            "request_id": request_id,
            "error": str(e),
            "error_type": type(e).__name__
        }))
        raise
    finally:
        # Calculate duration
        duration = time.time() - start_time
        
        # Log operation end
        logger.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "type": "operation_end",
            "operation": operation,
            "user_id": user_id,
            "request_id": request_id,
            "duration_ms": duration * 1000
        }))

def get_performance_metrics():
    """
    Get performance metrics.
    
    Returns:
        dict: Performance metrics
    """
    with metrics_lock:
        # Create a copy of the metrics
        metrics_copy = json.loads(json.dumps(performance_metrics, default=str))
        
        # Calculate additional metrics
        total_requests = performance_metrics["requests"]["total"]
        if total_requests > 0:
            metrics_copy["requests"]["error_rate"] = performance_metrics["requests"]["error"] / total_requests
            metrics_copy["requests"]["success_rate"] = performance_metrics["requests"]["success"] / total_requests
        
        # Calculate endpoint-specific metrics
        for endpoint, endpoint_metrics in metrics_copy["requests"]["by_endpoint"].items():
            if endpoint_metrics["latency_count"] > 0:
                endpoint_metrics["avg_latency"] = endpoint_metrics["latency_sum"] / endpoint_metrics["latency_count"]
            if endpoint_metrics["total"] > 0:
                endpoint_metrics["error_rate"] = endpoint_metrics["error"] / endpoint_metrics["total"]
                endpoint_metrics["success_rate"] = endpoint_metrics["success"] / endpoint_metrics["total"]
        
        # Convert active users set to count
        metrics_copy["users"]["active_count"] = len(performance_metrics["users"]["active"])
        metrics_copy["users"]["active"] = list(performance_metrics["users"]["active"])
        
        return metrics_copy

def reset_performance_metrics():
    """Reset performance metrics."""
    with metrics_lock:
        global performance_metrics, latency_measurements
        
        performance_metrics = {
            "requests": {
                "total": 0,
                "success": 0,
                "error": 0,
                "by_endpoint": {}
            },
            "latency": {
                "avg": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            },
            "resources": {
                "cpu": [],
                "memory": [],
                "quantum_resources": []
            },
            "users": {
                "active": set(),
                "requests_by_user": {}
            }
        }
        
        latency_measurements = []

# Start resource monitoring thread
def start_resource_monitoring(interval: int = 60):
    """
    Start resource monitoring thread.
    
    Args:
        interval: Monitoring interval in seconds
    """
    def monitor_resources():
        while True:
            try:
                # Get CPU usage
                try:
                    import psutil
                    cpu_usage = psutil.cpu_percent() / 100.0
                    memory_usage = psutil.virtual_memory().percent / 100.0
                except ImportError:
                    # Fallback if psutil is not available
                    cpu_usage = 0.5  # Placeholder
                    memory_usage = 0.5  # Placeholder
                
                # Get quantum resources (placeholder)
                quantum_resources = {
                    "qubits_used": 8,
                    "circuits_executed": 100,
                    "quantum_memory_usage": 0.3
                }
                
                # Update metrics
                update_resource_metrics(cpu_usage, memory_usage, quantum_resources)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_resources, daemon=True)
    thread.start()
    
    return thread

# Example alert callback
def email_alert_callback(alert_type: str, data: Any):
    """
    Send email alert.
    
    Args:
        alert_type: Type of alert
        data: Alert data
    """
    # In production, implement actual email sending
    logger.info(f"Would send email alert: {alert_type} - {data}")

# Register default alert callback
register_alert_callback(email_alert_callback)