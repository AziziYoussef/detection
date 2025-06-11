"""
WebSocket Package for Real-time Communication
Contains WebSocket handlers and routing
"""

# Import WebSocket components
__all__ = []

try:
    from . import stream_handler
    __all__.append("stream_handler")
except ImportError:
    pass

try:
    from .stream_handler import websocket_router
    __all__.append("websocket_router")
except ImportError:
    pass

# WebSocket configuration
WEBSOCKET_CONFIG = {
    "ping_interval": 20,
    "ping_timeout": 10,
    "max_size": 10 * 1024 * 1024,  # 10MB
    "max_queue": 32
}