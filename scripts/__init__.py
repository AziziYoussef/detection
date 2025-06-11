"""
Scripts Package for Utility Scripts
Contains deployment, testing, and maintenance scripts
"""

# Import script modules
__all__ = []

try:
    from . import deploy
    __all__.append("deploy")
except ImportError:
    pass

try:
    from . import test_service
    __all__.append("test_service")
except ImportError:
    pass

try:
    from . import cache_manager
    __all__.append("cache_manager")
except ImportError:
    pass

# Script metadata
SCRIPT_INFO = {
    "deploy": "Service deployment and management",
    "test_service": "Service testing and validation",
    "cache_manager": "Model cache and storage management"
}

# Common script configuration
SCRIPT_CONFIG = {
    "log_level": "INFO",
    "timeout": 300,
    "retry_attempts": 3,
    "verbose": False
}