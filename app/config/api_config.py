"""
🌐 CONFIGURATION API - SETTINGS SPÉCIALISÉS
============================================
Configuration dédiée à l'API FastAPI et aux endpoints

Fonctionnalités:
- Configuration des endpoints et routes
- Limitations de taux et quotas
- Configuration CORS détaillée
- Paramètres de sécurité API
- Configuration WebSocket
- Validation des requêtes
"""

from typing import Dict, List, Optional, Any, Set
from functools import lru_cache
from pydantic import BaseSettings, Field, validator
from enum import Enum

# 🏷️ ÉNUMÉRATIONS
class APIVersion(str, Enum):
    """📌 Versions d'API supportées"""
    V1 = "v1"
    V2 = "v2"

class RateLimitType(str, Enum):
    """🚦 Types de limitation de taux"""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"

class EndpointType(str, Enum):
    """🛣️ Types d'endpoints"""
    DETECTION = "detection"
    BATCH = "batch"
    STREAM = "stream"
    MANAGEMENT = "management"
    MONITORING = "monitoring"

# ⚙️ CONFIGURATION API
class APIConfiguration(BaseSettings):
    """🌐 Configuration de l'API FastAPI"""
    
    # 📌 INFORMATIONS API
    API_TITLE: str = Field(default="🤖 Lost Objects Detection AI API", env="API_TITLE")
    API_DESCRIPTION: str = Field(default="Service IA pour la détection d'objets perdus", env="API_DESCRIPTION")
    API_VERSION: str = Field(default="1.0.0", env="API_VERSION")
    API_PREFIX: str = Field(default="/api", env="API_PREFIX")
    
    # 🛣️ CONFIGURATION DES ROUTES
    CURRENT_API_VERSION: APIVersion = Field(default=APIVersion.V1, env="CURRENT_API_VERSION")
    ENABLE_DOCS: bool = Field(default=True, env="ENABLE_DOCS")
    DOCS_URL: str = Field(default="/docs", env="DOCS_URL")
    REDOC_URL: str = Field(default="/redoc", env="REDOC_URL")
    OPENAPI_URL: str = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # 🚦 LIMITATION DE TAUX (RATE LIMITING)
    RATE_LIMITING_ENABLED: bool = Field(default=True, env="RATE_LIMITING_ENABLED")
    RATE_LIMIT_DETECTION: int = Field(default=100, env="RATE_LIMIT_DETECTION")  # par minute
    RATE_LIMIT_BATCH: int = Field(default=10, env="RATE_LIMIT_BATCH")  # par minute
    RATE_LIMIT_STREAM: int = Field(default=5, env="RATE_LIMIT_STREAM")  # connexions simultanées
    RATE_LIMIT_MANAGEMENT: int = Field(default=20, env="RATE_LIMIT_MANAGEMENT")  # par minute
    
    # 🛡️ SÉCURITÉ API
    REQUIRE_API_KEY: bool = Field(default=False, env="REQUIRE_API_KEY")
    API_KEYS: Set[str] = Field(default=set(), env="API_KEYS")
    API_KEY_HEADER: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    ENABLE_CORS: bool = Field(default=True, env="ENABLE_CORS")
    
    # 🌐 CONFIGURATION CORS DÉTAILLÉE
    CORS_ALLOW_ORIGINS: List[str] = Field(default=["*"], env="CORS_ALLOW_ORIGINS")
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"], 
        env="CORS_ALLOW_METHODS"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    CORS_MAX_AGE: int = Field(default=3600, env="CORS_MAX_AGE")
    
    # 📤 CONFIGURATION UPLOAD
    MAX_UPLOAD_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 50MB
    ALLOWED_MIME_TYPES: List[str] = Field(
        default=[
            "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
            "video/mp4", "video/avi", "video/mov", "video/mkv", "video/webm"
        ],
        env="ALLOWED_MIME_TYPES"
    )
    
    # 📡 CONFIGURATION WEBSOCKET
    WEBSOCKET_ENABLED: bool = Field(default=True, env="WEBSOCKET_ENABLED")
    WEBSOCKET_MAX_CONNECTIONS: int = Field(default=50, env="WEBSOCKET_MAX_CONNECTIONS")
    WEBSOCKET_PING_INTERVAL: int = Field(default=30, env="WEBSOCKET_PING_INTERVAL")
    WEBSOCKET_PING_TIMEOUT: int = Field(default=60, env="WEBSOCKET_PING_TIMEOUT")
    WEBSOCKET_MAX_MESSAGE_SIZE: int = Field(default=10 * 1024 * 1024, env="WEBSOCKET_MAX_MESSAGE_SIZE")  # 10MB
    
    # ⏱️ TIMEOUTS
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    DETECTION_TIMEOUT: int = Field(default=60, env="DETECTION_TIMEOUT")  # 1 minute
    BATCH_TIMEOUT: int = Field(default=600, env="BATCH_TIMEOUT")  # 10 minutes
    STREAM_TIMEOUT: int = Field(default=30, env="STREAM_TIMEOUT")  # 30 secondes
    
    # 📊 MONITORING ET MÉTRIQUES
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_ENDPOINT: str = Field(default="/metrics", env="METRICS_ENDPOINT")
    HEALTH_ENDPOINT: str = Field(default="/health", env="HEALTH_ENDPOINT")
    STATUS_ENDPOINT: str = Field(default="/status", env="STATUS_ENDPOINT")
    
    # 🔧 CONFIGURATION AVANCÉE
    ENABLE_COMPRESSION: bool = Field(default=True, env="ENABLE_COMPRESSION")
    COMPRESSION_LEVEL: int = Field(default=6, env="COMPRESSION_LEVEL")
    ENABLE_CACHING: bool = Field(default=True, env="ENABLE_CACHING")
    CACHE_CONTROL_MAX_AGE: int = Field(default=300, env="CACHE_CONTROL_MAX_AGE")
    
    # 📝 LOGGING SPÉCIFIQUE API
    LOG_REQUESTS: bool = Field(default=True, env="LOG_REQUESTS")
    LOG_RESPONSES: bool = Field(default=False, env="LOG_RESPONSES")  # Peut être volumineux
    LOG_SLOW_REQUESTS: bool = Field(default=True, env="LOG_SLOW_REQUESTS")
    SLOW_REQUEST_THRESHOLD: float = Field(default=1.0, env="SLOW_REQUEST_THRESHOLD")  # secondes
    
    # 🔄 RETRY ET CIRCUIT BREAKER
    ENABLE_RETRY: bool = Field(default=True, env="ENABLE_RETRY")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    RETRY_DELAY: float = Field(default=1.0, env="RETRY_DELAY")
    CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, env="CIRCUIT_BREAKER_ENABLED")
    CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    # 🔍 VALIDATORS
    @validator("CURRENT_API_VERSION")
    def validate_api_version(cls, v):
        if isinstance(v, str):
            return APIVersion(v.lower())
        return v
    
    @validator("RATE_LIMIT_DETECTION", "RATE_LIMIT_BATCH", "RATE_LIMIT_MANAGEMENT")
    def validate_rate_limits(cls, v):
        if v < 1:
            raise ValueError("Rate limit doit être >= 1")
        return v
    
    @validator("COMPRESSION_LEVEL")
    def validate_compression_level(cls, v):
        if not 1 <= v <= 9:
            raise ValueError("Compression level doit être entre 1 et 9")
        return v
    
    @validator("API_KEYS", pre=True)
    def validate_api_keys(cls, v):
        if isinstance(v, str):
            return set(v.split(","))
        return v
    
    # 🛠️ MÉTHODES UTILITAIRES
    def get_full_api_prefix(self) -> str:
        """🛣️ Retourne le préfixe complet de l'API"""
        return f"{self.API_PREFIX}/{self.CURRENT_API_VERSION.value}"
    
    def get_endpoint_url(self, endpoint: str) -> str:
        """🛣️ Retourne l'URL complète d'un endpoint"""
        return f"{self.get_full_api_prefix()}/{endpoint.lstrip('/')}"
    
    def get_rate_limit_for_endpoint(self, endpoint_type: EndpointType) -> int:
        """🚦 Retourne la limite de taux pour un type d'endpoint"""
        rate_limit_mapping = {
            EndpointType.DETECTION: self.RATE_LIMIT_DETECTION,
            EndpointType.BATCH: self.RATE_LIMIT_BATCH,
            EndpointType.STREAM: self.RATE_LIMIT_STREAM,
            EndpointType.MANAGEMENT: self.RATE_LIMIT_MANAGEMENT,
            EndpointType.MONITORING: self.RATE_LIMIT_MANAGEMENT * 2  # Plus permissif
        }
        return rate_limit_mapping.get(endpoint_type, 60)  # Default: 60/min
    
    def get_timeout_for_endpoint(self, endpoint_type: EndpointType) -> int:
        """⏱️ Retourne le timeout pour un type d'endpoint"""
        timeout_mapping = {
            EndpointType.DETECTION: self.DETECTION_TIMEOUT,
            EndpointType.BATCH: self.BATCH_TIMEOUT,
            EndpointType.STREAM: self.STREAM_TIMEOUT,
            EndpointType.MANAGEMENT: self.REQUEST_TIMEOUT,
            EndpointType.MONITORING: 30  # Monitoring rapide
        }
        return timeout_mapping.get(endpoint_type, self.REQUEST_TIMEOUT)
    
    def is_api_key_valid(self, api_key: str) -> bool:
        """🔑 Vérifie la validité d'une clé API"""
        if not self.REQUIRE_API_KEY:
            return True
        return api_key in self.API_KEYS
    
    def get_cors_settings(self) -> Dict[str, Any]:
        """🌐 Retourne les paramètres CORS"""
        return {
            "allow_origins": self.CORS_ALLOW_ORIGINS,
            "allow_methods": self.CORS_ALLOW_METHODS,
            "allow_headers": self.CORS_ALLOW_HEADERS,
            "allow_credentials": self.CORS_ALLOW_CREDENTIALS,
            "max_age": self.CORS_MAX_AGE
        }
    
    def get_middleware_config(self) -> Dict[str, Any]:
        """🔧 Retourne la configuration des middleware"""
        return {
            "compression": {
                "enabled": self.ENABLE_COMPRESSION,
                "level": self.COMPRESSION_LEVEL
            },
            "cors": self.get_cors_settings(),
            "rate_limiting": {
                "enabled": self.RATE_LIMITING_ENABLED,
                "limits": {
                    "detection": self.RATE_LIMIT_DETECTION,
                    "batch": self.RATE_LIMIT_BATCH,
                    "stream": self.RATE_LIMIT_STREAM,
                    "management": self.RATE_LIMIT_MANAGEMENT
                }
            },
            "security": {
                "require_api_key": self.REQUIRE_API_KEY,
                "api_key_header": self.API_KEY_HEADER
            }
        }
    
    def get_documentation_config(self) -> Dict[str, Any]:
        """📚 Retourne la configuration de la documentation"""
        return {
            "title": self.API_TITLE,
            "description": self.API_DESCRIPTION,
            "version": self.API_VERSION,
            "docs_url": self.DOCS_URL if self.ENABLE_DOCS else None,
            "redoc_url": self.REDOC_URL if self.ENABLE_DOCS else None,
            "openapi_url": self.OPENAPI_URL if self.ENABLE_DOCS else None
        }

# 🏭 FACTORY FUNCTION
@lru_cache()
def get_api_config() -> APIConfiguration:
    """🏭 Factory pour récupérer la configuration de l'API"""
    return APIConfiguration()

# 🛠️ FONCTIONS UTILITAIRES
def get_current_api_version() -> APIVersion:
    """📌 Retourne la version actuelle de l'API"""
    return get_api_config().CURRENT_API_VERSION

def get_api_endpoints() -> Dict[str, str]:
    """🛣️ Retourne la liste des endpoints principaux"""
    config = get_api_config()
    base_url = config.get_full_api_prefix()
    
    return {
        "detection": f"{base_url}/detect",
        "batch_detection": f"{base_url}/detect/batch",
        "video_detection": f"{base_url}/detect/video",
        "stream_detection": f"{base_url}/detect/stream",
        "models": f"{base_url}/models",
        "health": config.HEALTH_ENDPOINT,
        "metrics": config.METRICS_ENDPOINT,
        "status": config.STATUS_ENDPOINT
    }

def is_endpoint_enabled(endpoint_type: EndpointType) -> bool:
    """✅ Vérifie si un type d'endpoint est activé"""
    config = get_api_config()
    
    # Logique pour déterminer si un endpoint est activé
    enabled_mapping = {
        EndpointType.DETECTION: True,  # Toujours activé
        EndpointType.BATCH: True,      # Toujours activé
        EndpointType.STREAM: config.WEBSOCKET_ENABLED,
        EndpointType.MANAGEMENT: True,  # Toujours activé
        EndpointType.MONITORING: config.ENABLE_METRICS
    }
    
    return enabled_mapping.get(endpoint_type, True)