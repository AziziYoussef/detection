"""
📋 CONFIGURATION - MODULE D'INITIALISATION
============================================
Point d'entrée pour toutes les configurations du service IA

Exports:
- get_settings(): Configuration globale
- get_model_config(): Configuration des modèles
- get_api_config(): Configuration de l'API
- Constantes et énumérations communes
"""

from .config import get_settings, Settings
from .model_config import get_model_config, ModelConfiguration, ModelType
from .api_config import get_api_config, APIConfiguration

__all__ = [
    "get_settings",
    "Settings", 
    "get_model_config",
    "ModelConfiguration",
    "ModelType",
    "get_api_config", 
    "APIConfiguration"
]
