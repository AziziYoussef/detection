"""
🛣️ API PACKAGE - INITIALISATION
==================================
Package central pour tous les endpoints de l'API de détection

Ce package organise toutes les routes de l'API:
- Image detection endpoints
- Video processing endpoints  
- Real-time streaming endpoints
- Batch processing endpoints
- Model management endpoints

Structure:
- routes.py: Router principal qui inclut tous les sous-routers
- endpoints/: Dossier avec les endpoints spécialisés par service
- websocket/: Gestion WebSocket pour streaming temps réel
"""

from .routes import api_router

__all__ = ["api_router"]