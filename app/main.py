#!/usr/bin/env python3
"""
 AI SERVICE - POINT D'ENTRÉE PRINCIPAL
===========================================
Service de détection d'objets perdus utilisant vos modèles PyTorch
Intégration avec Spring Boot backend et Next.js frontend

Architecture:
- FastAPI pour l'API REST haute performance
- WebSocket pour streaming temps réel
- Gestion asynchrone des tâches longues
- Support multi-modèles avec cache intelligent
- CORS configuré pour intégration web

Version: 1.0.0
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

# FastAPI et dépendances
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websocket import WebSocket, WebSocketDisconnect

# Configuration et logging
import uvicorn
from pydantic import BaseSettings

# Vos modules internes
from app.core.model_manager import ModelManager
from app.config.config import get_settings
from app.api.routes import api_router
from app.api.websocket.stream_handler import WebSocketManager
from app.services.model_service import ModelService

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_service.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Gestionnaire global des modèles et WebSocket
model_manager: ModelManager = None
websocket_manager: WebSocketManager = None
model_service: ModelService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🚀 CYCLE DE VIE DE L'APPLICATION
    Initialisation et nettoyage des ressources
    """
    # 🔥 DÉMARRAGE DE L'APPLICATION
    logger.info("🚀 Démarrage du service IA de détection d'objets perdus")
    
    global model_manager, websocket_manager, model_service
    
    try:
        # 1. Chargement de la configuration
        settings = get_settings()
        logger.info(f"📋 Configuration chargée: {settings.SERVICE_NAME}")
        
        # 2. Initialisation du gestionnaire de modèles
        logger.info("🤖 Initialisation du gestionnaire de modèles...")
        model_manager = ModelManager(settings)
        await model_manager.initialize()
        
        # 3. Chargement des modèles principaux
        logger.info("📦 Chargement des modèles de détection...")
        await model_manager.load_champion_models()
        
        # 4. Initialisation du service de modèles
        model_service = ModelService(model_manager)
        
        # 5. Initialisation du gestionnaire WebSocket
        logger.info("📡 Initialisation du gestionnaire WebSocket...")
        websocket_manager = WebSocketManager(model_service)
        
        # 6. Stockage dans l'état de l'application
        app.state.model_manager = model_manager
        app.state.websocket_manager = websocket_manager
        app.state.model_service = model_service
        app.state.settings = settings
        
        # 7. Vérification de l'état des modèles
        model_status = await model_manager.get_models_status()
        logger.info(f"✅ Modèles chargés: {model_status}")
        
        logger.info("🎉 Service IA démarré avec succès!")
        
        yield  # Point où l'application tourne
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {e}")
        raise
    
    # 🛑 ARRÊT DE L'APPLICATION
    logger.info("🛑 Arrêt du service IA...")
    
    try:
        # Nettoyage des ressources
        if websocket_manager:
            await websocket_manager.cleanup()
            
        if model_manager:
            await model_manager.cleanup()
            
        logger.info("✅ Nettoyage terminé")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt: {e}")

# 🏗️ CRÉATION DE L'APPLICATION FASTAPI
def create_application() -> FastAPI:
    """
    🏗️ FACTORY POUR CRÉER L'APPLICATION FASTAPI
    Configure tous les middleware et routes
    """
    
    settings = get_settings()
    
    # Création de l'app avec cycle de vie
    app = FastAPI(
        title="🤖 Lost Objects Detection AI Service",
        description="""
        ## 🎯 Service IA pour la détection d'objets perdus
        
        Ce service utilise vos modèles PyTorch entraînés pour détecter des objets perdus:
        - 📸 **Images statiques**: Upload et détection instantanée
        - 🎬 **Vidéos**: Traitement complet avec timeline des détections  
        - 📡 **Stream temps réel**: WebSocket pour caméra/webcam
        - 📦 **Traitement batch**: Multiples fichiers en parallèle
        
        ### 🏆 Modèles disponibles:
        - **Epoch 30**: F1=49.86%, Précision=60.73% (Champion)
        - **Extended**: 28 classes d'objets perdus
        - **Optimisé GPU**: Inference temps réel
        
        ### 🔗 Intégration:
        - **Backend**: Spring Boot REST API
        - **Frontend**: Next.js WebApp  
        - **Database**: Historique et métadonnées
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # 🌐 CONFIGURATION CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Next.js dev
            "http://localhost:8080",  # Spring Boot dev
            "http://frontend:3000",   # Docker frontend
            "http://backend:8080",    # Docker backend
            settings.FRONTEND_URL,    # Production frontend
            settings.BACKEND_URL,     # Production backend
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # 📦 COMPRESSION GZIP
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 📁 FICHIERS STATIQUES
    if os.path.exists("storage/temp/results"):
        app.mount("/results", StaticFiles(directory="storage/temp/results"), name="results")
    
    # 🛣️ INCLUSION DES ROUTES
    app.include_router(api_router, prefix="/api/v1")
    
    return app

# Instance de l'application
app = create_application()

# 🔧 DÉPENDANCES GLOBALES
def get_model_manager() -> ModelManager:
    """Récupère le gestionnaire de modèles"""
    if not hasattr(app.state, 'model_manager') or app.state.model_manager is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")
    return app.state.model_manager

def get_model_service() -> ModelService:
    """Récupère le service de modèles"""
    if not hasattr(app.state, 'model_service') or app.state.model_service is None:
        raise HTTPException(status_code=503, detail="Service de modèles non disponible")
    return app.state.model_service

def get_websocket_manager() -> WebSocketManager:
    """Récupère le gestionnaire WebSocket"""
    if not hasattr(app.state, 'websocket_manager') or app.state.websocket_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket non disponible")
    return app.state.websocket_manager

# 📡 ENDPOINTS WEBSOCKET GLOBAUX
@app.websocket("/ws/stream/{client_id}")
async def websocket_stream_endpoint(
    websocket: WebSocket, 
    client_id: str,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    📡 ENDPOINT WEBSOCKET POUR STREAMING TEMPS RÉEL
    
    Gère la détection en temps réel via WebSocket:
    - Connexion client unique par ID
    - Réception frames webcam/caméra
    - Détection asynchrone
    - Envoi résultats en temps réel
    """
    logger.info(f"📡 Nouvelle connexion WebSocket: {client_id}")
    
    try:
        await websocket_manager.connect(websocket, client_id)
        
        while True:
            # Réception des données (frame ou commande)
            data = await websocket.receive_json()
            
            # Traitement asynchrone
            await websocket_manager.handle_message(client_id, data)
            
    except WebSocketDisconnect:
        logger.info(f"📡 Déconnexion WebSocket: {client_id}")
        await websocket_manager.disconnect(client_id)
        
    except Exception as e:
        logger.error(f"❌ Erreur WebSocket {client_id}: {e}")
        await websocket_manager.disconnect(client_id)

# 🏥 ENDPOINT DE SANTÉ
@app.get("/health")
async def health_check():
    """
    🏥 VÉRIFICATION DE L'ÉTAT DU SERVICE
    Utilisé par Spring Boot pour monitoring
    """
    try:
        model_manager = get_model_manager()
        model_status = await model_manager.get_models_status()
        
        return {
            "status": "healthy",
            "service": "AI Detection Service",
            "version": "1.0.0",
            "models": model_status,
            "gpu_available": model_manager.device.type == "cuda",
            "memory_usage": model_manager.get_memory_usage()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": "AI Detection Service"
            }
        )

# 📊 ENDPOINT DE STATISTIQUES
@app.get("/stats")
async def get_service_stats(
    model_service: ModelService = Depends(get_model_service)
):
    """
    📊 STATISTIQUES DU SERVICE
    Métriques pour monitoring et debugging
    """
    try:
        stats = await model_service.get_service_statistics()
        return {
            "service_stats": stats,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔧 GESTION GLOBALE DES ERREURS
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    🚨 GESTIONNAIRE D'ERREURS GLOBAL
    Capture toutes les erreurs non gérées
    """
    logger.error(f"❌ Erreur non gérée: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du service IA",
            "detail": str(exc) if app.state.settings.DEBUG else "Contactez l'administrateur",
            "type": type(exc).__name__
        }
    )

# 🚀 POINT D'ENTRÉE PRINCIPAL
if __name__ == "__main__":
    """
    🚀 LANCEMENT DU SERVICE EN MODE DÉVELOPPEMENT
    Pour production, utilisez: gunicorn ou uvicorn avec workers
    """
    
    # Configuration depuis variables d'environnement
    HOST = os.getenv("AI_SERVICE_HOST", "0.0.0.0")
    PORT = int(os.getenv("AI_SERVICE_PORT", "8001"))
    WORKERS = int(os.getenv("AI_SERVICE_WORKERS", "1"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"🚀 Lancement du service IA sur {HOST}:{PORT}")
    logger.info(f"🔧 Mode debug: {DEBUG}")
    logger.info(f"👥 Workers: {WORKERS}")
    
    # Configuration uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": HOST,
        "port": PORT,
        "reload": DEBUG,
        "log_level": "info" if not DEBUG else "debug",
        "access_log": True,
        "workers": 1 if DEBUG else WORKERS,  # 1 worker en debug pour reload
    }
    
    # Lancement du serveur
    uvicorn.run(**uvicorn_config)

# 📝 CONFIGURATION POUR GUNICORN (PRODUCTION)
"""
🏭 POUR LANCEMENT EN PRODUCTION:

gunicorn main:app \
    --bind 0.0.0.0:8001 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile -

OU avec Docker:
docker run -p 8001:8001 ai-service:latest
"""