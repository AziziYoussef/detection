# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.model_manager import ModelManager
from app.config.config import settings

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    logger.info("🚀 Démarrage du service IA de détection d'objets perdus")
    
    # Initialisation du gestionnaire de modèles
    model_manager = ModelManager()
    await model_manager.initialize()
    app.state.model_manager = model_manager
    
    logger.info("✅ Service IA prêt !")
    yield
    
    # Nettoyage lors de l'arrêt
    logger.info("🛑 Arrêt du service IA")
    await model_manager.cleanup()

# Création de l'application FastAPI
app = FastAPI(
    title="🔍 Service IA - Détection d'Objets Perdus",
    description="Service intelligent de détection et surveillance d'objets perdus en temps réel",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inclusion des routes
app.include_router(router, prefix="/api/v1")

# Routes de base
@app.get("/")
async def root():
    """Page d'accueil du service"""
    return {
        "service": "🔍 Détection d'Objets Perdus",
        "version": "1.0.0",
        "status": "✅ Actif",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "detection": "/api/v1/detect/",
            "streaming": "/ws/stream/{client_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de santé du service"""
    try:
        model_manager = app.state.model_manager
        model_status = await model_manager.get_health_status()
        
        return {
            "status": "healthy",
            "timestamp": model_status.get("timestamp"),
            "models": model_status.get("models_loaded"),
            "gpu_available": model_status.get("gpu_available"),
            "memory_usage": model_status.get("memory_usage")
        }
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        raise HTTPException(status_code=500, detail="Service indisponible")

@app.get("/stats")
async def get_stats():
    """Statistiques du service"""
    try:
        model_manager = app.state.model_manager
        stats = await model_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Erreur récupération stats: {e}")
        raise HTTPException(status_code=500, detail="Erreur stats")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )