"""
🛣️ API ROUTES - ROUTER PRINCIPAL
==================================
Organisation centrale de toutes les routes de l'API de détection d'objets perdus

Ce fichier regroupe tous les endpoints par service:
📸 Image Detection - Détection sur images statiques
🎬 Video Processing - Traitement de vidéos uploadées
📡 Stream Detection - Détection temps réel via WebSocket
📦 Batch Processing - Traitement en lot de multiples fichiers
🤖 Model Management - Gestion et monitoring des modèles

Intégration avec votre architecture:
- Spring Boot appelle ces endpoints via HTTP
- Next.js utilise ces API pour les interfaces utilisateur
- WebSocket pour les fonctionnalités temps réel
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import logging

# Imports des routers spécialisés
from app.api.endpoints.image_detection import router as image_router
from app.api.endpoints.video_detection import router as video_router
from app.api.endpoints.stream_detection import router as stream_router
from app.api.endpoints.batch_detection import router as batch_router

# Services et dépendances
from app.services.model_service import ModelService
from app.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# 🏗️ CRÉATION DU ROUTER PRINCIPAL
api_router = APIRouter()

# 📸 ROUTES DE DÉTECTION D'IMAGES
api_router.include_router(
    image_router,
    prefix="/detect/image",
    tags=["📸 Image Detection"],
    responses={
        404: {"description": "Image non trouvée"},
        422: {"description": "Format d'image non supporté"},
        500: {"description": "Erreur de traitement"}
    }
)

# 🎬 ROUTES DE TRAITEMENT VIDÉO
api_router.include_router(
    video_router,
    prefix="/detect/video",
    tags=["🎬 Video Processing"],
    responses={
        404: {"description": "Vidéo non trouvée"},
        413: {"description": "Fichier vidéo trop volumineux"},
        422: {"description": "Format vidéo non supporté"},
        500: {"description": "Erreur de traitement vidéo"}
    }
)

# 📡 ROUTES DE STREAMING TEMPS RÉEL
api_router.include_router(
    stream_router,
    prefix="/detect/stream",
    tags=["📡 Real-time Streaming"],
    responses={
        426: {"description": "WebSocket requis"},
        500: {"description": "Erreur de streaming"}
    }
)

# 📦 ROUTES DE TRAITEMENT BATCH
api_router.include_router(
    batch_router,
    prefix="/detect/batch",
    tags=["📦 Batch Processing"],
    responses={
        413: {"description": "Trop de fichiers"},
        422: {"description": "Format de batch invalide"},
        500: {"description": "Erreur de traitement batch"}
    }
)

# 🤖 ROUTES DE GESTION DES MODÈLES
@api_router.get(
    "/models/status",
    tags=["🤖 Model Management"],
    summary="État des modèles chargés",
    description="""
    📊 **Informations sur l'état des modèles de détection**
    
    Retourne:
    - État de chargement de chaque modèle
    - Performance et métriques (F1-Score, précision)  
    - Utilisation GPU/CPU et mémoire
    - Temps de réponse moyen
    
    Utilisé par Spring Boot pour monitoring.
    """
)
async def get_models_status():
    """🤖 Récupère l'état actuel de tous les modèles"""
    try:
        from main import get_model_manager
        model_manager = get_model_manager()
        
        status = await model_manager.get_models_status()
        
        return {
            "status": "success",
            "models": status,
            "timestamp": model_manager.get_current_timestamp(),
            "gpu_available": model_manager.device.type == "cuda",
            "memory_info": model_manager.get_memory_usage()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération statut modèles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur récupération statut: {str(e)}"
        )

@api_router.post(
    "/models/reload",
    tags=["🤖 Model Management"],
    summary="Recharger les modèles",
    description="""
    🔄 **Rechargement à chaud des modèles**
    
    Permet de recharger les modèles sans redémarrer le service:
    - Utile après mise à jour des fichiers .pth
    - Libère la mémoire GPU avant rechargement
    - Vérifie l'intégrité des nouveaux modèles
    
    ⚠️ Attention: Le service sera brièvement indisponible pendant le rechargement.
    """
)
async def reload_models():
    """🔄 Recharge tous les modèles à chaud"""
    try:
        from main import get_model_manager
        model_manager = get_model_manager()
        
        logger.info("🔄 Début du rechargement des modèles...")
        
        # Sauvegarde des statistiques actuelles
        old_stats = await model_manager.get_models_status()
        
        # Rechargement
        reload_result = await model_manager.reload_all_models()
        
        logger.info("✅ Rechargement des modèles terminé")
        
        return {
            "status": "success",
            "message": "Modèles rechargés avec succès",
            "old_models": old_stats,
            "new_models": reload_result,
            "timestamp": model_manager.get_current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur rechargement modèles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur rechargement: {str(e)}"
        )

@api_router.get(
    "/models/performance",
    tags=["🤖 Model Management"],
    summary="Métriques de performance",
    description="""
    📈 **Métriques détaillées de performance des modèles**
    
    Informations incluses:
    - Temps de traitement moyen par type (image/vidéo/stream)
    - Nombre de détections effectuées
    - Taux de succès/échec
    - Utilisation ressources (GPU/CPU/RAM)
    - Historique des performances
    
    Utile pour optimisation et monitoring.
    """
)
async def get_models_performance():
    """📈 Récupère les métriques détaillées de performance"""
    try:
        from main import get_model_service
        model_service = get_model_service()
        
        performance_data = await model_service.get_detailed_performance_metrics()
        
        return {
            "status": "success",
            "performance": performance_data,
            "timestamp": model_service.get_current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération métriques: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur métriques: {str(e)}"
        )

# 🔧 ROUTES DE CONFIGURATION ET UTILITIES
@api_router.get(
    "/config",
    tags=["🔧 Configuration"],
    summary="Configuration du service",
    description="""
    ⚙️ **Configuration actuelle du service de détection**
    
    Informations retournées:
    - Classes détectables et leurs noms français
    - Seuils de confiance configurés
    - Formats de fichiers supportés
    - Limites de taille et durée
    - Paramètres GPU/CPU
    
    Utilisé par le frontend pour adapter l'interface.
    """
)
async def get_service_config():
    """⚙️ Récupère la configuration actuelle du service"""
    try:
        from app.config.config import get_settings
        settings = get_settings()
        
        return {
            "status": "success",
            "config": {
                "service_name": settings.SERVICE_NAME,
                "version": settings.VERSION,
                "classes": settings.DETECTION_CLASSES,
                "classes_fr": settings.CLASSES_FR_NAMES,
                "confidence_threshold": settings.DEFAULT_CONFIDENCE_THRESHOLD,
                "supported_image_formats": settings.SUPPORTED_IMAGE_FORMATS,
                "supported_video_formats": settings.SUPPORTED_VIDEO_FORMATS,
                "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
                "max_video_duration_sec": settings.MAX_VIDEO_DURATION_SEC,
                "gpu_enabled": settings.GPU_ENABLED,
                "batch_size_limit": settings.MAX_BATCH_SIZE
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur configuration: {str(e)}"
        )

# 🏥 ROUTE DE SANTÉ DÉTAILLÉE
@api_router.get(
    "/health/detailed",
    tags=["🏥 Health Check"],
    summary="Vérification santé détaillée",
    description="""
    🏥 **Vérification complète de l'état du service**
    
    Contrôles effectués:
    - État des modèles et leur disponibilité
    - Connectivité GPU/CUDA si applicable
    - Espace disque disponible
    - Mémoire disponible
    - Performances récentes
    - État des connexions WebSocket actives
    
    Plus détaillé que `/health` du main.py
    """
)
async def detailed_health_check():
    """🏥 Vérification de santé détaillée du service"""
    try:
        from main import get_model_manager, get_websocket_manager
        
        model_manager = get_model_manager()
        websocket_manager = get_websocket_manager()
        
        # Vérifications détaillées
        health_status = {
            "status": "healthy",
            "timestamp": model_manager.get_current_timestamp(),
            "service": "AI Detection Service",
            "checks": {
                "models": await model_manager.health_check(),
                "gpu": model_manager.check_gpu_health(),
                "memory": model_manager.get_memory_usage(),
                "disk_space": model_manager.check_disk_space(),
                "websockets": websocket_manager.get_connection_stats(),
                "recent_performance": await model_manager.get_recent_performance()
            }
        }
        
        # Déterminer l'état global
        all_checks_ok = all(
            check.get("status") == "ok" 
            for check in health_status["checks"].values()
            if isinstance(check, dict) and "status" in check
        )
        
        if not all_checks_ok:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur health check détaillé: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "unknown"
            }
        )

# 📊 ROUTE DE STATISTIQUES GLOBALES
@api_router.get(
    "/statistics",
    tags=["📊 Statistics"],
    summary="Statistiques globales du service",
    description="""
    📊 **Statistiques complètes du service de détection**
    
    Données incluses:
    - Nombre total de détections par type
    - Objets les plus fréquemment détectés
    - Temps de traitement moyens
    - Évolution des performances
    - Utilisation des ressources dans le temps
    - Statistiques par modèle utilisé
    
    Parfait pour tableaux de bord et analytics.
    """
)
async def get_global_statistics():
    """📊 Récupère les statistiques globales du service"""
    try:
        from main import get_model_service
        model_service = get_model_service()
        
        stats = await model_service.get_comprehensive_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "generated_at": model_service.get_current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération statistiques: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur statistiques: {str(e)}"
        )

# ❌ GESTIONNAIRE D'ERREURS POUR LES ROUTES API
@api_router.exception_handler(HTTPException)
async def api_exception_handler(request, exc: HTTPException):
    """❌ Gestionnaire d'erreurs spécifique aux routes API"""
    logger.error(f"❌ Erreur API: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url),
            "method": request.method
        }
    )

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "api_router"
]