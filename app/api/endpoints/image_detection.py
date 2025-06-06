"""
📸 SERVICE DE DÉTECTION D'IMAGES STATIQUES
==========================================
Endpoint spécialisé pour la détection d'objets perdus sur images uploadées

Fonctionnalités:
- Upload d'images (JPEG, PNG, WEBP, BMP)
- Détection avec vos modèles PyTorch (Epoch 30 champion)
- Retour des résultats avec bounding boxes et confiances
- Support des seuils de confiance personnalisés
- Optimisation GPU/CPU automatique
- Gestion des erreurs et formats non supportés

Intégration:
- Spring Boot uploade via cet endpoint
- Next.js affiche les résultats visuellement
- Métadonnées sauvegardées en base via Spring Boot
"""

import os
import io
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from PIL import Image
import numpy as np

# Services internes
from app.services.image_service import ImageDetectionService
from app.schemas.detection import DetectionResult, BoundingBox, DetectionResponse
from app.config.config import get_settings
from app.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# 🛣️ CRÉATION DU ROUTER
router = APIRouter()

# 📋 SCHÉMAS DE DONNÉES
class ImageDetectionRequest(BaseModel):
    """📋 Schéma de requête pour détection d'image"""
    confidence_threshold: Optional[float] = Field(
        default=0.5, 
        ge=0.1, 
        le=0.9,
        description="Seuil de confiance (0.1-0.9)"
    )
    model_name: Optional[str] = Field(
        default="epoch_30",
        description="Nom du modèle à utiliser"
    )
    return_annotated_image: Optional[bool] = Field(
        default=False,
        description="Retourner l'image annotée avec les détections"
    )
    save_results: Optional[bool] = Field(
        default=True,
        description="Sauvegarder les résultats pour historique"
    )

class ImageDetectionResponse(BaseModel):
    """📋 Schéma de réponse pour détection d'image"""
    success: bool
    message: str
    detection_id: str
    image_info: Dict[str, Any]
    detections: List[DetectionResult]
    processing_time_ms: float
    model_used: str
    confidence_threshold: float
    annotated_image_url: Optional[str] = None
    statistics: Dict[str, int]

# 🔧 DÉPENDANCES
async def get_image_service() -> ImageDetectionService:
    """Récupère le service de détection d'images"""
    try:
        from main import get_model_service
        model_service = get_model_service()
        return ImageDetectionService(model_service)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service non disponible: {e}")

def validate_image_file(file: UploadFile) -> None:
    """🔍 Valide le fichier image uploadé"""
    settings = get_settings()
    
    # Vérifier l'extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier requis")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=422, 
            detail=f"Format non supporté. Formats acceptés: {settings.SUPPORTED_IMAGE_FORMATS}"
        )
    
    # Vérifier la taille (approximative via content-length)
    if hasattr(file.file, 'seek'):
        file.file.seek(0, 2)  # Aller à la fin
        size = file.file.tell()
        file.file.seek(0)  # Revenir au début
        
        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille max: {settings.MAX_FILE_SIZE_MB}MB"
            )

# 📸 ENDPOINT PRINCIPAL - DÉTECTION SUR IMAGE
@router.post(
    "/detect",
    response_model=ImageDetectionResponse,
    summary="🎯 Détection d'objets sur image",
    description="""
    ## 📸 Détection d'objets perdus sur image statique
    
    **Fonctionnement:**
    1. Upload d'une image (JPEG, PNG, WEBP, BMP)
    2. Détection avec votre modèle Epoch 30 (F1=49.86%)
    3. Retour des objets détectés avec positions et confiances
    
    **Paramètres:**
    - `image`: Fichier image à analyser
    - `confidence_threshold`: Seuil de confiance (défaut: 0.5)
    - `model_name`: Modèle à utiliser (défaut: epoch_30)
    - `return_annotated_image`: Retourner image avec annotations
    
    **Classes détectées:** 28 objets perdus (sacs, téléphones, clés, etc.)
    
    **Utilisé par:** Spring Boot pour traitement, Next.js pour affichage
    """
)
async def detect_objects_in_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Image à analyser"),
    confidence_threshold: float = Form(default=0.5, ge=0.1, le=0.9),
    model_name: str = Form(default="epoch_30"),
    return_annotated_image: bool = Form(default=False),
    save_results: bool = Form(default=True),
    image_service: ImageDetectionService = Depends(get_image_service)
):
    """🎯 Détecte les objets perdus dans une image uploadée"""
    
    start_time = time.time()
    detection_id = f"img_{int(time.time() * 1000)}"
    
    logger.info(f"📸 Nouvelle détection image: {detection_id} - {image.filename}")
    
    try:
        # 🔍 Validation du fichier
        validate_image_file(image)
        
        # 📖 Lecture de l'image
        image_bytes = await image.read()
        
        # 🖼️ Traitement avec PIL pour validation
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_info = {
                "filename": image.filename,
                "format": pil_image.format,
                "size": pil_image.size,
                "mode": pil_image.mode,
                "size_bytes": len(image_bytes)
            }
            
            # Convertir en RGB si nécessaire
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Image corrompue ou invalide: {e}")
        
        # 🤖 Détection avec votre modèle
        logger.info(f"🔍 Début détection avec modèle {model_name}")
        
        detection_result = await image_service.detect_objects(
            image=pil_image,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            detection_id=detection_id
        )
        
        # 📊 Calcul des statistiques
        detection_stats = {
            "total_detections": len(detection_result.detections),
            "person_count": len([d for d in detection_result.detections if d.class_name == "person"]),
            "object_count": len([d for d in detection_result.detections if d.class_name != "person"]),
            "high_confidence": len([d for d in detection_result.detections if d.confidence > 0.7]),
            "classes_detected": len(set(d.class_name for d in detection_result.detections))
        }
        
        # 🎨 Génération de l'image annotée si demandée
        annotated_image_url = None
        if return_annotated_image and detection_result.detections:
            annotated_image_url = await image_service.create_annotated_image(
                image=pil_image,
                detections=detection_result.detections,
                detection_id=detection_id
            )
        
        # ⏱️ Temps de traitement
        processing_time = (time.time() - start_time) * 1000
        
        # 📝 Sauvegarde asynchrone des résultats
        if save_results:
            background_tasks.add_task(
                save_detection_results,
                detection_id=detection_id,
                image_info=image_info,
                detections=detection_result.detections,
                processing_time=processing_time,
                model_name=model_name
            )
        
        logger.info(
            f"✅ Détection terminée {detection_id}: "
            f"{len(detection_result.detections)} objets en {processing_time:.1f}ms"
        )
        
        # 📤 Réponse
        return ImageDetectionResponse(
            success=True,
            message=f"Détection terminée avec succès",
            detection_id=detection_id,
            image_info=image_info,
            detections=detection_result.detections,
            processing_time_ms=processing_time,
            model_used=model_name,
            confidence_threshold=confidence_threshold,
            annotated_image_url=annotated_image_url,
            statistics=detection_stats
        )
        
    except HTTPException:
        # Re-lancer les erreurs HTTP
        raise
        
    except Exception as e:
        logger.error(f"❌ Erreur détection {detection_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la détection: {str(e)}"
        )

# 📄 ENDPOINT - DÉTECTION PAR URL D'IMAGE
@router.post(
    "/detect-url",
    response_model=ImageDetectionResponse,
    summary="🔗 Détection via URL d'image",
    description="""
    ## 🔗 Détection d'objets via URL d'image
    
    **Alternative à l'upload direct:**
    - Fournir une URL d'image accessible
    - Téléchargement et détection automatiques
    - Mêmes fonctionnalités que l'upload direct
    
    **Utilisé pour:** Intégration avec systèmes externes, API tiers
    """
)
async def detect_objects_from_url(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    image_service: ImageDetectionService = Depends(get_image_service)
):
    """🔗 Détecte les objets dans une image via URL"""
    
    detection_id = f"url_{int(time.time() * 1000)}"
    start_time = time.time()
    
    try:
        image_url = request.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail="URL d'image requise")
        
        confidence_threshold = request.get("confidence_threshold", 0.5)
        model_name = request.get("model_name", "epoch_30")
        
        logger.info(f"🔗 Détection via URL {detection_id}: {image_url}")
        
        # 🌐 Téléchargement et détection
        detection_result = await image_service.detect_from_url(
            image_url=image_url,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            detection_id=detection_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "detection_id": detection_id,
            "detections": detection_result.detections,
            "processing_time_ms": processing_time,
            "source_url": image_url
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur détection URL {detection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 📊 ENDPOINT - INFORMATIONS SUR UNE DÉTECTION
@router.get(
    "/detection/{detection_id}",
    summary="📋 Informations détection",
    description="Récupère les informations détaillées d'une détection par son ID"
)
async def get_detection_info(
    detection_id: str,
    image_service: ImageDetectionService = Depends(get_image_service)
):
    """📋 Récupère les informations d'une détection spécifique"""
    
    try:
        detection_info = await image_service.get_detection_details(detection_id)
        
        if not detection_info:
            raise HTTPException(status_code=404, detail="Détection non trouvée")
        
        return {
            "success": True,
            "detection": detection_info
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération détection {detection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🖼️ ENDPOINT - IMAGE ANNOTÉE
@router.get(
    "/annotated/{detection_id}",
    summary="🎨 Image avec annotations",
    description="Récupère l'image annotée avec les détections visualisées"
)
async def get_annotated_image(
    detection_id: str,
    image_service: ImageDetectionService = Depends(get_image_service)
):
    """🎨 Retourne l'image annotée avec les détections"""
    
    try:
        annotated_image_path = await image_service.get_annotated_image_path(detection_id)
        
        if not annotated_image_path or not os.path.exists(annotated_image_path):
            raise HTTPException(status_code=404, detail="Image annotée non trouvée")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            annotated_image_path,
            media_type="image/jpeg",
            filename=f"annotated_{detection_id}.jpg"
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur image annotée {detection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 📈 ENDPOINT - STATISTIQUES DES DÉTECTIONS D'IMAGES
@router.get(
    "/statistics",
    summary="📈 Statistiques détections images",
    description="Statistiques globales des détections d'images effectuées"
)
async def get_image_detection_statistics(
    image_service: ImageDetectionService = Depends(get_image_service)
):
    """📈 Récupère les statistiques des détections d'images"""
    
    try:
        stats = await image_service.get_detection_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🧹 FONCTION UTILITAIRE - SAUVEGARDE ASYNCHRONE
async def save_detection_results(
    detection_id: str,
    image_info: Dict[str, Any],
    detections: List[DetectionResult],
    processing_time: float,
    model_name: str
):
    """💾 Sauvegarde asynchrone des résultats de détection"""
    
    try:
        # Ici vous pouvez implémenter la sauvegarde
        # - Base de données locale
        # - Envoi vers Spring Boot
        # - Cache Redis
        # - Fichiers JSON
        
        logger.info(f"💾 Résultats sauvegardés pour {detection_id}")
        
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde {detection_id}: {e}")

# 📝 INFORMATIONS D'EXPORT
__all__ = ["router"]