"""
🎬 SERVICE DE DÉTECTION VIDÉO
=============================
Endpoint spécialisé pour le traitement de vidéos uploadées avec détection d'objets perdus

Fonctionnalités:
- Upload de vidéos (MP4, AVI, MOV, MKV, WEBM)
- Traitement asynchrone pour vidéos longues
- Détection frame par frame avec vos modèles (Epoch 30)
- Timeline des détections avec timestamps
- Modes de vitesse (FAST, BALANCED, QUALITY)
- Génération de vidéo annotée avec détections
- Support GPU/CPU optimisé

Intégration:
- Spring Boot gère l'upload et le suivi des jobs
- Next.js affiche la progression et les résultats
- WebSocket pour notifications temps réel de progression
"""

import os
import asyncio
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import aiofiles

# Services internes
from app.services.video_service import VideoDetectionService
from app.schemas.detection import DetectionResult, VideoDetectionResponse, ProcessingStatus
from app.config.config import get_settings
from app.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# 🛣️ CRÉATION DU ROUTER
router = APIRouter()

# 📋 ÉNUMÉRATIONS ET SCHÉMAS
class ProcessingMode(str, Enum):
    """🎛️ Modes de traitement vidéo"""
    ULTRA_FAST = "ultra_fast"    # 10x plus rapide - Skip 10 frames
    FAST = "fast"                # 5x plus rapide - Skip 5 frames  
    BALANCED = "balanced"        # 3x plus rapide - Skip 3 frames
    QUALITY = "quality"          # Vitesse normale - Toutes les frames

class VideoDetectionRequest(BaseModel):
    """📋 Schéma de requête pour détection vidéo"""
    confidence_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Seuil de confiance (0.1-0.9)"
    )
    processing_mode: Optional[ProcessingMode] = Field(
        default=ProcessingMode.FAST,
        description="Mode de traitement (vitesse vs qualité)"
    )
    model_name: Optional[str] = Field(
        default="epoch_30",
        description="Modèle à utiliser"
    )
    generate_annotated_video: Optional[bool] = Field(
        default=True,
        description="Générer vidéo avec annotations"
    )
    extract_keyframes: Optional[bool] = Field(
        default=True,
        description="Extraire les frames clés avec détections"
    )
    notification_webhook: Optional[str] = Field(
        default=None,
        description="URL webhook pour notifications (Spring Boot)"
    )

class VideoProcessingJob(BaseModel):
    """📋 Schéma d'un job de traitement vidéo"""
    job_id: str
    status: ProcessingStatus
    progress_percentage: float
    video_info: Dict[str, Any]
    current_frame: int
    total_frames: int
    detections_count: int
    processing_time_elapsed: float
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None

# 🗃️ STOCKAGE DES JOBS EN COURS (En production: Redis ou DB)
active_jobs: Dict[str, VideoProcessingJob] = {}

# 🔧 DÉPENDANCES
async def get_video_service() -> VideoDetectionService:
    """Récupère le service de détection vidéo"""
    try:
        from main import get_model_service
        model_service = get_model_service()
        return VideoDetectionService(model_service)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service vidéo non disponible: {e}")

def validate_video_file(file: UploadFile) -> None:
    """🔍 Valide le fichier vidéo uploadé"""
    settings = get_settings()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier requis")
    
    # Vérifier l'extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Format vidéo non supporté. Formats acceptés: {settings.SUPPORTED_VIDEO_FORMATS}"
        )
    
    # Vérifier la taille (approximative)
    content_length = getattr(file, 'size', None)
    if content_length:
        max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
        if content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Vidéo trop volumineuse. Taille max: {settings.MAX_VIDEO_SIZE_MB}MB"
            )

# 🎬 ENDPOINT PRINCIPAL - DÉMARRER TRAITEMENT VIDÉO
@router.post(
    "/process",
    summary="🎬 Traitement vidéo asynchrone",
    description="""
    ## 🎬 Traitement complet d'une vidéo avec détection d'objets
    
    **Fonctionnement:**
    1. Upload de la vidéo (formats: MP4, AVI, MOV, MKV, WEBM)
    2. Traitement asynchrone en arrière-plan
    3. Détection frame par frame avec votre modèle Epoch 30
    4. Génération timeline des détections + vidéo annotée
    
    **Modes de traitement:**
    - `ULTRA_FAST`: 10x plus rapide (skip 10 frames) - Vidéos très longues
    - `FAST`: 5x plus rapide (skip 5 frames) - Équilibre optimal
    - `BALANCED`: 3x plus rapide (skip 3 frames) - Bonne qualité
    - `QUALITY`: Vitesse normale - Toutes les frames analysées
    
    **Retour immédiat:** Job ID pour suivi de progression
    **Notification:** Webhook vers Spring Boot à la fin
    """
)
async def start_video_processing(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Vidéo à analyser"),
    confidence_threshold: float = Form(default=0.5, ge=0.1, le=0.9),
    processing_mode: ProcessingMode = Form(default=ProcessingMode.FAST),
    model_name: str = Form(default="epoch_30"),
    generate_annotated_video: bool = Form(default=True),
    extract_keyframes: bool = Form(default=True),
    notification_webhook: Optional[str] = Form(default=None),
    video_service: VideoDetectionService = Depends(get_video_service)
):
    """🎬 Démarre le traitement asynchrone d'une vidéo"""
    
    job_id = f"video_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"🎬 Nouveau job vidéo: {job_id} - {video.filename}")
    
    try:
        # 🔍 Validation du fichier
        validate_video_file(video)
        
        # 📁 Sauvegarde temporaire du fichier
        settings = get_settings()
        upload_dir = Path(settings.TEMP_UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)
        
        video_path = upload_dir / f"{job_id}_{video.filename}"
        
        # 💾 Écriture asynchrone du fichier
        async with aiofiles.open(video_path, 'wb') as f:
            content = await video.read()
            await f.write(content)
        
        # 📊 Analyse rapide de la vidéo
        video_info = await video_service.analyze_video_file(str(video_path))
        
        # 🚀 Création du job de traitement
        job = VideoProcessingJob(
            job_id=job_id,
            status=ProcessingStatus.QUEUED,
            progress_percentage=0.0,
            video_info=video_info,
            current_frame=0,
            total_frames=video_info.get("total_frames", 0),
            detections_count=0,
            processing_time_elapsed=0.0
        )
        
        # 🗃️ Stockage du job
        active_jobs[job_id] = job
        
        # 📋 Paramètres de traitement
        processing_params = {
            "job_id": job_id,
            "video_path": str(video_path),
            "confidence_threshold": confidence_threshold,
            "processing_mode": processing_mode,
            "model_name": model_name,
            "generate_annotated_video": generate_annotated_video,
            "extract_keyframes": extract_keyframes,
            "notification_webhook": notification_webhook
        }
        
        # 🚀 Lancement du traitement en arrière-plan
        background_tasks.add_task(
            process_video_background,
            video_service=video_service,
            **processing_params
        )
        
        logger.info(f"✅ Job {job_id} créé et mis en queue")
        
        return {
            "success": True,
            "message": "Traitement vidéo démarré",
            "job_id": job_id,
            "status": "queued",
            "video_info": video_info,
            "estimated_processing_time_minutes": video_info.get("duration_seconds", 0) / 60 / processing_mode_speed_factor(processing_mode),
            "progress_url": f"/api/v1/detect/video/progress/{job_id}",
            "result_url": f"/api/v1/detect/video/result/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur création job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur traitement: {str(e)}")

# 📊 ENDPOINT - PROGRESSION D'UN JOB
@router.get(
    "/progress/{job_id}",
    response_model=VideoProcessingJob,
    summary="📊 Progression du traitement",
    description="""
    ## 📊 Suivi de la progression d'un job de traitement vidéo
    
    **Informations retournées:**
    - Statut actuel (QUEUED, PROCESSING, COMPLETED, FAILED)
    - Pourcentage de progression (0-100%)
    - Frame actuelle / Total frames
    - Nombre de détections trouvées
    - Temps écoulé et temps estimé restant
    - Messages d'erreur si applicable
    
    **Utilisé par:** Next.js pour barre de progression, Spring Boot pour monitoring
    """
)
async def get_job_progress(job_id: str):
    """📊 Récupère la progression d'un job de traitement"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = active_jobs[job_id]
    
    return job

# 📋 ENDPOINT - RÉSULTAT D'UN JOB
@router.get(
    "/result/{job_id}",
    summary="📋 Résultats du traitement",
    description="""
    ## 📋 Récupération des résultats complets d'un job terminé
    
    **Contenu des résultats:**
    - Timeline détaillée des détections avec timestamps
    - Statistiques globales (objets les plus détectés, etc.)
    - URLs de téléchargement (vidéo annotée, keyframes)
    - Métadonnées de traitement (temps, modèle utilisé)
    - Erreurs rencontrées si applicable
    
    **Statut requis:** COMPLETED
    """
)
async def get_job_result(job_id: str):
    """📋 Récupère les résultats complets d'un job terminé"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = active_jobs[job_id]
    
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job non terminé. Statut actuel: {job.status}"
        )
    
    try:
        # Chargement des résultats depuis le stockage
        results_path = Path(f"storage/temp/results/{job_id}_results.json")
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Résultats non trouvés")
        
        import json
        async with aiofiles.open(results_path, 'r') as f:
            results = json.loads(await f.read())
        
        return {
            "success": True,
            "job_id": job_id,
            "results": results,
            "job_info": job
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération résultats {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🎥 ENDPOINT - TÉLÉCHARGEMENT VIDÉO ANNOTÉE
@router.get(
    "/download/{job_id}/annotated",
    summary="🎥 Télécharger vidéo annotée",
    description="Télécharge la vidéo avec les détections visualisées"
)
async def download_annotated_video(job_id: str):
    """🎥 Télécharge la vidéo annotée d'un job terminé"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = active_jobs[job_id]
    
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Traitement non terminé")
    
    # Chemin de la vidéo annotée
    annotated_video_path = Path(f"storage/temp/results/{job_id}_annotated.mp4")
    
    if not annotated_video_path.exists():
        raise HTTPException(status_code=404, detail="Vidéo annotée non trouvée")
    
    return FileResponse(
        annotated_video_path,
        media_type="video/mp4",
        filename=f"detected_{job_id}.mp4"
    )

# 🖼️ ENDPOINT - TÉLÉCHARGEMENT KEYFRAMES
@router.get(
    "/download/{job_id}/keyframes",
    summary="🖼️ Télécharger keyframes",
    description="Télécharge un ZIP avec les frames clés contenant des détections"
)
async def download_keyframes(job_id: str):
    """🖼️ Télécharge les keyframes d'un job terminé"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    keyframes_zip_path = Path(f"storage/temp/results/{job_id}_keyframes.zip")
    
    if not keyframes_zip_path.exists():
        raise HTTPException(status_code=404, detail="Keyframes non trouvées")
    
    return FileResponse(
        keyframes_zip_path,
        media_type="application/zip",
        filename=f"keyframes_{job_id}.zip"
    )

# ❌ ENDPOINT - ANNULER UN JOB
@router.delete(
    "/cancel/{job_id}",
    summary="❌ Annuler traitement",
    description="Annule un job de traitement en cours"
)
async def cancel_job(job_id: str):
    """❌ Annule un job de traitement vidéo"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = active_jobs[job_id]
    
    if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Job déjà terminé")
    
    # Marquer comme annulé
    job.status = ProcessingStatus.CANCELLED
    
    logger.info(f"❌ Job {job_id} annulé")
    
    return {
        "success": True,
        "message": f"Job {job_id} annulé",
        "job_id": job_id
    }

# 📈 ENDPOINT - STATISTIQUES VIDÉOS
@router.get(
    "/statistics",
    summary="📈 Statistiques traitement vidéos",
    description="Statistiques globales des traitements vidéo effectués"
)
async def get_video_statistics(
    video_service: VideoDetectionService = Depends(get_video_service)
):
    """📈 Récupère les statistiques des traitements vidéo"""
    
    try:
        stats = await video_service.get_processing_statistics()
        
        # Ajouter les stats des jobs actifs
        active_stats = {
            "active_jobs": len(active_jobs),
            "jobs_by_status": {},
            "total_frames_processed": 0,
            "total_detections": 0
        }
        
        for job in active_jobs.values():
            status = job.status.value
            active_stats["jobs_by_status"][status] = active_stats["jobs_by_status"].get(status, 0) + 1
            active_stats["total_frames_processed"] += job.current_frame
            active_stats["total_detections"] += job.detections_count
        
        return {
            "success": True,
            "statistics": {
                **stats,
                "active_jobs": active_stats
            },
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques vidéo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🔧 FONCTIONS UTILITAIRES
def processing_mode_speed_factor(mode: ProcessingMode) -> float:
    """⚡ Facteur de vitesse selon le mode de traitement"""
    factors = {
        ProcessingMode.ULTRA_FAST: 10.0,
        ProcessingMode.FAST: 5.0,
        ProcessingMode.BALANCED: 3.0,
        ProcessingMode.QUALITY: 1.0
    }
    return factors.get(mode, 5.0)

async def process_video_background(
    video_service: VideoDetectionService,
    job_id: str,
    video_path: str,
    confidence_threshold: float,
    processing_mode: ProcessingMode,
    model_name: str,
    generate_annotated_video: bool,
    extract_keyframes: bool,
    notification_webhook: Optional[str]
):
    """🎬 Fonction de traitement vidéo en arrière-plan"""
    
    job = active_jobs[job_id]
    
    try:
        logger.info(f"🚀 Début traitement vidéo {job_id}")
        
        # Marquer comme en cours
        job.status = ProcessingStatus.PROCESSING
        
        # Traitement avec votre service vidéo
        async for progress_update in video_service.process_video_with_progress(
            video_path=video_path,
            job_id=job_id,
            confidence_threshold=confidence_threshold,
            processing_mode=processing_mode,
            model_name=model_name,
            generate_annotated_video=generate_annotated_video,
            extract_keyframes=extract_keyframes
        ):
            # Mise à jour du job
            job.current_frame = progress_update.get("current_frame", 0)
            job.progress_percentage = progress_update.get("progress_percentage", 0.0)
            job.detections_count = progress_update.get("detections_count", 0)
            job.processing_time_elapsed = progress_update.get("processing_time", 0.0)
            job.estimated_time_remaining = progress_update.get("estimated_remaining", None)
        
        # Traitement terminé
        job.status = ProcessingStatus.COMPLETED
        job.progress_percentage = 100.0
        
        logger.info(f"✅ Traitement terminé {job_id}")
        
        # Notification webhook vers Spring Boot
        if notification_webhook:
            await send_webhook_notification(notification_webhook, job_id, "completed")
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement {job_id}: {e}", exc_info=True)
        
        job.status = ProcessingStatus.FAILED
        job.error_message = str(e)
        
        # Notification d'erreur
        if notification_webhook:
            await send_webhook_notification(notification_webhook, job_id, "failed", str(e))

async def send_webhook_notification(webhook_url: str, job_id: str, status: str, error: str = None):
    """📡 Envoie une notification webhook vers Spring Boot"""
    
    try:
        import aiohttp
        
        payload = {
            "job_id": job_id,
            "status": status,
            "timestamp": time.time()
        }
        
        if error:
            payload["error"] = error
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"📡 Webhook envoyé pour {job_id}")
                else:
                    logger.warning(f"⚠️ Webhook failed for {job_id}: {response.status}")
                    
    except Exception as e:
        logger.error(f"❌ Erreur webhook {job_id}: {e}")

# 📝 INFORMATIONS D'EXPORT
__all__ = ["router"]