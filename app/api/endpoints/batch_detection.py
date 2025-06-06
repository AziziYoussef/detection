"""
📦 SERVICE DE TRAITEMENT BATCH
==============================
Endpoint spécialisé pour le traitement en lot de multiples fichiers

Fonctionnalités:
- Upload simultané de multiples images/vidéos
- Traitement parallèle optimisé GPU/CPU
- Queue de traitement avec priorités
- Résultats agrégés et comparaisons
- Export en différents formats (JSON, CSV, Excel)
- Statistiques globales et par lot
- Support de dossiers ZIP complets

Intégration:
- Spring Boot: Gestion des batches et suivi des jobs
- Next.js: Interface de progression et visualisation des résultats
- Optimisation pour traitements industriels ou recherche
"""

import os
import asyncio
import time
import uuid
import zipfile
import tempfile
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum
import aiofiles
import aiofiles.os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

# Services internes
from app.services.batch_service import BatchDetectionService
from app.schemas.detection import DetectionResult, BatchProcessingResult
from app.config.config import get_settings

logger = logging.getLogger(__name__)

# 🛣️ CRÉATION DU ROUTER
router = APIRouter()

# 📋 ÉNUMÉRATIONS ET SCHÉMAS
class BatchProcessingMode(str, Enum):
    """🎛️ Modes de traitement batch"""
    SEQUENTIAL = "sequential"     # Un par un (plus lent mais moins de mémoire)
    PARALLEL = "parallel"         # Parallèle optimisé (plus rapide)
    MIXED = "mixed"              # Images en parallèle, vidéos séquentielles
    QUEUE = "queue"              # Système de queue avec priorités

class BatchExportFormat(str, Enum):
    """📊 Formats d'export des résultats"""
    JSON = "json"                # JSON structuré
    CSV = "csv"                  # CSV pour Excel
    EXCEL = "excel"              # Fichier Excel avec onglets
    PDF = "pdf"                  # Rapport PDF
    HTML = "html"                # Rapport HTML interactif

class BatchJobStatus(str, Enum):
    """📊 Status des jobs batch"""
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    PROCESSING = "processing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchJob(BaseModel):
    """📋 Job de traitement batch"""
    job_id: str
    status: BatchJobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_files: int
    processed_files: int
    failed_files: int
    progress_percentage: float
    processing_mode: BatchProcessingMode
    current_file: Optional[str] = None
    estimated_completion: Optional[float] = None
    error_message: Optional[str] = None

class BatchDetectionRequest(BaseModel):
    """📋 Requête de traitement batch"""
    confidence_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.1, 
        le=0.9,
        description="Seuil de confiance global"
    )
    processing_mode: Optional[BatchProcessingMode] = Field(
        default=BatchProcessingMode.PARALLEL,
        description="Mode de traitement"
    )
    model_name: Optional[str] = Field(
        default="epoch_30",
        description="Modèle à utiliser"
    )
    export_format: Optional[BatchExportFormat] = Field(
        default=BatchExportFormat.JSON,
        description="Format d'export des résultats"
    )
    generate_summary_report: Optional[bool] = Field(
        default=True,
        description="Générer rapport de synthèse"
    )
    include_annotated_files: Optional[bool] = Field(
        default=False,
        description="Inclure fichiers annotés"
    )
    notification_webhook: Optional[str] = Field(
        default=None,
        description="URL webhook pour notifications"
    )

# 🗃️ STOCKAGE DES JOBS BATCH
active_batch_jobs: Dict[str, BatchJob] = {}
batch_processing_queue = asyncio.Queue()

# 🔧 DÉPENDANCES
async def get_batch_service() -> BatchDetectionService:
    """Récupère le service de traitement batch"""
    try:
        from main import get_model_service
        model_service = get_model_service()
        return BatchDetectionService(model_service)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service batch non disponible: {e}")

def validate_batch_files(files: List[UploadFile]) -> None:
    """🔍 Valide les fichiers du batch"""
    settings = get_settings()
    
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Trop de fichiers. Maximum: {settings.MAX_BATCH_SIZE}"
        )
    
    total_size = 0
    supported_formats = settings.SUPPORTED_IMAGE_FORMATS + settings.SUPPORTED_VIDEO_FORMATS
    
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Tous les fichiers doivent avoir un nom")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=422,
                detail=f"Format non supporté: {file_ext}. Formats acceptés: {supported_formats}"
            )
        
        # Estimation taille (approximative)
        if hasattr(file, 'size') and file.size:
            total_size += file.size
    
    max_total_size = settings.MAX_BATCH_SIZE_MB * 1024 * 1024
    if total_size > max_total_size:
        raise HTTPException(
            status_code=413,
            detail=f"Taille totale trop importante. Maximum: {settings.MAX_BATCH_SIZE_MB}MB"
        )

# 📦 ENDPOINT PRINCIPAL - TRAITEMENT BATCH
@router.post(
    "/process",
    summary="📦 Traitement batch de fichiers",
    description="""
    ## 📦 Traitement en lot de multiples fichiers (images + vidéos)
    
    **Fonctionnalités:**
    - Upload simultané jusqu'à 100 fichiers
    - Traitement parallèle optimisé GPU/CPU
    - Support images (JPEG, PNG, WEBP) + vidéos (MP4, AVI, MOV)
    - Résultats agrégés avec statistiques globales
    - Export multi-formats (JSON, CSV, Excel, PDF)
    
    **Modes de traitement:**
    - `SEQUENTIAL`: Un par un (économe en mémoire)
    - `PARALLEL`: Parallèle optimisé (plus rapide)
    - `MIXED`: Images || Vidéos séquentielles (équilibré)
    - `QUEUE`: Système de queue avec priorités
    
    **Résultats:**
    - Détections par fichier avec métadonnées
    - Statistiques globales et comparaisons
    - Rapport de synthèse avec visualisations
    - Export dans le format choisi
    
    **Utilisé pour:** Analyse industrielle, recherche, validation de datasets
    """
)
async def start_batch_processing(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Fichiers à traiter"),
    confidence_threshold: float = Form(default=0.5, ge=0.1, le=0.9),
    processing_mode: BatchProcessingMode = Form(default=BatchProcessingMode.PARALLEL),
    model_name: str = Form(default="epoch_30"),
    export_format: BatchExportFormat = Form(default=BatchExportFormat.JSON),
    generate_summary_report: bool = Form(default=True),
    include_annotated_files: bool = Form(default=False),
    notification_webhook: Optional[str] = Form(default=None),
    batch_service: BatchDetectionService = Depends(get_batch_service)
):
    """📦 Démarre le traitement batch de multiples fichiers"""
    
    job_id = f"batch_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"📦 Nouveau job batch: {job_id} - {len(files)} fichiers")
    
    try:
        # 🔍 Validation des fichiers
        validate_batch_files(files)
        
        # 📁 Création du dossier temporaire pour ce batch
        settings = get_settings()
        batch_dir = Path(settings.TEMP_UPLOAD_DIR) / job_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 💾 Sauvegarde des fichiers
        saved_files = []
        
        for i, file in enumerate(files):
            file_path = batch_dir / f"{i:04d}_{file.filename}"
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            saved_files.append({
                "index": i,
                "filename": file.filename,
                "path": str(file_path),
                "size": len(content),
                "type": "image" if Path(file.filename).suffix.lower() in settings.SUPPORTED_IMAGE_FORMATS else "video"
            })
        
        # 🚀 Création du job
        job = BatchJob(
            job_id=job_id,
            status=BatchJobStatus.QUEUED,
            created_at=time.time(),
            total_files=len(files),
            processed_files=0,
            failed_files=0,
            progress_percentage=0.0,
            processing_mode=processing_mode
        )
        
        # 🗃️ Stockage du job
        active_batch_jobs[job_id] = job
        
        # 📋 Paramètres de traitement
        processing_params = {
            "job_id": job_id,
            "files": saved_files,
            "batch_dir": str(batch_dir),
            "confidence_threshold": confidence_threshold,
            "processing_mode": processing_mode,
            "model_name": model_name,
            "export_format": export_format,
            "generate_summary_report": generate_summary_report,
            "include_annotated_files": include_annotated_files,
            "notification_webhook": notification_webhook
        }
        
        # 🚀 Lancement du traitement en arrière-plan
        background_tasks.add_task(
            process_batch_background,
            batch_service=batch_service,
            **processing_params
        )
        
        logger.info(f"✅ Job batch {job_id} créé et mis en queue")
        
        return {
            "success": True,
            "message": "Traitement batch démarré",
            "job_id": job_id,
            "status": "queued",
            "total_files": len(files),
            "processing_mode": processing_mode,
            "estimated_duration_minutes": estimate_batch_duration(saved_files, processing_mode),
            "progress_url": f"/api/v1/detect/batch/progress/{job_id}",
            "result_url": f"/api/v1/detect/batch/result/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur création job batch {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur traitement batch: {str(e)}")

# 📦 ENDPOINT - TRAITEMENT ZIP
@router.post(
    "/process-zip",
    summary="📦 Traitement batch depuis ZIP",
    description="""
    ## 📦 Traitement batch depuis archive ZIP
    
    **Avantages:**
    - Upload plus rapide d'un seul fichier ZIP
    - Préservation de l'arborescence des dossiers
    - Support de grandes collections de fichiers
    - Extraction automatique et traitement
    
    **Formats supportés dans le ZIP:**
    - Images: JPEG, PNG, WEBP, BMP, TIFF
    - Vidéos: MP4, AVI, MOV, MKV, WEBM
    
    **Limite:** 500MB par archive ZIP
    """
)
async def process_batch_from_zip(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(..., description="Archive ZIP à traiter"),
    confidence_threshold: float = Form(default=0.5, ge=0.1, le=0.9),
    processing_mode: BatchProcessingMode = Form(default=BatchProcessingMode.PARALLEL),
    model_name: str = Form(default="epoch_30"),
    export_format: BatchExportFormat = Form(default=BatchExportFormat.JSON),
    preserve_folder_structure: bool = Form(default=True),
    batch_service: BatchDetectionService = Depends(get_batch_service)
):
    """📦 Traite un batch de fichiers depuis une archive ZIP"""
    
    job_id = f"zipbatch_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"📦 Nouveau job ZIP batch: {job_id} - {zip_file.filename}")
    
    try:
        # Validation ZIP
        if not zip_file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=422, detail="Fichier ZIP requis")
        
        # Sauvegarde temporaire du ZIP
        settings = get_settings()
        temp_dir = Path(settings.TEMP_UPLOAD_DIR)
        temp_dir.mkdir(exist_ok=True)
        
        zip_path = temp_dir / f"{job_id}.zip"
        
        async with aiofiles.open(zip_path, 'wb') as f:
            content = await zip_file.read()
            await f.write(content)
        
        # Extraction et validation
        extracted_files = await extract_and_validate_zip(zip_path, job_id)
        
        if not extracted_files:
            raise HTTPException(status_code=422, detail="Aucun fichier valide trouvé dans le ZIP")
        
        # Création du job
        job = BatchJob(
            job_id=job_id,
            status=BatchJobStatus.PREPROCESSING,
            created_at=time.time(),
            total_files=len(extracted_files),
            processed_files=0,
            failed_files=0,
            progress_percentage=0.0,
            processing_mode=processing_mode
        )
        
        active_batch_jobs[job_id] = job
        
        # Paramètres de traitement
        processing_params = {
            "job_id": job_id,
            "files": extracted_files,
            "batch_dir": str(temp_dir / job_id),
            "confidence_threshold": confidence_threshold,
            "processing_mode": processing_mode,
            "model_name": model_name,
            "export_format": export_format,
            "preserve_folder_structure": preserve_folder_structure
        }
        
        # Lancement du traitement
        background_tasks.add_task(
            process_batch_background,
            batch_service=batch_service,
            **processing_params
        )
        
        return {
            "success": True,
            "message": "Traitement ZIP batch démarré",
            "job_id": job_id,
            "total_files": len(extracted_files),
            "zip_filename": zip_file.filename,
            "progress_url": f"/api/v1/detect/batch/progress/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur ZIP batch {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 📊 ENDPOINT - PROGRESSION BATCH
@router.get(
    "/progress/{job_id}",
    response_model=BatchJob,
    summary="📊 Progression traitement batch",
    description="""
    ## 📊 Suivi de progression d'un job de traitement batch
    
    **Informations temps réel:**
    - Statut global du job (QUEUED → PROCESSING → COMPLETED)
    - Progression en pourcentage (0-100%)
    - Fichiers traités / Total fichiers
    - Fichier en cours de traitement
    - Temps estimé de fin
    - Erreurs rencontrées
    
    **Utilisé par:** Next.js pour barres de progression, Spring Boot pour monitoring
    """
)
async def get_batch_progress(job_id: str):
    """📊 Récupère la progression d'un job batch"""
    
    if job_id not in active_batch_jobs:
        raise HTTPException(status_code=404, detail="Job batch non trouvé")
    
    job = active_batch_jobs[job_id]
    
    # Calcul du temps estimé restant
    if job.started_at and job.processed_files > 0:
        elapsed_time = time.time() - job.started_at
        avg_time_per_file = elapsed_time / job.processed_files
        remaining_files = job.total_files - job.processed_files
        job.estimated_completion = time.time() + (avg_time_per_file * remaining_files)
    
    return job

# 📋 ENDPOINT - RÉSULTATS BATCH
@router.get(
    "/result/{job_id}",
    summary="📋 Résultats traitement batch",
    description="""
    ## 📋 Récupération des résultats complets d'un batch terminé
    
    **Contenu des résultats:**
    - Détections par fichier avec métadonnées complètes
    - Statistiques globales et comparaisons
    - Rapport de synthèse avec graphiques
    - Objets les plus/moins détectés
    - Temps de traitement par fichier
    - Erreurs et avertissements
    
    **Formats de sortie:** JSON, CSV, Excel, PDF, HTML
    """
)
async def get_batch_result(job_id: str):
    """📋 Récupère les résultats complets d'un batch terminé"""
    
    if job_id not in active_batch_jobs:
        raise HTTPException(status_code=404, detail="Job batch non trouvé")
    
    job = active_batch_jobs[job_id]
    
    if job.status != BatchJobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job non terminé. Statut: {job.status}"
        )
    
    try:
        # Chargement des résultats
        results_path = Path(f"storage/temp/results/{job_id}_results.json")
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Résultats non trouvés")
        
        import json
        async with aiofiles.open(results_path, 'r') as f:
            results = json.loads(await f.read())
        
        return {
            "success": True,
            "job_id": job_id,
            "job_info": job,
            "results": results,
            "summary": results.get("summary", {}),
            "download_links": {
                "json": f"/api/v1/detect/batch/download/{job_id}/json",
                "csv": f"/api/v1/detect/batch/download/{job_id}/csv",
                "excel": f"/api/v1/detect/batch/download/{job_id}/excel",
                "report": f"/api/v1/detect/batch/download/{job_id}/report"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération résultats batch {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 📥 ENDPOINT - TÉLÉCHARGEMENT RÉSULTATS
@router.get(
    "/download/{job_id}/{format}",
    summary="📥 Télécharger résultats",
    description="Télécharge les résultats dans le format spécifié"
)
async def download_batch_results(job_id: str, format: BatchExportFormat):
    """📥 Télécharge les résultats d'un batch dans le format spécifié"""
    
    if job_id not in active_batch_jobs:
        raise HTTPException(status_code=404, detail="Job batch non trouvé")
    
    job = active_batch_jobs[job_id]
    
    if job.status != BatchJobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job non terminé")
    
    try:
        # Chemins des fichiers selon le format
        results_dir = Path(f"storage/temp/results")
        
        format_files = {
            BatchExportFormat.JSON: f"{job_id}_results.json",
            BatchExportFormat.CSV: f"{job_id}_results.csv", 
            BatchExportFormat.EXCEL: f"{job_id}_results.xlsx",
            BatchExportFormat.PDF: f"{job_id}_report.pdf",
            BatchExportFormat.HTML: f"{job_id}_report.html"
        }
        
        filename = format_files.get(format)
        if not filename:
            raise HTTPException(status_code=400, detail="Format non supporté")
        
        file_path = results_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Fichier {format} non trouvé")
        
        # Types MIME
        mime_types = {
            BatchExportFormat.JSON: "application/json",
            BatchExportFormat.CSV: "text/csv",
            BatchExportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            BatchExportFormat.PDF: "application/pdf",
            BatchExportFormat.HTML: "text/html"
        }
        
        return FileResponse(
            file_path,
            media_type=mime_types.get(format, "application/octet-stream"),
            filename=f"batch_results_{job_id}.{format.value}"
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement {job_id}/{format}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ❌ ENDPOINT - ANNULER BATCH
@router.delete(
    "/cancel/{job_id}",
    summary="❌ Annuler traitement batch",
    description="Annule un job de traitement batch en cours"
)
async def cancel_batch_job(job_id: str):
    """❌ Annule un job de traitement batch"""
    
    if job_id not in active_batch_jobs:
        raise HTTPException(status_code=404, detail="Job batch non trouvé")
    
    job = active_batch_jobs[job_id]
    
    if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Job déjà terminé")
    
    # Marquer comme annulé
    job.status = BatchJobStatus.CANCELLED
    
    logger.info(f"❌ Job batch {job_id} annulé")
    
    return {
        "success": True,
        "message": f"Job batch {job_id} annulé",
        "job_id": job_id
    }

# 📈 ENDPOINT - STATISTIQUES BATCH
@router.get(
    "/statistics",
    summary="📈 Statistiques traitement batch",
    description="Statistiques globales des traitements batch effectués"
)
async def get_batch_statistics(
    batch_service: BatchDetectionService = Depends(get_batch_service)
):
    """📈 Récupère les statistiques des traitements batch"""
    
    try:
        stats = await batch_service.get_batch_statistics()
        
        # Ajouter stats des jobs actifs
        active_stats = {
            "active_jobs": len(active_batch_jobs),
            "jobs_by_status": {},
            "total_files_processing": 0,
            "total_files_completed": 0
        }
        
        for job in active_batch_jobs.values():
            status = job.status.value
            active_stats["jobs_by_status"][status] = active_stats["jobs_by_status"].get(status, 0) + 1
            active_stats["total_files_processing"] += job.total_files
            active_stats["total_files_completed"] += job.processed_files
        
        return {
            "success": True,
            "statistics": {
                **stats,
                "active_jobs": active_stats
            },
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🔧 FONCTIONS UTILITAIRES
def estimate_batch_duration(files: List[Dict], mode: BatchProcessingMode) -> float:
    """⏱️ Estime la durée de traitement d'un batch"""
    
    # Estimations approximatives (en secondes par fichier)
    base_times = {
        "image": 0.5,  # 0.5s par image
        "video": 30.0  # 30s par vidéo (moyenne)
    }
    
    # Facteurs selon le mode
    mode_factors = {
        BatchProcessingMode.SEQUENTIAL: 1.0,
        BatchProcessingMode.PARALLEL: 0.3,
        BatchProcessingMode.MIXED: 0.5,
        BatchProcessingMode.QUEUE: 0.4
    }
    
    total_time = 0
    for file in files:
        file_type = file.get("type", "image")
        base_time = base_times.get(file_type, 1.0)
        total_time += base_time
    
    factor = mode_factors.get(mode, 1.0)
    estimated_minutes = (total_time * factor) / 60
    
    return max(1.0, estimated_minutes)  # Minimum 1 minute

async def extract_and_validate_zip(zip_path: Path, job_id: str) -> List[Dict]:
    """📦 Extrait et valide les fichiers d'un ZIP"""
    
    settings = get_settings()
    extract_dir = Path(settings.TEMP_UPLOAD_DIR) / job_id
    extract_dir.mkdir(exist_ok=True)
    
    valid_files = []
    supported_formats = settings.SUPPORTED_IMAGE_FORMATS + settings.SUPPORTED_VIDEO_FORMATS
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Lister les fichiers
            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue
                
                file_ext = Path(file_info.filename).suffix.lower()
                if file_ext in supported_formats:
                    # Extraire le fichier
                    zip_ref.extract(file_info, extract_dir)
                    
                    extracted_path = extract_dir / file_info.filename
                    
                    valid_files.append({
                        "index": len(valid_files),
                        "filename": file_info.filename,
                        "path": str(extracted_path),
                        "size": file_info.file_size,
                        "type": "image" if file_ext in settings.SUPPORTED_IMAGE_FORMATS else "video",
                        "folder": str(Path(file_info.filename).parent) if Path(file_info.filename).parent.name else ""
                    })
        
        return valid_files
        
    except Exception as e:
        logger.error(f"❌ Erreur extraction ZIP {job_id}: {e}")
        raise HTTPException(status_code=422, detail=f"Erreur extraction ZIP: {e}")

async def process_batch_background(
    batch_service: BatchDetectionService,
    job_id: str,
    files: List[Dict],
    batch_dir: str,
    confidence_threshold: float,
    processing_mode: BatchProcessingMode,
    model_name: str,
    export_format: BatchExportFormat,
    **kwargs
):
    """📦 Fonction de traitement batch en arrière-plan"""
    
    job = active_batch_jobs[job_id]
    
    try:
        logger.info(f"🚀 Début traitement batch {job_id} - {len(files)} fichiers")
        
        # Marquer comme en cours
        job.status = BatchJobStatus.PROCESSING
        job.started_at = time.time()
        
        # Traitement avec le service batch
        async for progress_update in batch_service.process_batch_with_progress(
            job_id=job_id,
            files=files,
            confidence_threshold=confidence_threshold,
            processing_mode=processing_mode,
            model_name=model_name,
            export_format=export_format,
            **kwargs
        ):
            # Mise à jour du job
            job.processed_files = progress_update.get("processed_files", 0)
            job.failed_files = progress_update.get("failed_files", 0)
            job.progress_percentage = progress_update.get("progress_percentage", 0.0)
            job.current_file = progress_update.get("current_file")
        
        # Post-traitement
        job.status = BatchJobStatus.POSTPROCESSING
        await batch_service.generate_batch_report(job_id, export_format)
        
        # Terminé
        job.status = BatchJobStatus.COMPLETED
        job.completed_at = time.time()
        job.progress_percentage = 100.0
        
        logger.info(f"✅ Traitement batch terminé {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement batch {job_id}: {e}", exc_info=True)
        
        job.status = BatchJobStatus.FAILED
        job.error_message = str(e)

# 📝 INFORMATIONS D'EXPORT
__all__ = ["router"]