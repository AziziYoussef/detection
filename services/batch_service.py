"""
📦 BATCH SERVICE - SERVICE DE TRAITEMENT BATCH MASSIF
====================================================
Service spécialisé pour le traitement batch de multiples fichiers simultanément

Fonctionnalités:
- Traitement parallèle de centaines de fichiers
- Support mixte images + vidéos dans un même batch
- Gestion intelligente des ressources (CPU/GPU/RAM)
- Priorisation et scheduling des tâches
- Progression en temps réel et reporting
- Reprise automatique en cas d'échec
- Optimisation pour datasets volumineux
- Export de rapports détaillés et analytics

Optimisations batch:
- Pool de workers dynamique
- Load balancing intelligent
- Gestion mémoire adaptative
- Cache distribué pour éviter retraitement
- Chunking automatique des gros datasets
- Parallélisation multi-niveaux (fichiers + frames)
"""

import asyncio
import time
import logging
import json
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import concurrent.futures
import psutil
import math

import numpy as np
from PIL import Image

# Imports internes
from app.schemas.detection import DetectionResult, BatchProcessingResult, BatchJob, BatchStatus
from app.config.config import get_settings
from .model_service import ModelService, PerformanceProfile
from .image_service import ImageService, ImageProcessingConfig
from .video_service import VideoService, VideoProcessingConfig

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class BatchPriority(str, Enum):
    """🎯 Priorités des tâches batch"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class BatchStrategy(str, Enum):
    """📊 Stratégies de traitement batch"""
    FIFO = "fifo"                    # Premier arrivé, premier servi
    PRIORITY = "priority"            # Par priorité
    SIZE_OPTIMIZED = "size_optimized" # Optimisé par taille fichier
    RESOURCE_BALANCED = "resource_balanced" # Équilibrage ressources
    MIXED_OPTIMAL = "mixed_optimal"  # Mélange optimal

class WorkerType(str, Enum):
    """👷 Types de workers"""
    IMAGE_WORKER = "image_worker"
    VIDEO_WORKER = "video_worker"
    MIXED_WORKER = "mixed_worker"

class BatchState(str, Enum):
    """📊 États des batches"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchProcessingConfig:
    """⚙️ Configuration du traitement batch"""
    
    # 🔧 Paramètres de base
    max_concurrent_jobs: int = 10
    max_workers_per_job: int = 4
    max_total_workers: int = 20
    
    # 📊 Stratégie et priorité
    strategy: BatchStrategy = BatchStrategy.RESOURCE_BALANCED
    default_priority: BatchPriority = BatchPriority.NORMAL
    
    # 💾 Gestion mémoire
    max_memory_usage_gb: float = 8.0
    memory_check_interval: float = 30.0  # secondes
    auto_memory_management: bool = True
    
    # 🔄 Retry et récupération
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    auto_resume_failed: bool = True
    
    # 📁 Gestion fichiers
    supported_image_formats: List[str] = field(default_factory=lambda: 
        ["jpg", "jpeg", "png", "bmp", "tiff", "webp"])
    supported_video_formats: List[str] = field(default_factory=lambda: 
        ["mp4", "avi", "mov", "mkv", "webm"])
    max_file_size_mb: float = 500.0
    
    # 💾 Sauvegarde et rapports
    save_individual_results: bool = True
    generate_batch_report: bool = True
    save_progress_checkpoints: bool = True
    checkpoint_interval: int = 50  # Tous les 50 fichiers
    
    # 🎯 Détection
    confidence_threshold: float = 0.5
    batch_model_optimization: bool = True  # Optimise modèle pour batch
    
    # ⚡ Optimisations
    enable_caching: bool = True
    cache_processed_files: bool = True
    skip_duplicates: bool = True
    parallel_file_reading: bool = True

@dataclass
class BatchWorker:
    """👷 Worker de traitement batch"""
    worker_id: str
    worker_type: WorkerType
    is_busy: bool = False
    current_task: Optional[str] = None
    processed_count: int = 0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def get_average_time(self) -> float:
        """⏱️ Temps de traitement moyen"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_efficiency(self) -> float:
        """📊 Efficacité du worker"""
        total_tasks = self.processed_count + self.error_count
        return self.processed_count / total_tasks if total_tasks > 0 else 1.0

@dataclass
class BatchJob:
    """📦 Job de traitement batch"""
    job_id: str
    files: List[Path]
    config: BatchProcessingConfig
    priority: BatchPriority = BatchPriority.NORMAL
    state: BatchState = BatchState.PENDING
    
    # Progression
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    
    # Timing
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    
    # Résultats
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.total_files == 0:
            self.total_files = len(self.files)
    
    def get_progress_percentage(self) -> float:
        """📊 Pourcentage de progression"""
        return (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0.0
    
    def get_estimated_time_remaining(self) -> float:
        """⏱️ Temps estimé restant en secondes"""
        if not self.started_time or self.processed_files == 0:
            return 0.0
        
        elapsed = time.time() - self.started_time
        rate = self.processed_files / elapsed
        remaining_files = self.total_files - self.processed_files
        
        return remaining_files / rate if rate > 0 else 0.0

# 📦 SERVICE PRINCIPAL
class BatchService:
    """📦 Service de traitement batch"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.settings = get_settings()
        
        # Configuration par défaut
        self.config = BatchProcessingConfig()
        
        # Services spécialisés
        self.image_service = ImageService(model_service)
        self.video_service = VideoService(model_service)
        
        # Gestion des jobs
        self._jobs: Dict[str, BatchJob] = {}
        self._job_queue: deque = deque()
        self._active_jobs: Dict[str, BatchJob] = {}
        
        # Pool de workers
        self._workers: Dict[str, BatchWorker] = {}
        self._executor = None
        
        # Gestion ressources
        self._resource_monitor = None
        self._memory_usage_gb = 0.0
        
        # Cache des fichiers traités
        self._file_cache: Dict[str, Any] = {}
        
        # Statistiques
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_files_processed": 0,
            "total_processing_time": 0.0,
            "start_time": time.time(),
            "peak_memory_gb": 0.0,
            "peak_workers": 0
        }
        
        # Chemins de stockage
        self.batch_results_path = Path(self.settings.RESULTS_PATH) / "batch"
        self.checkpoints_path = Path(self.settings.CACHE_PATH) / "batch_checkpoints"
        
        for path in [self.batch_results_path, self.checkpoints_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info("📦 BatchService initialisé")
    
    async def initialize(self):
        """🚀 Initialise le service batch"""
        logger.info("🚀 Initialisation BatchService...")
        
        # Initialiser les services
        await self.image_service.initialize()
        await self.video_service.initialize()
        
        # Créer l'executor principal
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_total_workers,
            thread_name_prefix="batch_worker"
        )
        
        # Créer les workers
        await self._initialize_workers()
        
        # Démarrer le monitoring des ressources
        if self.config.auto_memory_management:
            await self._start_resource_monitoring()
        
        # Reprendre les jobs interrompus si configuré
        if self.config.auto_resume_failed:
            await self._resume_interrupted_jobs()
        
        logger.info("✅ BatchService initialisé")
    
    async def submit_batch_job(
        self,
        files: List[Union[str, Path]],
        config: Optional[BatchProcessingConfig] = None,
        priority: BatchPriority = BatchPriority.NORMAL,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        📦 Soumet un nouveau job batch
        
        Args:
            files: Liste des fichiers à traiter
            config: Configuration personnalisée
            priority: Priorité du job
            job_id: ID personnalisé (généré si None)
            metadata: Métadonnées additionnelles
            
        Returns:
            ID du job créé
        """
        
        job_id = job_id or str(uuid.uuid4())
        config = config or self.config
        metadata = metadata or {}
        
        # Validation et préparation des fichiers
        validated_files = await self._validate_and_prepare_files(files)
        
        if not validated_files:
            raise ValueError("Aucun fichier valide trouvé")
        
        # Création du job
        job = BatchJob(
            job_id=job_id,
            files=validated_files,
            config=config,
            priority=priority,
            metadata=metadata
        )
        
        # Ajout à la queue
        self._jobs[job_id] = job
        self._add_job_to_queue(job)
        
        # Statistiques
        self.stats["total_jobs"] += 1
        
        logger.info(f"📦 Job batch soumis: {job_id} - {len(validated_files)} fichiers")
        
        # Démarrer le traitement si possible
        await self._process_job_queue()
        
        return job_id
    
    async def _validate_and_prepare_files(
        self, 
        files: List[Union[str, Path]]
    ) -> List[Path]:
        """✅ Valide et prépare la liste des fichiers"""
        
        validated_files = []
        supported_extensions = (
            self.config.supported_image_formats + 
            self.config.supported_video_formats
        )
        
        for file_path in files:
            try:
                path = Path(file_path)
                
                # Vérifications de base
                if not path.exists():
                    logger.warning(f"⚠️ Fichier non trouvé: {path}")
                    continue
                
                if not path.is_file():
                    continue
                
                # Extension supportée
                extension = path.suffix.lower().lstrip('.')
                if extension not in supported_extensions:
                    logger.warning(f"⚠️ Extension non supportée: {extension}")
                    continue
                
                # Taille du fichier
                file_size_mb = path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.max_file_size_mb:
                    logger.warning(f"⚠️ Fichier trop volumineux: {path} ({file_size_mb:.1f}MB)")
                    continue
                
                # Skip duplicatas si configuré
                if self.config.skip_duplicates:
                    file_hash = await self._calculate_file_hash(path)
                    if file_hash in self._file_cache:
                        logger.debug(f"⏭️ Fichier déjà traité: {path}")
                        continue
                
                validated_files.append(path)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur validation fichier {file_path}: {e}")
                continue
        
        return validated_files
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """🔐 Calcule le hash d'un fichier"""
        
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                # Lire par chunks pour gros fichiers
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul hash {file_path}: {e}")
            return str(file_path)  # Fallback
    
    def _add_job_to_queue(self, job: BatchJob):
        """📋 Ajoute un job à la queue selon la stratégie"""
        
        if self.config.strategy == BatchStrategy.FIFO:
            self._job_queue.append(job)
            
        elif self.config.strategy == BatchStrategy.PRIORITY:
            # Insertion par priorité
            priority_order = [BatchPriority.URGENT, BatchPriority.HIGH, BatchPriority.NORMAL, BatchPriority.LOW]
            job_priority_index = priority_order.index(job.priority)
            
            inserted = False
            for i, queued_job in enumerate(self._job_queue):
                queued_priority_index = priority_order.index(queued_job.priority)
                if job_priority_index < queued_priority_index:
                    self._job_queue.insert(i, job)
                    inserted = True
                    break
            
            if not inserted:
                self._job_queue.append(job)
                
        else:
            # Autres stratégies - pour l'instant FIFO
            self._job_queue.append(job)
    
    async def _process_job_queue(self):
        """🔄 Traite la queue des jobs"""
        
        while (self._job_queue and 
               len(self._active_jobs) < self.config.max_concurrent_jobs):
            
            # Vérifier les ressources disponibles
            if not await self._check_resources_available():
                break
            
            # Prendre le prochain job
            job = self._job_queue.popleft()
            
            # Démarrer le job
            await self._start_job(job)
    
    async def _check_resources_available(self) -> bool:
        """🔍 Vérifie si les ressources sont disponibles"""
        
        # Vérifier mémoire
        if self.config.auto_memory_management:
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 85:  # Plus de 85% utilisé
                logger.warning(f"⚠️ Mémoire élevée: {memory_usage}%")
                return False
        
        # Vérifier workers disponibles
        available_workers = len([w for w in self._workers.values() if not w.is_busy])
        if available_workers < 2:  # Garder au moins 2 workers
            return False
        
        return True
    
    async def _start_job(self, job: BatchJob):
        """🚀 Démarre l'exécution d'un job"""
        
        job.state = BatchState.RUNNING
        job.started_time = time.time()
        self._active_jobs[job.job_id] = job
        
        logger.info(f"🚀 Démarrage job batch: {job.job_id}")
        
        # Traitement asynchrone
        asyncio.create_task(self._execute_job(job))
    
    async def _execute_job(self, job: BatchJob):
        """⚙️ Exécute un job batch"""
        
        try:
            # Chunking des fichiers pour traitement parallèle optimal
            file_chunks = self._create_file_chunks(job.files, job.config)
            
            # Traitement par chunks
            for chunk_idx, chunk in enumerate(file_chunks):
                if job.state == BatchState.CANCELLED:
                    break
                
                # Traitement du chunk
                chunk_results = await self._process_file_chunk(chunk, job, chunk_idx)
                job.results.extend(chunk_results)
                
                # Mise à jour progression
                job.processed_files += len(chunk)
                job.successful_files += len([r for r in chunk_results if not r.get('error')])
                job.failed_files += len([r for r in chunk_results if r.get('error')])
                
                # Checkpoint si configuré
                if (job.config.save_progress_checkpoints and 
                    job.processed_files % job.config.checkpoint_interval == 0):
                    await self._save_checkpoint(job)
                
                # Gestion mémoire
                if self.config.auto_memory_management:
                    await self._manage_memory()
            
            # Finaliser le job
            await self._finalize_job(job)
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution job {job.job_id}: {e}")
            job.state = BatchState.FAILED
            job.errors.append({
                "type": "job_execution_error",
                "error": str(e),
                "timestamp": time.time()
            })
        finally:
            # Nettoyer
            if job.job_id in self._active_jobs:
                del self._active_jobs[job.job_id]
            
            # Traiter le prochain job
            await self._process_job_queue()
    
    def _create_file_chunks(
        self, 
        files: List[Path], 
        config: BatchProcessingConfig
    ) -> List[List[Path]]:
        """📦 Crée des chunks de fichiers pour traitement optimal"""
        
        # Taille de chunk basée sur nombre de workers et type de fichiers
        available_workers = len([w for w in self._workers.values() if not w.is_busy])
        chunk_size = max(1, min(config.max_workers_per_job, available_workers))
        
        # Séparer images et vidéos pour optimisation
        images = [f for f in files if f.suffix.lower().lstrip('.') in config.supported_image_formats]
        videos = [f for f in files if f.suffix.lower().lstrip('.') in config.supported_video_formats]
        
        chunks = []
        
        # Chunks d'images (plus petits, traitement plus rapide)
        for i in range(0, len(images), chunk_size * 2):
            chunks.append(images[i:i + chunk_size * 2])
        
        # Chunks de vidéos (plus gros, traitement plus lent)
        for i in range(0, len(videos), chunk_size):
            chunks.append(videos[i:i + chunk_size])
        
        return [chunk for chunk in chunks if chunk]
    
    async def _process_file_chunk(
        self, 
        chunk: List[Path], 
        job: BatchJob, 
        chunk_idx: int
    ) -> List[Dict[str, Any]]:
        """🔄 Traite un chunk de fichiers"""
        
        logger.debug(f"🔄 Traitement chunk {chunk_idx}: {len(chunk)} fichiers")
        
        # Traitement en parallèle des fichiers du chunk
        tasks = []
        semaphore = asyncio.Semaphore(job.config.max_workers_per_job)
        
        for file_path in chunk:
            task = self._process_single_file_with_semaphore(
                file_path, job, semaphore
            )
            tasks.append(task)
        
        # Attendre tous les traitements
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traiter les résultats et exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = {
                    "file_path": str(chunk[i]),
                    "error": str(result),
                    "timestamp": time.time()
                }
                processed_results.append(error_result)
                job.errors.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_file_with_semaphore(
        self, 
        file_path: Path, 
        job: BatchJob, 
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """📁 Traite un fichier individuel avec semaphore"""
        
        async with semaphore:
            return await self._process_single_file(file_path, job)
    
    async def _process_single_file(
        self, 
        file_path: Path, 
        job: BatchJob
    ) -> Dict[str, Any]:
        """📁 Traite un fichier individuel"""
        
        start_time = time.time()
        
        try:
            # Vérifier cache si activé
            if job.config.enable_caching:
                file_hash = await self._calculate_file_hash(file_path)
                if file_hash in self._file_cache:
                    cached_result = self._file_cache[file_hash]
                    cached_result["from_cache"] = True
                    return cached_result
            
            # Déterminer le type de fichier
            extension = file_path.suffix.lower().lstrip('.')
            
            if extension in job.config.supported_image_formats:
                # Traitement image
                result = await self._process_image_file(file_path, job)
            elif extension in job.config.supported_video_formats:
                # Traitement vidéo
                result = await self._process_video_file(file_path, job)
            else:
                raise ValueError(f"Format non supporté: {extension}")
            
            # Enrichir le résultat
            processing_time = time.time() - start_time
            result.update({
                "file_path": str(file_path),
                "processing_time_seconds": processing_time,
                "timestamp": time.time(),
                "job_id": job.job_id
            })
            
            # Mettre en cache si configuré
            if job.config.cache_processed_files and file_hash:
                self._file_cache[file_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement fichier {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
                "timestamp": time.time(),
                "job_id": job.job_id
            }
    
    async def _process_image_file(self, file_path: Path, job: BatchJob) -> Dict[str, Any]:
        """📸 Traite un fichier image"""
        
        # Configuration optimisée pour batch
        image_config = ImageProcessingConfig(
            confidence_threshold=job.config.confidence_threshold,
            quality="balanced",
            save_results=job.config.save_individual_results,
            save_annotated_images=False,  # Pas d'images annotées en batch
            cache_results=False,  # Cache géré au niveau batch
            auto_model_selection=job.config.batch_model_optimization
        )
        
        # Traitement
        result = await self.image_service.process_image(
            image_input=file_path,
            config=image_config,
            processing_id=f"{job.job_id}_{file_path.stem}"
        )
        
        # Conversion pour batch
        return {
            "type": "image",
            "detections": [d.__dict__ for d in result.detections],
            "model_used": result.model_used,
            "processing_config": result.processing_config,
            "metadata": result.metadata
        }
    
    async def _process_video_file(self, file_path: Path, job: BatchJob) -> Dict[str, Any]:
        """🎬 Traite un fichier vidéo"""
        
        # Configuration optimisée pour batch
        video_config = VideoProcessingConfig(
            processing_strategy="sample_adaptive",  # Échantillonnage pour batch
            max_frames_to_process=100,  # Limiter pour batch
            confidence_threshold=job.config.confidence_threshold,
            save_annotated_video=False,  # Pas de vidéo annotée en batch
            save_timeline_json=job.config.save_individual_results,
            parallel_processing=True
        )
        
        # Traitement
        result = await self.video_service.process_video(
            video_input=file_path,
            config=video_config,
            processing_id=f"{job.job_id}_{file_path.stem}"
        )
        
        # Conversion pour batch
        return {
            "type": "video",
            "timeline": {
                "frames": len(result.timeline.frames),
                "unique_objects": len(result.timeline.unique_objects),
                "duration_seconds": result.timeline.duration_seconds
            },
            "summary": result.summary,
            "processing_config": result.processing_config,
            "metadata": result.metadata
        }
    
    async def _finalize_job(self, job: BatchJob):
        """✅ Finalise un job terminé"""
        
        job.completed_time = time.time()
        job.state = BatchState.COMPLETED if job.failed_files == 0 else BatchState.FAILED
        
        # Générer le rapport si configuré
        if job.config.generate_batch_report:
            await self._generate_batch_report(job)
        
        # Statistiques
        self.stats["completed_jobs"] += 1 if job.state == BatchState.COMPLETED else 0
        self.stats["failed_jobs"] += 1 if job.state == BatchState.FAILED else 0
        self.stats["total_files_processed"] += job.processed_files
        
        if job.started_time:
            processing_time = job.completed_time - job.started_time
            self.stats["total_processing_time"] += processing_time
        
        logger.info(
            f"✅ Job terminé: {job.job_id} - "
            f"{job.successful_files}/{job.total_files} succès "
            f"en {processing_time:.1f}s"
        )
    
    async def _generate_batch_report(self, job: BatchJob):
        """📊 Génère un rapport détaillé du batch"""
        
        try:
            # Calculs statistiques
            processing_time = job.completed_time - job.started_time if job.started_time else 0
            success_rate = job.successful_files / job.total_files if job.total_files > 0 else 0
            
            # Analyse par type de fichier
            file_types = defaultdict(int)
            detection_stats = defaultdict(list)
            
            for result in job.results:
                if not result.get('error'):
                    file_type = result.get('type', 'unknown')
                    file_types[file_type] += 1
                    
                    # Statistiques de détection
                    detections = result.get('detections', [])
                    detection_stats['total_detections'].append(len(detections))
                    
                    for detection in detections:
                        class_name = detection.get('class_name_fr', 'unknown')
                        detection_stats['classes'].append(class_name)
            
            # Rapport complet
            report = {
                "job_info": {
                    "job_id": job.job_id,
                    "created_time": job.created_time,
                    "started_time": job.started_time,
                    "completed_time": job.completed_time,
                    "processing_time_seconds": processing_time,
                    "priority": job.priority.value,
                    "state": job.state.value
                },
                "file_statistics": {
                    "total_files": job.total_files,
                    "processed_files": job.processed_files,
                    "successful_files": job.successful_files,
                    "failed_files": job.failed_files,
                    "success_rate": success_rate,
                    "file_types": dict(file_types)
                },
                "detection_statistics": {
                    "total_detections": sum(detection_stats['total_detections']),
                    "avg_detections_per_file": np.mean(detection_stats['total_detections']) if detection_stats['total_detections'] else 0,
                    "classes_found": list(set(detection_stats['classes'])),
                    "class_distribution": dict(defaultdict(int, 
                        [(cls, detection_stats['classes'].count(cls)) for cls in set(detection_stats['classes'])]))
                },
                "performance": {
                    "avg_processing_time_per_file": processing_time / job.processed_files if job.processed_files > 0 else 0,
                    "files_per_second": job.processed_files / processing_time if processing_time > 0 else 0,
                    "errors": job.errors
                },
                "metadata": job.metadata
            }
            
            # Sauvegarde du rapport
            report_file = self.batch_results_path / f"{job.job_id}_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Rapport généré: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport {job.job_id}: {e}")
    
    async def _save_checkpoint(self, job: BatchJob):
        """💾 Sauvegarde un checkpoint"""
        
        try:
            checkpoint_data = {
                "job_id": job.job_id,
                "processed_files": job.processed_files,
                "successful_files": job.successful_files,
                "failed_files": job.failed_files,
                "results": job.results,
                "errors": job.errors,
                "timestamp": time.time()
            }
            
            checkpoint_file = self.checkpoints_path / f"{job.job_id}_checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde checkpoint {job.job_id}: {e}")
    
    async def _initialize_workers(self):
        """👷 Initialise le pool de workers"""
        
        # Créer différents types de workers
        total_workers = self.config.max_total_workers
        
        # Répartition: 60% mixed, 25% image, 15% video
        mixed_workers = int(total_workers * 0.6)
        image_workers = int(total_workers * 0.25)
        video_workers = total_workers - mixed_workers - image_workers
        
        worker_id = 1
        
        # Workers mixtes (peuvent traiter images et vidéos)
        for _ in range(mixed_workers):
            worker = BatchWorker(
                worker_id=f"mixed_{worker_id}",
                worker_type=WorkerType.MIXED_WORKER
            )
            self._workers[worker.worker_id] = worker
            worker_id += 1
        
        # Workers spécialisés images
        for _ in range(image_workers):
            worker = BatchWorker(
                worker_id=f"image_{worker_id}",
                worker_type=WorkerType.IMAGE_WORKER
            )
            self._workers[worker.worker_id] = worker
            worker_id += 1
        
        # Workers spécialisés vidéos
        for _ in range(video_workers):
            worker = BatchWorker(
                worker_id=f"video_{worker_id}",
                worker_type=WorkerType.VIDEO_WORKER
            )
            self._workers[worker.worker_id] = worker
            worker_id += 1
        
        logger.info(f"👷 {len(self._workers)} workers initialisés")
    
    async def _start_resource_monitoring(self):
        """📊 Démarre le monitoring des ressources"""
        
        async def monitor_resources():
            while True:
                try:
                    # Mémoire
                    memory = psutil.virtual_memory()
                    self._memory_usage_gb = memory.used / (1024**3)
                    
                    if self._memory_usage_gb > self.stats["peak_memory_gb"]:
                        self.stats["peak_memory_gb"] = self._memory_usage_gb
                    
                    # Workers actifs
                    active_workers = len([w for w in self._workers.values() if w.is_busy])
                    if active_workers > self.stats["peak_workers"]:
                        self.stats["peak_workers"] = active_workers
                    
                    # Gestion mémoire si nécessaire
                    if memory.percent > 90:
                        await self._manage_memory()
                    
                    await asyncio.sleep(self.config.memory_check_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur monitoring ressources: {e}")
                    await asyncio.sleep(60)  # Retry dans 1 minute
        
        # Lancer en tâche de fond
        asyncio.create_task(monitor_resources())
    
    async def _manage_memory(self):
        """💾 Gère la mémoire système"""
        
        # Nettoyer le cache des fichiers
        if len(self._file_cache) > 1000:
            # Garder seulement les 500 plus récents
            cache_items = list(self._file_cache.items())
            self._file_cache = dict(cache_items[-500:])
            logger.info("🧹 Cache des fichiers nettoyé")
        
        # Forcer garbage collection
        import gc
        gc.collect()
    
    async def _resume_interrupted_jobs(self):
        """🔄 Reprend les jobs interrompus"""
        
        try:
            checkpoint_files = list(self.checkpoints_path.glob("*_checkpoint.json"))
            
            for checkpoint_file in checkpoint_files:
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    job_id = checkpoint_data["job_id"]
                    
                    # Vérifier si le job existe encore
                    if job_id in self._jobs:
                        job = self._jobs[job_id]
                        
                        # Restaurer les données
                        job.processed_files = checkpoint_data["processed_files"]
                        job.successful_files = checkpoint_data["successful_files"]
                        job.failed_files = checkpoint_data["failed_files"]
                        job.results = checkpoint_data["results"]
                        job.errors = checkpoint_data["errors"]
                        
                        # Remettre en queue si pas terminé
                        if job.processed_files < job.total_files:
                            job.state = BatchState.PENDING
                            self._add_job_to_queue(job)
                            logger.info(f"🔄 Job repris: {job_id}")
                        else:
                            # Job était terminé
                            job.state = BatchState.COMPLETED
                    
                    # Supprimer le checkpoint
                    checkpoint_file.unlink()
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur reprise checkpoint {checkpoint_file}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Erreur reprise jobs interrompus: {e}")
    
    # API DE GESTION DES JOBS
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """📊 Retourne le statut d'un job"""
        
        if job_id not in self._jobs:
            return None
        
        job = self._jobs[job_id]
        
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "priority": job.priority.value,
            "progress": {
                "total_files": job.total_files,
                "processed_files": job.processed_files,
                "successful_files": job.successful_files,
                "failed_files": job.failed_files,
                "percentage": job.get_progress_percentage()
            },
            "timing": {
                "created_time": job.created_time,
                "started_time": job.started_time,
                "completed_time": job.completed_time,
                "estimated_time_remaining": job.get_estimated_time_remaining()
            },
            "metadata": job.metadata
        }
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """📊 Retourne le statut de tous les jobs"""
        
        return [
            self.get_job_status(job_id)
            for job_id in self._jobs.keys()
        ]
    
    async def cancel_job(self, job_id: str) -> bool:
        """❌ Annule un job"""
        
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.state = BatchState.CANCELLED
        
        # Retirer de la queue si en attente
        self._job_queue = deque([j for j in self._job_queue if j.job_id != job_id])
        
        # Retirer des jobs actifs
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]
        
        logger.info(f"❌ Job annulé: {job_id}")
        return True
    
    async def pause_job(self, job_id: str) -> bool:
        """⏸️ Met en pause un job"""
        
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        if job.state == BatchState.RUNNING:
            job.state = BatchState.PAUSED
            logger.info(f"⏸️ Job mis en pause: {job_id}")
            return True
        
        return False
    
    async def resume_job(self, job_id: str) -> bool:
        """▶️ Reprend un job en pause"""
        
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        if job.state == BatchState.PAUSED:
            job.state = BatchState.PENDING
            self._add_job_to_queue(job)
            await self._process_job_queue()
            logger.info(f"▶️ Job repris: {job_id}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Retourne les statistiques du service"""
        
        uptime_hours = (time.time() - self.stats["start_time"]) / 3600
        
        # Statistiques des workers
        worker_stats = {
            "total_workers": len(self._workers),
            "busy_workers": len([w for w in self._workers.values() if w.is_busy]),
            "worker_efficiency": {
                worker.worker_id: {
                    "processed": worker.processed_count,
                    "errors": worker.error_count,
                    "efficiency": worker.get_efficiency(),
                    "avg_time": worker.get_average_time()
                }
                for worker in self._workers.values()
            }
        }
        
        return {
            "service_type": "batch_processing",
            "uptime_hours": uptime_hours,
            "jobs": {
                "total_jobs": self.stats["total_jobs"],
                "completed_jobs": self.stats["completed_jobs"],
                "failed_jobs": self.stats["failed_jobs"],
                "active_jobs": len(self._active_jobs),
                "queued_jobs": len(self._job_queue)
            },
            "performance": {
                "total_files_processed": self.stats["total_files_processed"],
                "total_processing_time": self.stats["total_processing_time"],
                "avg_files_per_hour": (
                    self.stats["total_files_processed"] / uptime_hours 
                    if uptime_hours > 0 else 0
                ),
                "files_per_second": (
                    self.stats["total_files_processed"] / self.stats["total_processing_time"]
                    if self.stats["total_processing_time"] > 0 else 0
                )
            },
            "resources": {
                "memory_usage_gb": self._memory_usage_gb,
                "peak_memory_gb": self.stats["peak_memory_gb"],
                "peak_workers": self.stats["peak_workers"],
                "cache_size": len(self._file_cache)
            },
            "workers": worker_stats
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        logger.info("🧹 Nettoyage BatchService...")
        
        # Annuler tous les jobs actifs
        for job_id in list(self._active_jobs.keys()):
            await self.cancel_job(job_id)
        
        # Fermer l'executor
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Nettoyer les caches
        self._file_cache.clear()
        
        # Nettoyer les services
        await self.image_service.cleanup()
        await self.video_service.cleanup()
        
        logger.info("✅ BatchService nettoyé")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "BatchService",
    "BatchProcessingConfig",
    "BatchPriority",
    "BatchStrategy",
    "BatchState",
    "WorkerType",
    "BatchWorker",
    "BatchJob"
]