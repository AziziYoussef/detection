# app/api/endpoints/video_detection.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import tempfile
import os
import uuid
import logging
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from app.schemas.detection import DetectionResponse, LostObjectAlert
from app.utils.image_utils import validate_image, get_image_info
from app.core.model_manager import ModelManager
from app.config.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Stockage des tâches de traitement vidéo
video_tasks: Dict[str, Dict] = {}

class VideoProcessor:
    """Processeur de vidéo pour la détection"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.temp_dir = settings.TEMP_DIR
    
    async def process_video(self, video_path: Path, task_id: str, **kwargs):
        """Traite une vidéo complète"""
        try:
            # Mise à jour du statut
            video_tasks[task_id]['status'] = 'processing'
            video_tasks[task_id]['started_at'] = datetime.now()
            
            # Ouverture de la vidéo
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError("Impossible d'ouvrir la vidéo")
            
            # Informations vidéo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_tasks[task_id].update({
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'resolution': (width, height),
                'processed_frames': 0
            })
            
            # Paramètres de traitement
            model_name = kwargs.get('model_name', 'stable_epoch_30')
            confidence_threshold = kwargs.get('confidence_threshold', 0.5)
            frame_skip = kwargs.get('frame_skip', 1)  # Traiter 1 frame sur N
            max_frames = kwargs.get('max_frames', None)
            
            # Récupération du détecteur
            detector = await self.model_manager.get_detector(model_name)
            
            # Résultats de traitement
            detections_timeline = []
            alerts_timeline = []
            frame_count = 0
            processed_count = 0
            
            logger.info(f"Traitement vidéo {task_id}: {total_frames} frames, {fps:.1f} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames si nécessaire
                if frame_count % frame_skip != 0:
                    continue
                
                # Limite de frames si spécifiée
                if max_frames and processed_count >= max_frames:
                    break
                
                # Validation de la frame
                if not validate_image(frame):
                    continue
                
                try:
                    # Détection sur cette frame
                    objects, persons, alerts = detector.detect(
                        frame,
                        confidence_threshold=confidence_threshold,
                        enable_tracking=True,
                        enable_lost_detection=True
                    )
                    
                    # Timestamp de la frame
                    timestamp = frame_count / fps if fps > 0 else processed_count
                    
                    # Stockage des résultats
                    frame_result = {
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'objects_count': len(objects),
                        'persons_count': len(persons),
                        'objects': [self._serialize_object(obj) for obj in objects],
                        'persons': [self._serialize_person(person) for person in persons]
                    }
                    detections_timeline.append(frame_result)
                    
                    # Alertes
                    for alert in alerts:
                        alert_data = alert.dict()
                        alert_data['frame_number'] = frame_count
                        alert_data['timestamp'] = timestamp
                        alerts_timeline.append(alert_data)
                    
                    processed_count += 1
                    
                    # Mise à jour du progrès
                    video_tasks[task_id]['processed_frames'] = processed_count
                    video_tasks[task_id]['progress'] = (frame_count / total_frames) * 100
                    
                    # Log périodique
                    if processed_count % 100 == 0:
                        logger.info(f"Vidéo {task_id}: {processed_count} frames traitées "
                                  f"({video_tasks[task_id]['progress']:.1f}%)")
                
                except Exception as e:
                    logger.error(f"Erreur traitement frame {frame_count}: {e}")
                    continue
            
            # Fermeture
            cap.release()
            
            # Génération du rapport final
            report = self._generate_video_report(
                detections_timeline, alerts_timeline, video_tasks[task_id]
            )
            
            # Sauvegarde du rapport
            report_path = self.temp_dir / f"report_{task_id}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Finalisation
            video_tasks[task_id].update({
                'status': 'completed',
                'completed_at': datetime.now(),
                'report_path': str(report_path),
                'detections_count': len(detections_timeline),
                'alerts_count': len(alerts_timeline),
                'report': report
            })
            
            logger.info(f"Vidéo {task_id} traitée: {processed_count} frames, "
                       f"{len(alerts_timeline)} alertes")
            
        except Exception as e:
            logger.error(f"Erreur traitement vidéo {task_id}: {e}")
            video_tasks[task_id].update({
                'status': 'error',
                'error': str(e),
                'completed_at': datetime.now()
            })
            raise
        
        finally:
            # Nettoyage du fichier temporaire
            try:
                if video_path.exists():
                    os.unlink(video_path)
            except Exception as e:
                logger.warning(f"Impossible de supprimer {video_path}: {e}")
    
    def _serialize_object(self, obj) -> dict:
        """Sérialise un objet pour JSON"""
        return {
            'object_id': obj.object_id,
            'class_name': obj.class_name,
            'class_name_fr': obj.class_name_fr,
            'confidence': obj.confidence,
            'bounding_box': {
                'x': obj.bounding_box.x,
                'y': obj.bounding_box.y,
                'width': obj.bounding_box.width,
                'height': obj.bounding_box.height
            },
            'status': obj.status.value,
            'duration_stationary': obj.duration_stationary
        }
    
    def _serialize_person(self, person) -> dict:
        """Sérialise une personne pour JSON"""
        return {
            'person_id': person.person_id,
            'confidence': person.confidence,
            'bounding_box': {
                'x': person.bounding_box.x,
                'y': person.bounding_box.y,
                'width': person.bounding_box.width,
                'height': person.bounding_box.height
            },
            'position': person.position
        }
    
    def _generate_video_report(self, detections: List, alerts: List, task_info: dict) -> dict:
        """Génère un rapport de traitement vidéo"""
        # Statistiques globales
        total_objects = sum(frame['objects_count'] for frame in detections)
        total_persons = sum(frame['persons_count'] for frame in detections)
        
        # Analyse temporelle des objets perdus
        lost_objects_over_time = []
        for frame in detections:
            lost_count = sum(1 for obj in frame['objects'] 
                           if obj['status'] in ['lost', 'critical'])
            lost_objects_over_time.append({
                'timestamp': frame['timestamp'],
                'frame_number': frame['frame_number'],
                'lost_objects': lost_count
            })
        
        # Distribution des classes d'objets
        class_distribution = {}
        for frame in detections:
            for obj in frame['objects']:
                class_name = obj['class_name_fr']
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        # Alertes par période
        alerts_by_minute = {}
        for alert in alerts:
            minute = int(alert['timestamp'] // 60)
            alerts_by_minute[minute] = alerts_by_minute.get(minute, 0) + 1
        
        # Périodes critiques (> 2 alertes par minute)
        critical_periods = [
            {'minute': minute, 'alerts_count': count}
            for minute, count in alerts_by_minute.items()
            if count > 2
        ]
        
        return {
            'task_info': {
                'task_id': task_info['task_id'],
                'video_duration': task_info['duration'],
                'total_frames': task_info['total_frames'],
                'processed_frames': task_info['processed_frames'],
                'fps': task_info['fps'],
                'resolution': task_info['resolution'],
                'processing_time': (task_info.get('completed_at', datetime.now()) - 
                                  task_info.get('started_at', datetime.now())).total_seconds()
            },
            'statistics': {
                'total_detections': len(detections),
                'total_objects': total_objects,
                'total_persons': total_persons,
                'total_alerts': len(alerts),
                'avg_objects_per_frame': total_objects / len(detections) if detections else 0,
                'avg_persons_per_frame': total_persons / len(detections) if detections else 0
            },
            'class_distribution': class_distribution,
            'lost_objects_timeline': lost_objects_over_time,
            'alerts_timeline': alerts,
            'critical_periods': critical_periods,
            'recommendations': self._generate_recommendations(
                detections, alerts, class_distribution
            )
        }
    
    def _generate_recommendations(self, detections: List, alerts: List, 
                                class_distribution: dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        if len(alerts) > 10:
            recommendations.append("🚨 Nombre élevé d'alertes détecté - vérifier la configuration des seuils")
        
        if len(detections) > 0:
            avg_objects = sum(f['objects_count'] for f in detections) / len(detections)
            if avg_objects > 10:
                recommendations.append("📦 Scène dense détectée - considérer l'optimisation des paramètres NMS")
        
        # Top objets perdus
        top_lost_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_lost_classes:
            top_class = top_lost_classes[0][0]
            recommendations.append(f"📊 Objet le plus fréquent: {top_class} - surveillance renforcée recommandée")
        
        if not recommendations:
            recommendations.append("✅ Aucune anomalie détectée - surveillance normale")
        
        return recommendations

async def get_model_manager(request: Request) -> ModelManager:
    """Récupère le gestionnaire de modèles"""
    return request.app.state.model_manager

@router.post("/video")
async def detect_objects_in_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("stable_epoch_30"),
    confidence_threshold: Optional[float] = Form(0.5),
    frame_skip: Optional[int] = Form(1),
    max_frames: Optional[int] = Form(None),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🎬 Lance le traitement d'une vidéo pour détecter les objets perdus
    
    **Paramètres:**
    - file: Fichier vidéo (MP4, AVI, MOV, etc.)
    - model_name: Modèle à utiliser pour la détection
    - confidence_threshold: Seuil de confiance (0.0-1.0)
    - frame_skip: Traiter 1 frame sur N (1=toutes, 2=une sur deux, etc.)
    - max_frames: Limite du nombre de frames à traiter
    
    **Retour:**
    - task_id: ID de la tâche pour suivre le progrès
    - status_url: URL pour suivre l'avancement
    """
    
    try:
        # Validation du fichier
        if not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté: {file.content_type}. "
                       f"Formats supportés: MP4, AVI, MOV, MKV"
            )
        
        # Validation des paramètres
        if not (0.0 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="confidence_threshold doit être entre 0.0 et 1.0"
            )
        
        if frame_skip < 1:
            raise HTTPException(
                status_code=400,
                detail="frame_skip doit être >= 1"
            )
        
        if max_frames is not None and max_frames < 1:
            raise HTTPException(
                status_code=400,
                detail="max_frames doit être >= 1"
            )
        
        # Génération de l'ID de tâche
        task_id = str(uuid.uuid4())
        
        # Sauvegarde temporaire du fichier
        temp_path = settings.TEMP_DIR / f"video_{task_id}_{file.filename}"
        
        try:
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur sauvegarde fichier: {str(e)}"
            )
        
        # Validation basique de la vidéo
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400,
                detail="Fichier vidéo invalide ou corrompu"
            )
        
        # Informations basiques
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Vérification des limites
        if duration > 1800:  # 30 minutes max
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Vidéo trop longue: {duration/60:.1f} min. Maximum: 30 min"
            )
        
        # Création de la tâche
        video_tasks[task_id] = {
            'task_id': task_id,
            'filename': file.filename,
            'status': 'queued',
            'created_at': datetime.now(),
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'progress': 0,
            'model_name': model_name,
            'confidence_threshold': confidence_threshold,
            'frame_skip': frame_skip,
            'max_frames': max_frames
        }
        
        # Lancement du traitement en arrière-plan
        processor = VideoProcessor(model_manager)
        background_tasks.add_task(
            processor.process_video,
            temp_path,
            task_id,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            frame_skip=frame_skip,
            max_frames=max_frames
        )
        
        logger.info(f"Tâche vidéo lancée: {task_id} ({file.filename}, {duration:.1f}s)")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"Traitement vidéo lancé pour {file.filename}",
            "estimated_duration": f"{duration:.1f} secondes",
            "total_frames": total_frames,
            "status_url": f"/api/v1/detect/video/status/{task_id}",
            "download_url": f"/api/v1/detect/video/report/{task_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur traitement vidéo: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne: {str(e)}"
        )

@router.get("/video/status/{task_id}")
async def get_video_task_status(task_id: str):
    """
    📊 Récupère le statut d'une tâche de traitement vidéo
    
    **États possibles:**
    - queued: En attente de traitement
    - processing: En cours de traitement
    - completed: Terminé avec succès
    - error: Erreur lors du traitement
    """
    
    if task_id not in video_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Tâche {task_id} non trouvée"
        )
    
    task = video_tasks[task_id]
    
    # Calcul du temps écoulé
    now = datetime.now()
    elapsed = (now - task['created_at']).total_seconds()
    
    # Estimation du temps restant
    eta = None
    if task['status'] == 'processing' and task.get('progress', 0) > 0:
        progress_ratio = task['progress'] / 100
        estimated_total = elapsed / progress_ratio
        eta = max(0, estimated_total - elapsed)
    
    response = {
        "task_id": task_id,
        "status": task['status'],
        "progress": task.get('progress', 0),
        "created_at": task['created_at'].isoformat(),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": eta,
        "filename": task['filename'],
        "total_frames": task.get('total_frames', 0),
        "processed_frames": task.get('processed_frames', 0)
    }
    
    # Informations additionnelles selon le statut
    if task['status'] == 'completed':
        response.update({
            "completed_at": task.get('completed_at', '').isoformat() if task.get('completed_at') else '',
            "detections_count": task.get('detections_count', 0),
            "alerts_count": task.get('alerts_count', 0),
            "report_available": True
        })
    
    elif task['status'] == 'error':
        response.update({
            "error": task.get('error', 'Erreur inconnue'),
            "completed_at": task.get('completed_at', '').isoformat() if task.get('completed_at') else ''
        })
    
    return response

@router.get("/video/report/{task_id}")
async def download_video_report(task_id: str):
    """
    📥 Télécharge le rapport de traitement vidéo
    
    Retourne un fichier JSON contenant:
    - Timeline des détections
    - Alertes générées
    - Statistiques globales
    - Recommandations
    """
    
    if task_id not in video_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Tâche {task_id} non trouvée"
        )
    
    task = video_tasks[task_id]
    
    if task['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Tâche pas encore terminée (statut: {task['status']})"
        )
    
    report_path = task.get('report_path')
    if not report_path or not Path(report_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Rapport non disponible"
        )
    
    return FileResponse(
        path=report_path,
        filename=f"video_report_{task_id}.json",
        media_type="application/json"
    )

@router.get("/video/tasks")
async def list_video_tasks(limit: int = 50, status: Optional[str] = None):
    """
    📋 Liste les tâches de traitement vidéo
    
    **Paramètres:**
    - limit: Nombre maximum de tâches à retourner
    - status: Filtrer par statut (queued, processing, completed, error)
    """
    
    tasks = list(video_tasks.values())
    
    # Filtrage par statut
    if status:
        tasks = [task for task in tasks if task['status'] == status]
    
    # Tri par date de création (plus récent en premier)
    tasks.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limitation
    tasks = tasks[:limit]
    
    # Formatage pour la réponse
    formatted_tasks = []
    for task in tasks:
        formatted_task = {
            "task_id": task['task_id'],
            "filename": task['filename'],
            "status": task['status'],
            "progress": task.get('progress', 0),
            "created_at": task['created_at'].isoformat(),
            "duration": task.get('duration', 0)
        }
        
        if task['status'] == 'completed':
            formatted_task.update({
                "completed_at": task.get('completed_at', '').isoformat() if task.get('completed_at') else '',
                "alerts_count": task.get('alerts_count', 0)
            })
        
        formatted_tasks.append(formatted_task)
    
    return {
        "success": True,
        "total_tasks": len(video_tasks),
        "filtered_tasks": len(formatted_tasks),
        "tasks": formatted_tasks
    }

@router.delete("/video/task/{task_id}")
async def delete_video_task(task_id: str):
    """
    🗑️ Supprime une tâche de traitement vidéo
    
    Nettoie les fichiers temporaires et supprime la tâche de la mémoire
    """
    
    if task_id not in video_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Tâche {task_id} non trouvée"
        )
    
    task = video_tasks[task_id]
    
    # Nettoyage des fichiers
    try:
        report_path = task.get('report_path')
        if report_path and Path(report_path).exists():
            os.unlink(report_path)
    except Exception as e:
        logger.warning(f"Impossible de supprimer le rapport {task_id}: {e}")
    
    # Suppression de la mémoire
    del video_tasks[task_id]
    
    logger.info(f"Tâche {task_id} supprimée")
    
    return {
        "success": True,
        "message": f"Tâche {task_id} supprimée"
    }