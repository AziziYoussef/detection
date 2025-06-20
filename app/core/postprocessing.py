"""
🔄 POSTPROCESSING - POST-TRAITEMENT ET LOGIQUE OBJETS PERDUS
==========================================================
Pipeline de post-traitement pour transformer les détections brutes en objets perdus

Fonctionnalités:
- Filtrage et validation des détections
- Tracking temporel des objets
- Logique métier des objets perdus
- Analyse contextuelle (proximité propriétaire)
- Gestion des états et alertes
- Optimisation NMS et suppression doublons
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import uuid
import threading

import numpy as np
import torch
from scipy.spatial.distance import euclidean

# Imports internes
from app.schemas.detection import (
    DetectionResult, BoundingBox, LostObjectState, ObjectStatus,
    DetectionMode
)
from app.config.config import Settings

logger = logging.getLogger(__name__)

class ObjectTracker:
    """📍 Tracker pour objets individuels"""
    
    def __init__(self, object_id: str, initial_detection: DetectionResult):
        self.object_id = object_id
        self.class_name = initial_detection.class_name
        
        # Historique temporal
        self.detections_history: deque = deque(maxlen=100)
        self.positions_history: deque = deque(maxlen=50)
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.last_movement = datetime.now()
        
        # État actuel
        self.current_bbox = initial_detection.bbox
        self.current_confidence = initial_detection.confidence
        self.status = ObjectStatus.NORMAL
        
        # Métriques de tracking
        self.tracking_stability = 1.0
        self.detection_count = 1
        self.movement_distance = 0.0
        
        # Contextuel
        self.nearest_person_distance = float('inf')
        self.has_owner_nearby = False
        self.location_context = None
        
        # Alertes
        self.alert_level = 0
        self.alerts_sent: List[str] = []
        
        # Ajout de la première détection
        self.detections_history.append(initial_detection)
        self.positions_history.append(initial_detection.bbox)
    
    def update(self, detection: DetectionResult) -> bool:
        """🔄 Met à jour le tracker avec une nouvelle détection"""
        
        # Vérification de cohérence
        if detection.class_name != self.class_name:
            return False
        
        # Calcul de la distance de mouvement
        center_old = self.current_bbox.center
        center_new = detection.bbox.center
        movement = euclidean(center_old, center_new)
        
        # Mise à jour de l'historique
        self.detections_history.append(detection)
        self.positions_history.append(detection.bbox)
        
        # Mise à jour des propriétés
        self.current_bbox = detection.bbox
        self.current_confidence = detection.confidence
        self.last_seen = datetime.now()
        self.detection_count += 1
        
        # Mise à jour du mouvement
        if movement > 10:  # Seuil de mouvement en pixels
            self.movement_distance += movement
            self.last_movement = datetime.now()
        
        # Calcul de la stabilité du tracking
        self._update_tracking_stability()
        
        return True
    
    def _update_tracking_stability(self):
        """📊 Met à jour la stabilité du tracking"""
        
        if len(self.detections_history) < 2:
            self.tracking_stability = 1.0
            return
        
        # Stabilité basée sur la cohérence des confidences
        confidences = [d.confidence for d in list(self.detections_history)[-10:]]
        confidence_std = np.std(confidences)
        confidence_stability = max(0.0, 1.0 - confidence_std)
        
        # Stabilité basée sur la cohérence des positions
        if len(self.positions_history) >= 3:
            positions = [(bbox.center[0], bbox.center[1]) for bbox in list(self.positions_history)[-5:]]
            position_variance = np.var([euclidean(positions[i], positions[i-1]) 
                                      for i in range(1, len(positions))])
            position_stability = max(0.0, 1.0 - position_variance / 1000)  # Normalisation
        else:
            position_stability = 1.0
        
        # Stabilité combinée
        self.tracking_stability = (confidence_stability + position_stability) / 2
    
    @property
    def stationary_duration(self) -> int:
        """⏱️ Durée d'immobilité en secondes"""
        return int((datetime.now() - self.last_movement).total_seconds())
    
    @property
    def total_duration(self) -> int:
        """⏱️ Durée totale de tracking en secondes"""
        return int((self.last_seen - self.first_seen).total_seconds())
    
    @property
    def is_lost_candidate(self) -> bool:
        """🚨 Vérifie si l'objet est candidat "perdu" """
        return (
            self.stationary_duration > 60 and  # Immobile > 1 minute
            not self.has_owner_nearby and     # Pas de propriétaire proche
            self.tracking_stability > 0.7     # Tracking stable
        )

class LostObjectDetector:
    """🚨 Détecteur logique d'objets perdus"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Seuils configurables
        self.suspect_duration = settings.SUSPECT_DURATION
        self.lost_duration = settings.LOST_DURATION
        self.critical_duration = settings.CRITICAL_DURATION
        self.owner_proximity_radius = settings.OWNER_PROXIMITY_RADIUS
        self.movement_threshold = settings.MOVEMENT_THRESHOLD
        
        # Classes considérées comme objets personnels
        self.personal_objects = {
            'backpack', 'suitcase', 'handbag', 'cell phone',
            'laptop', 'umbrella', 'book', 'bottle'
        }
        
        # Tracking des objets
        self.active_trackers: Dict[str, ObjectTracker] = {}
        self.resolved_objects: Set[str] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("🚨 LostObjectDetector initialisé")
    
    def analyze_detections(
        self,
        detections: List[DetectionResult],
        frame_timestamp: Optional[datetime] = None
    ) -> List[LostObjectState]:
        """🔍 Analyse les détections pour identifier les objets perdus"""
        
        with self._lock:
            current_time = frame_timestamp or datetime.now()
            
            # 1. Séparer personnes et objets
            persons = [d for d in detections if d.class_name == 'person']
            objects = [d for d in detections if d.class_name in self.personal_objects]
            
            # 2. Associer détections aux trackers existants
            self._update_trackers(objects, current_time)
            
            # 3. Analyser la proximité avec les personnes
            self._analyze_person_proximity(persons)
            
            # 4. Mettre à jour les statuts des objets
            self._update_object_statuses()
            
            # 5. Générer les états des objets perdus
            lost_objects = self._generate_lost_object_states()
            
            # 6. Nettoyage des trackers obsolètes
            self._cleanup_old_trackers()
            
            return lost_objects
    
    def _update_trackers(self, detections: List[DetectionResult], timestamp: datetime):
        """🔄 Met à jour les trackers avec nouvelles détections"""
        
        unmatched_detections = []
        matched_tracker_ids = set()
        
        # Tentative d'association avec trackers existants
        for detection in detections:
            best_tracker_id = None
            best_score = 0.0
            
            for tracker_id, tracker in self.active_trackers.items():
                if tracker.class_name != detection.class_name:
                    continue
                
                # Calcul score de similarité (position + IoU)
                score = self._calculate_association_score(detection.bbox, tracker.current_bbox)
                
                if score > best_score and score > 0.3:  # Seuil minimum
                    best_score = score
                    best_tracker_id = tracker_id
            
            # Association ou création nouveau tracker
            if best_tracker_id and best_tracker_id not in matched_tracker_ids:
                self.active_trackers[best_tracker_id].update(detection)
                matched_tracker_ids.add(best_tracker_id)
            else:
                unmatched_detections.append(detection)
        
        # Créer nouveaux trackers pour détections non associées
        for detection in unmatched_detections:
            tracker_id = str(uuid.uuid4())
            self.active_trackers[tracker_id] = ObjectTracker(tracker_id, detection)
            logger.debug(f"🆕 Nouveau tracker créé: {tracker_id} ({detection.class_name})")
    
    def _calculate_association_score(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """📊 Calcule le score d'association entre deux boîtes"""
        
        # IoU (Intersection over Union)
        iou = bbox1.iou(bbox2)
        
        # Distance des centres
        center1 = bbox1.center
        center2 = bbox2.center
        distance = euclidean(center1, center2)
        
        # Normalisation de la distance (plus proche = meilleur score)
        max_distance = 200  # pixels
        distance_score = max(0, 1 - distance / max_distance)
        
        # Score combiné
        combined_score = (iou * 0.7) + (distance_score * 0.3)
        
        return combined_score
    
    def _analyze_person_proximity(self, persons: List[DetectionResult]):
        """👥 Analyse la proximité des objets avec les personnes"""
        
        for tracker in self.active_trackers.values():
            min_distance = float('inf')
            
            # Calculer distance avec chaque personne
            for person in persons:
                distance = self._calculate_object_person_distance(
                    tracker.current_bbox, person.bbox
                )
                min_distance = min(min_distance, distance)
            
            # Mise à jour du tracker
            tracker.nearest_person_distance = min_distance
            tracker.has_owner_nearby = min_distance <= self.owner_proximity_radius
    
    def _calculate_object_person_distance(
        self, 
        object_bbox: BoundingBox, 
        person_bbox: BoundingBox
    ) -> float:
        """📏 Calcule la distance objet-personne (en mètres estimés)"""
        
        # Distance entre centres (en pixels)
        obj_center = object_bbox.center
        person_center = person_bbox.center
        pixel_distance = euclidean(obj_center, person_center)
        
        # Conversion approximative pixels → mètres
        # Approximation basée sur la taille moyenne d'une personne
        person_height_pixels = person_bbox.height
        assumed_person_height_meters = 1.7  # mètres
        
        if person_height_pixels > 0:
            pixels_per_meter = person_height_pixels / assumed_person_height_meters
            distance_meters = pixel_distance / pixels_per_meter
        else:
            # Fallback avec estimation grossière
            distance_meters = pixel_distance / 100  # Approximation
        
        return distance_meters
    
    def _update_object_statuses(self):
        """🚦 Met à jour les statuts des objets selon la logique métier"""
        
        for tracker in self.active_trackers.values():
            previous_status = tracker.status
            new_status = self._determine_object_status(tracker)
            
            if new_status != previous_status:
                tracker.status = new_status
                self._handle_status_change(tracker, previous_status, new_status)
    
    def _determine_object_status(self, tracker: ObjectTracker) -> ObjectStatus:
        """🎯 Détermine le statut d'un objet selon la logique métier"""
        
        stationary_time = tracker.stationary_duration
        has_owner = tracker.has_owner_nearby
        stability = tracker.tracking_stability
        
        # Conditions de base
        if has_owner or stationary_time < self.suspect_duration:
            return ObjectStatus.NORMAL
        
        # Objet suspect (surveillance)
        if (stationary_time >= self.suspect_duration and 
            stationary_time < self.lost_duration and
            stability > 0.5):
            return ObjectStatus.SUSPECT
        
        # Objet perdu
        if (stationary_time >= self.lost_duration and
            stationary_time < self.critical_duration and
            not has_owner and
            stability > 0.7):
            return ObjectStatus.LOST
        
        # Objet critique (escalade)
        if (stationary_time >= self.critical_duration and
            not has_owner and
            stability > 0.7):
            return ObjectStatus.CRITICAL
        
        return ObjectStatus.NORMAL
    
    def _handle_status_change(
        self, 
        tracker: ObjectTracker, 
        old_status: ObjectStatus, 
        new_status: ObjectStatus
    ):
        """🔔 Gère les changements de statut"""
        
        logger.info(f"🚦 Changement statut {tracker.object_id}: {old_status} → {new_status}")
        
        # Mise à jour niveau d'alerte
        if new_status == ObjectStatus.SUSPECT:
            tracker.alert_level = 1
        elif new_status == ObjectStatus.LOST:
            tracker.alert_level = 3
        elif new_status == ObjectStatus.CRITICAL:
            tracker.alert_level = 5
        else:
            tracker.alert_level = 0
        
        # Génération d'alertes
        if new_status in [ObjectStatus.LOST, ObjectStatus.CRITICAL]:
            alert_message = f"Objet {tracker.class_name} {new_status.value} détecté"
            tracker.alerts_sent.append(alert_message)
            
            # TODO: Intégration avec système d'alertes (webhook, email, etc.)
            logger.warning(f"🚨 ALERTE: {alert_message}")
    
    def _generate_lost_object_states(self) -> List[LostObjectState]:
        """📋 Génère les états des objets perdus"""
        
        lost_objects = []
        
        for tracker in self.active_trackers.values():
            # Ne retourner que les objets suspects ou perdus
            if tracker.status in [ObjectStatus.SUSPECT, ObjectStatus.LOST, ObjectStatus.CRITICAL]:
                
                # Récupération de la dernière détection
                latest_detection = list(tracker.detections_history)[-1]
                
                # Création de l'état
                lost_object = LostObjectState(
                    object_id=tracker.object_id,
                    detection_result=latest_detection,
                    first_seen=tracker.first_seen,
                    last_seen=tracker.last_seen,
                    last_movement=tracker.last_movement,
                    stationary_duration=tracker.stationary_duration,
                    positions_history=list(tracker.positions_history),
                    movement_distance=tracker.movement_distance,
                    nearest_person_distance=tracker.nearest_person_distance,
                    has_owner_nearby=tracker.has_owner_nearby,
                    location_context=tracker.location_context,
                    status=tracker.status,
                    alert_level=tracker.alert_level,
                    alerts_sent=tracker.alerts_sent.copy(),
                    tracking_stability=tracker.tracking_stability,
                    detection_consistency=tracker.current_confidence
                )
                
                lost_objects.append(lost_object)
        
        return lost_objects
    
    def _cleanup_old_trackers(self):
        """🧹 Nettoie les trackers obsolètes"""
        
        current_time = datetime.now()
        timeout_duration = timedelta(minutes=30)  # Timeout après 30 minutes
        
        trackers_to_remove = []
        
        for tracker_id, tracker in self.active_trackers.items():
            time_since_last_seen = current_time - tracker.last_seen
            
            if time_since_last_seen > timeout_duration:
                trackers_to_remove.append(tracker_id)
                
                # Marquer comme résolu si c'était un objet perdu
                if tracker.status in [ObjectStatus.LOST, ObjectStatus.CRITICAL]:
                    self.resolved_objects.add(tracker_id)
                    logger.info(f"✅ Objet résolu (timeout): {tracker_id}")
        
        # Suppression des trackers obsolètes
        for tracker_id in trackers_to_remove:
            del self.active_trackers[tracker_id]
            logger.debug(f"🗑️ Tracker supprimé: {tracker_id}")
    
    def mark_object_resolved(self, object_id: str) -> bool:
        """✅ Marque un objet comme résolu manuellement"""
        
        with self._lock:
            if object_id in self.active_trackers:
                tracker = self.active_trackers[object_id]
                tracker.status = ObjectStatus.RESOLVED
                self.resolved_objects.add(object_id)
                
                logger.info(f"✅ Objet marqué résolu: {object_id}")
                return True
            
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques du détecteur d'objets perdus"""
        
        with self._lock:
            active_count = len(self.active_trackers)
            resolved_count = len(self.resolved_objects)
            
            status_counts = defaultdict(int)
            for tracker in self.active_trackers.values():
                status_counts[tracker.status.value] += 1
            
            return {
                "active_trackers": active_count,
                "resolved_objects": resolved_count,
                "status_distribution": dict(status_counts),
                "configuration": {
                    "suspect_duration": self.suspect_duration,
                    "lost_duration": self.lost_duration,
                    "critical_duration": self.critical_duration,
                    "owner_proximity_radius": self.owner_proximity_radius
                }
            }

class ResultPostprocessor:
    """🔄 Post-processeur principal des résultats"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lost_object_detector = LostObjectDetector(settings)
        
        # Cache des résultats récents
        self._results_cache: Dict[str, Any] = {}
        
        logger.info("🔄 ResultPostprocessor initialisé")
    
    def process_detections(
        self,
        raw_detections: List[DetectionResult],
        enable_lost_detection: bool = True,
        frame_timestamp: Optional[datetime] = None
    ) -> Tuple[List[DetectionResult], List[LostObjectState]]:
        """🔄 Traite les détections brutes"""
        
        # 1. Filtrage et validation des détections
        filtered_detections = self._filter_detections(raw_detections)
        
        # 2. Application NMS si nécessaire
        nms_detections = self._apply_nms(filtered_detections)
        
        # 3. Détection d'objets perdus
        lost_objects = []
        if enable_lost_detection:
            lost_objects = self.lost_object_detector.analyze_detections(
                nms_detections, frame_timestamp
            )
        
        return nms_detections, lost_objects
    
    def _filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """🔍 Filtre les détections selon critères de qualité"""
        
        filtered = []
        
        for detection in detections:
            # Filtres de base
            if (detection.confidence >= self.settings.LOST_OBJECT_MIN_CONFIDENCE and
                detection.bbox.area >= self.settings.MIN_BOX_AREA):
                
                # Filtres spécialisés
                if self._is_valid_detection(detection):
                    filtered.append(detection)
        
        return filtered
    
    def _is_valid_detection(self, detection: DetectionResult) -> bool:
        """✅ Vérifie la validité d'une détection"""
        
        # Vérifications géométriques
        bbox = detection.bbox
        
        # Ratio aspect raisonnable
        aspect_ratio = bbox.width / max(bbox.height, 1)
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        
        # Taille minimale
        if bbox.width < 10 or bbox.height < 10:
            return False
        
        # Positions valides
        if bbox.x1 < 0 or bbox.y1 < 0:
            return False
        
        return True
    
    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """🎯 Applique Non-Maximum Suppression"""
        
        if len(detections) <= 1:
            return detections
        
        # Grouper par classe
        class_groups = defaultdict(list)
        for detection in detections:
            class_groups[detection.class_name].append(detection)
        
        # Appliquer NMS par classe
        final_detections = []
        
        for class_name, class_detections in class_groups.items():
            nms_detections = self._nms_per_class(class_detections)
            final_detections.extend(nms_detections)
        
        return final_detections
    
    def _nms_per_class(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """🎯 NMS pour une classe spécifique"""
        
        if len(detections) <= 1:
            return detections
        
        # Tri par confiance décroissante
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in suppressed:
                continue
            
            keep.append(detection)
            
            # Supprimer les détections avec IoU élevé
            for j in range(i + 1, len(sorted_detections)):
                if j in suppressed:
                    continue
                
                iou = detection.bbox.iou(sorted_detections[j].bbox)
                if iou > self.settings.DEFAULT_NMS_THRESHOLD:
                    suppressed.add(j)
        
        return keep

class DetectionFilter:
    """🔍 Filtre avancé pour les détections"""
    
    def __init__(self):
        # Filtres par classe
        self.class_filters = {
            'person': {'min_confidence': 0.3, 'min_area': 400},
            'cell phone': {'min_confidence': 0.6, 'min_area': 100},
            'laptop': {'min_confidence': 0.5, 'min_area': 2000},
            'backpack': {'min_confidence': 0.4, 'min_area': 500},
        }
        
        # Zones d'exclusion (coordonnées relatives)
        self.exclusion_zones = [
            # Exemple: zone en haut à gauche
            {'x1': 0.0, 'y1': 0.0, 'x2': 0.1, 'y2': 0.1}
        ]
    
    def filter_detections(
        self,
        detections: List[DetectionResult],
        confidence_threshold: float,
        max_detections: int
    ) -> List[DetectionResult]:
        """🔍 Filtre les détections selon critères avancés"""
        
        filtered = []
        
        for detection in detections:
            if self._should_keep_detection(detection, confidence_threshold):
                filtered.append(detection)
        
        # Limitation du nombre de détections
        if len(filtered) > max_detections:
            # Tri par confiance et garde les meilleures
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:max_detections]
        
        return filtered
    
    def _should_keep_detection(self, detection: DetectionResult, threshold: float) -> bool:
        """✅ Détermine si garder une détection"""
        
        # Seuil de confiance global
        if detection.confidence < threshold:
            return False
        
        # Filtres par classe
        class_filter = self.class_filters.get(detection.class_name, {})
        
        min_confidence = class_filter.get('min_confidence', threshold)
        if detection.confidence < min_confidence:
            return False
        
        min_area = class_filter.get('min_area', 25)
        if detection.bbox.area < min_area:
            return False
        
        # Zones d'exclusion
        if self._is_in_exclusion_zone(detection.bbox):
            return False
        
        return True
    
    def _is_in_exclusion_zone(self, bbox: BoundingBox) -> bool:
        """🚫 Vérifie si une boîte est dans une zone d'exclusion"""
        
        # Pour cet exemple, supposons image 640x480
        image_width, image_height = 640, 480
        
        bbox_center_x = bbox.center[0] / image_width
        bbox_center_y = bbox.center[1] / image_height
        
        for zone in self.exclusion_zones:
            if (zone['x1'] <= bbox_center_x <= zone['x2'] and
                zone['y1'] <= bbox_center_y <= zone['y2']):
                return True
        
        return False

# === EXPORTS ===
__all__ = [
    "ResultPostprocessor",
    "LostObjectDetector",
    "ObjectTracker",
    "DetectionFilter"
]