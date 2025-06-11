# app/core/detector.py
import torch
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import logging
from collections import defaultdict, deque

from app.schemas.detection import (
    ObjectDetection, PersonDetection, ObjectStatus, 
    DetectionConfidence, BoundingBox, LostObjectAlert
)
from app.utils.image_utils import ImageProcessor
from app.utils.box_utils import (
    nms_by_class, filter_by_confidence, filter_by_size,
    box_distance, box_iou, clip_boxes
)
from app.config.config import MODEL_CONFIG, LOST_OBJECT_CONFIG

logger = logging.getLogger(__name__)

class ObjectTracker:
    """Suivi des objets dans le temps"""
    
    def __init__(self, max_frames_missing: int = 30):
        self.tracks = {}  # track_id -> track_info
        self.max_frames_missing = max_frames_missing
        self.next_id = 1
    
    def update(self, detections: List[ObjectDetection]) -> List[ObjectDetection]:
        """Met à jour le suivi des objets"""
        # Marquer tous les tracks comme non vus
        for track_id in self.tracks:
            self.tracks[track_id]['frames_missing'] += 1
        
        # Associer les détections aux tracks existants
        for detection in detections:
            track_id = self._find_matching_track(detection)
            
            if track_id is None:
                # Nouveau track
                track_id = f"track_{self.next_id}"
                self.next_id += 1
                self.tracks[track_id] = {
                    'last_position': detection.bounding_box,
                    'last_seen': datetime.now(),
                    'frames_missing': 0,
                    'total_detections': 1,
                    'first_seen': detection.first_seen,
                    'class_name': detection.class_name
                }
            else:
                # Track existant
                self.tracks[track_id]['last_position'] = detection.bounding_box
                self.tracks[track_id]['last_seen'] = datetime.now()
                self.tracks[track_id]['frames_missing'] = 0
                self.tracks[track_id]['total_detections'] += 1
            
            detection.track_id = track_id
            detection.track_confidence = self._calculate_track_confidence(track_id)
        
        # Nettoyer les tracks perdus
        self._cleanup_lost_tracks()
        
        return detections
    
    def _find_matching_track(self, detection: ObjectDetection) -> Optional[str]:
        """Trouve le track correspondant à une détection"""
        best_match = None
        best_score = float('inf')
        
        for track_id, track_info in self.tracks.items():
            if track_info['frames_missing'] > self.max_frames_missing:
                continue
            
            if track_info['class_name'] != detection.class_name:
                continue
            
            # Distance entre les centres
            last_pos = track_info['last_position']
            current_pos = detection.bounding_box
            
            last_center = (last_pos.x + last_pos.width/2, last_pos.y + last_pos.height/2)
            current_center = (current_pos.x + current_pos.width/2, current_pos.y + current_pos.height/2)
            
            distance = np.sqrt((last_center[0] - current_center[0])**2 + 
                             (last_center[1] - current_center[1])**2)
            
            # Seuil adaptatif basé sur la taille de l'objet
            size_threshold = max(last_pos.width, last_pos.height) * 0.5
            
            if distance < size_threshold and distance < best_score:
                best_score = distance
                best_match = track_id
        
        return best_match
    
    def _calculate_track_confidence(self, track_id: str) -> float:
        """Calcule la confiance du suivi"""
        track_info = self.tracks[track_id]
        
        # Facteurs: durée de vie, détections consécutives, etc.
        lifetime = (datetime.now() - track_info['first_seen']).total_seconds()
        consistency = track_info['total_detections'] / max(1, lifetime / 2)  # détections par 2 secondes
        
        confidence = min(1.0, consistency * 0.5 + min(lifetime / 10, 0.5))
        return confidence
    
    def _cleanup_lost_tracks(self):
        """Nettoie les tracks perdus"""
        to_remove = []
        for track_id, track_info in self.tracks.items():
            if track_info['frames_missing'] > self.max_frames_missing:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]

class LostObjectDetector:
    """Détecteur d'objets perdus avec logique métier"""
    
    def __init__(self):
        self.object_states = {}  # object_id -> state_info
        self.alerts_history = deque(maxlen=1000)
        self.config = LOST_OBJECT_CONFIG
    
    def analyze_objects(self, objects: List[ObjectDetection], 
                       persons: List[PersonDetection]) -> Tuple[List[ObjectDetection], List[LostObjectAlert]]:
        """Analyse les objets pour détecter ceux perdus"""
        alerts = []
        
        for obj in objects:
            # Ignorer les objets en blacklist
            if obj.class_name in self.config['blacklist_objects']:
                obj.status = ObjectStatus.NORMAL
                continue
            
            # Analyser l'état de l'objet
            previous_status = obj.status
            obj = self._analyze_object_state(obj, persons)
            
            # Générer une alerte si changement d'état critique
            if self._should_generate_alert(obj, previous_status):
                alert = self._create_alert(obj)
                alerts.append(alert)
                self.alerts_history.append(alert)
        
        return objects, alerts
    
    def _analyze_object_state(self, obj: ObjectDetection, 
                            persons: List[PersonDetection]) -> ObjectDetection:
        """Analyse l'état d'un objet spécifique"""
        now = datetime.now()
        
        # Initialiser l'état si nouveau
        if obj.object_id not in self.object_states:
            self.object_states[obj.object_id] = {
                'first_detection': now,
                'last_movement': now,
                'position_history': deque(maxlen=10),
                'status_history': deque(maxlen=20),
                'owner_last_seen': now
            }
        
        state = self.object_states[obj.object_id]
        
        # Mettre à jour l'historique des positions
        current_pos = (obj.bounding_box.x + obj.bounding_box.width/2,
                      obj.bounding_box.y + obj.bounding_box.height/2)
        state['position_history'].append((now, current_pos))
        
        # Détecter le mouvement
        movement_detected = self._detect_movement(state['position_history'])
        if movement_detected:
            state['last_movement'] = now
            obj.last_movement = now
        
        # Calculer la durée immobile
        time_stationary = (now - state['last_movement']).total_seconds()
        obj.duration_stationary = time_stationary
        
        # Trouver le propriétaire le plus proche
        nearest_person, distance = self._find_nearest_person(obj, persons)
        obj.nearest_person_distance = distance
        
        if nearest_person and distance < self.config['spatial_thresholds']['owner_proximity']:
            state['owner_last_seen'] = now
        
        # Déterminer le statut
        obj.status, obj.status_reason = self._determine_status(obj, state)
        
        # Mettre à jour l'historique
        state['status_history'].append((now, obj.status))
        
        return obj
    
    def _detect_movement(self, position_history: deque) -> bool:
        """Détecte si l'objet a bougé récemment"""
        if len(position_history) < 2:
            return False
        
        threshold = self.config['spatial_thresholds']['movement_threshold']
        recent_positions = list(position_history)[-5:]  # 5 dernières positions
        
        for i in range(1, len(recent_positions)):
            _, pos1 = recent_positions[i-1]
            _, pos2 = recent_positions[i]
            
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            if distance > threshold:
                return True
        
        return False
    
    def _find_nearest_person(self, obj: ObjectDetection, 
                           persons: List[PersonDetection]) -> Tuple[Optional[PersonDetection], float]:
        """Trouve la personne la plus proche de l'objet"""
        if not persons:
            return None, float('inf')
        
        obj_center = (obj.bounding_box.x + obj.bounding_box.width/2,
                     obj.bounding_box.y + obj.bounding_box.height/2)
        
        nearest_person = None
        min_distance = float('inf')
        
        for person in persons:
            person_center = (person.bounding_box.x + person.bounding_box.width/2,
                           person.bounding_box.y + person.bounding_box.height/2)
            
            distance = np.sqrt((obj_center[0] - person_center[0])**2 + 
                             (obj_center[1] - person_center[1])**2)
            
            # Conversion pixels -> mètres (approximatif)
            distance_meters = distance * 0.01  # À ajuster selon la caméra
            
            if distance_meters < min_distance:
                min_distance = distance_meters
                nearest_person = person
        
        return nearest_person, min_distance
    
    def _determine_status(self, obj: ObjectDetection, state: dict) -> Tuple[ObjectStatus, str]:
        """Détermine le statut d'un objet"""
        now = datetime.now()
        time_stationary = obj.duration_stationary
        time_without_owner = (now - state['owner_last_seen']).total_seconds()
        
        thresholds = self.config['temporal_thresholds']
        
        # Objet critique (très longue durée)
        if time_stationary > thresholds['critical']:
            return ObjectStatus.CRITICAL, f"Abandonné depuis {time_stationary//60:.0f} minutes"
        
        # Objet perdu confirmé
        if time_stationary > thresholds['alert'] and time_without_owner > thresholds['alert']:
            return ObjectStatus.LOST, f"Perdu - aucun propriétaire depuis {time_without_owner//60:.0f} minutes"
        
        # Objet suspect
        if time_stationary > thresholds['surveillance']:
            if time_without_owner > thresholds['surveillance']:
                return ObjectStatus.SUSPECT, f"Suspect - immobile depuis {time_stationary:.0f}s"
            else:
                return ObjectStatus.SURVEILLANCE, f"Sous surveillance - propriétaire proche"
        
        # Objet normal
        return ObjectStatus.NORMAL, "Objet normal avec propriétaire"
    
    def _should_generate_alert(self, obj: ObjectDetection, previous_status: ObjectStatus) -> bool:
        """Détermine s'il faut générer une alerte"""
        # Alerte seulement pour les transitions vers des états critiques
        if previous_status != obj.status:
            if obj.status in [ObjectStatus.LOST, ObjectStatus.CRITICAL]:
                return True
            
            # Alerte pour objets prioritaires devenus suspects
            if (obj.status == ObjectStatus.SUSPECT and 
                obj.class_name in self.config['priority_objects']):
                return True
        
        return False
    
    def _create_alert(self, obj: ObjectDetection) -> LostObjectAlert:
        """Crée une alerte pour un objet perdu"""
        alert_level = "CRITICAL" if obj.status == ObjectStatus.CRITICAL else "WARNING"
        
        message = f"🚨 {obj.class_name_fr} {obj.status.value} détecté"
        if obj.duration_stationary > 60:
            message += f" (immobile depuis {obj.duration_stationary//60:.0f} min)"
        
        actions = self._get_recommended_actions(obj)
        
        return LostObjectAlert(
            alert_id=str(uuid.uuid4()),
            object_detection=obj,
            alert_level=alert_level,
            message=message,
            recommended_actions=actions,
            created_at=datetime.now()
        )
    
    def _get_recommended_actions(self, obj: ObjectDetection) -> List[str]:
        """Retourne les actions recommandées"""
        actions = []
        
        if obj.status == ObjectStatus.SUSPECT:
            actions.extend([
                "🔍 Vérifier si le propriétaire est à proximité",
                "📱 Annoncer l'objet trouvé dans la zone",
                "👀 Surveiller pendant 5 minutes supplémentaires"
            ])
        
        elif obj.status == ObjectStatus.LOST:
            actions.extend([
                "🚨 Alerter l'équipe de sécurité",
                "📢 Faire une annonce générale",
                "📝 Enregistrer l'objet trouvé",
                "📍 Sécuriser la zone autour de l'objet"
            ])
        
        elif obj.status == ObjectStatus.CRITICAL:
            actions.extend([
                "🚑 Intervention prioritaire requise",
                "🔒 Récupérer l'objet immédiatement",
                "📞 Contacter les autorités si nécessaire",
                "📋 Procédure d'objet abandonné"
            ])
        
        return actions

class ObjectDetector:
    """Détecteur d'objets principal"""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.image_processor = ImageProcessor()
        self.tracker = ObjectTracker()
        self.lost_detector = LostObjectDetector()
        self.config = MODEL_CONFIG
        
        logger.info(f"Détecteur initialisé sur {device}")
    
    def detect(self, image: np.ndarray, **kwargs) -> Tuple[List[ObjectDetection], List[PersonDetection], List[LostObjectAlert]]:
        """
        Détecte les objets dans une image
        
        Args:
            image: Image numpy array
            **kwargs: Paramètres de détection
            
        Returns:
            Tuple (objets, personnes, alertes)
        """
        # Paramètres
        confidence_threshold = kwargs.get('confidence_threshold', self.config['confidence_threshold'])
        nms_threshold = kwargs.get('nms_threshold', self.config['nms_threshold'])
        max_detections = kwargs.get('max_detections', self.config['max_detections'])
        enable_tracking = kwargs.get('enable_tracking', True)
        enable_lost_detection = kwargs.get('enable_lost_detection', True)
        
        # Prétraitement
        tensor, transform_info = self.image_processor.preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # Inférence
        with torch.no_grad():
            predictions = self.model(tensor)
        
        # Post-traitement
        detections = self._postprocess_predictions(predictions, transform_info, image.shape[:2])
        
        # Filtrage
        detections = filter_by_confidence(detections, confidence_threshold)
        detections = nms_by_class(detections, nms_threshold)
        detections = detections[:max_detections]
        
        # Conversion en objets métier
        objects, persons = self._convert_to_business_objects(detections)
        
        # Suivi des objets
        if enable_tracking:
            objects = self.tracker.update(objects)
        
        # Détection d'objets perdus
        alerts = []
        if enable_lost_detection:
            objects, alerts = self.lost_detector.analyze_objects(objects, persons)
        
        return objects, persons, alerts
    
    def _postprocess_predictions(self, predictions, transform_info: dict, 
                               original_shape: Tuple[int, int]) -> np.ndarray:
        """Post-traite les prédictions du modèle"""
        # Cette fonction dépend de l'architecture exacte du modèle
        # Ici on suppose un format YOLO-like
        
        # Exemple de décodage (à adapter selon votre modèle)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]  # Premier niveau de prédiction
        
        # Conversion en format détection [x1, y1, x2, y2, conf, class]
        detections = self._decode_predictions(predictions)
        
        # Transformation inverse vers l'espace original
        detections = self.image_processor.postprocess_detections(detections, transform_info)
        
        # Clipper aux limites de l'image
        detections = clip_boxes(detections, original_shape)
        
        return detections
    
    def _decode_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Décode les prédictions brutes du modèle"""
        # Cette fonction dépend de votre architecture de modèle
        # Voici un exemple générique
        
        if predictions.dim() == 4:  # [batch, anchors, grid_h, grid_w]
            predictions = predictions.squeeze(0)  # Enlever batch dimension
        
        # Exemple de décodage simple (à adapter)
        # Supposons que predictions contient [x, y, w, h, conf, class_probs...]
        
        # Extraction des coordonnées et confiances
        boxes = predictions[..., :4]  # x, y, w, h
        confidences = predictions[..., 4]  # objectness
        class_probs = predictions[..., 5:]  # probabilités des classes
        
        # Conversion center -> corner
        from app.utils.box_utils import center_to_corner
        boxes_corner = center_to_corner(boxes.view(-1, 4))
        
        # Calcul des confiances finales
        class_confidences = confidences.unsqueeze(-1) * class_probs
        class_ids = torch.argmax(class_confidences, dim=-1)
        max_confidences = torch.max(class_confidences, dim=-1)[0]
        
        # Assemblage final
        detections = torch.cat([
            boxes_corner,
            max_confidences.unsqueeze(-1),
            class_ids.unsqueeze(-1).float()
        ], dim=-1)
        
        return detections
    
    def _convert_to_business_objects(self, detections: np.ndarray) -> Tuple[List[ObjectDetection], List[PersonDetection]]:
        """Convertit les détections en objets métier"""
        objects = []
        persons = []
        now = datetime.now()
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_name = self.config['classes'][int(class_id)]
            class_name_fr = self.config['class_names_fr'].get(class_name, class_name)
            
            # Déterminer le niveau de confiance
            if conf >= 0.9:
                conf_level = DetectionConfidence.VERY_HIGH
            elif conf >= 0.7:
                conf_level = DetectionConfidence.HIGH
            elif conf >= 0.5:
                conf_level = DetectionConfidence.MEDIUM
            else:
                conf_level = DetectionConfidence.LOW
            
            # Créer la bounding box
            bbox = BoundingBox(
                x=float(x1),
                y=float(y1),
                width=float(x2 - x1),
                height=float(y2 - y1)
            )
            
            if class_name == 'person':
                # Détection de personne
                person = PersonDetection(
                    person_id=str(uuid.uuid4()),
                    confidence=float(conf),
                    bounding_box=bbox,
                    position=(float(x1 + (x2-x1)/2), float(y1 + (y2-y1)/2))
                )
                persons.append(person)
            
            else:
                # Détection d'objet
                obj = ObjectDetection(
                    object_id=str(uuid.uuid4()),
                    class_name=class_name,
                    class_name_fr=class_name_fr,
                    confidence=float(conf),
                    confidence_level=conf_level,
                    bounding_box=bbox,
                    first_seen=now,
                    last_seen=now
                )
                objects.append(obj)
        
        return objects, persons