# app/schemas/detection.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ObjectStatus(str, Enum):
    """États d'un objet détecté"""
    NORMAL = "normal"           # Objet avec propriétaire
    SURVEILLANCE = "surveillance"  # Sous surveillance
    SUSPECT = "suspect"         # Potentiellement perdu
    LOST = "lost"              # Confirmé perdu
    CRITICAL = "critical"       # Critique (longue durée)
    RESOLVED = "resolved"       # Résolu (récupéré)

class DetectionConfidence(str, Enum):
    """Niveaux de confiance"""
    LOW = "low"           # < 0.5
    MEDIUM = "medium"     # 0.5 - 0.7
    HIGH = "high"         # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9

class BoundingBox(BaseModel):
    """Boîte englobante d'un objet"""
    x: float = Field(..., description="Position X (top-left)")
    y: float = Field(..., description="Position Y (top-left)")
    width: float = Field(..., description="Largeur")
    height: float = Field(..., description="Hauteur")
    
    def area(self) -> float:
        """Calcule l'aire de la boîte"""
        return self.width * self.height
    
    def center(self) -> tuple:
        """Retourne le centre de la boîte"""
        return (self.x + self.width/2, self.y + self.height/2)

class ObjectDetection(BaseModel):
    """Détection d'un objet individuel"""
    object_id: str = Field(..., description="ID unique de l'objet")
    class_name: str = Field(..., description="Classe de l'objet")
    class_name_fr: str = Field(..., description="Nom français de la classe")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de détection")
    confidence_level: DetectionConfidence = Field(..., description="Niveau de confiance")
    bounding_box: BoundingBox = Field(..., description="Boîte englobante")
    
    # Métadonnées temporelles
    first_seen: datetime = Field(..., description="Première détection")
    last_seen: datetime = Field(..., description="Dernière détection")
    last_movement: Optional[datetime] = Field(None, description="Dernier mouvement détecté")
    duration_stationary: float = Field(0.0, description="Durée immobile (secondes)")
    
    # État de l'objet
    status: ObjectStatus = Field(ObjectStatus.NORMAL, description="État actuel")
    status_reason: str = Field("", description="Raison du statut")
    
    # Contexte spatial
    nearest_person_distance: Optional[float] = Field(None, description="Distance personne la plus proche")
    is_in_public_area: bool = Field(True, description="Dans une zone publique")
    zone_id: Optional[str] = Field(None, description="ID de la zone")
    
    # Tracking
    track_id: Optional[str] = Field(None, description="ID de suivi")
    track_confidence: float = Field(1.0, description="Confiance du suivi")

class PersonDetection(BaseModel):
    """Détection d'une personne"""
    person_id: str = Field(..., description="ID unique de la personne")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    position: tuple = Field(..., description="Position (x, y)")
    movement_vector: Optional[tuple] = Field(None, description="Vecteur de mouvement")
    is_stationary: bool = Field(False, description="Personne immobile")

class DetectionRequest(BaseModel):
    """Requête de détection"""
    model_name: Optional[str] = Field("default", description="Nom du modèle à utiliser")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    nms_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_detections: Optional[int] = Field(None, ge=1, le=100)
    enable_tracking: bool = Field(True, description="Activer le suivi")
    enable_lost_detection: bool = Field(True, description="Activer détection objets perdus")

class StreamConfig(BaseModel):
    """Configuration du streaming"""
    fps: int = Field(15, ge=1, le=30, description="Images par seconde")
    resolution: tuple = Field((640, 480), description="Résolution (width, height)")
    quality: int = Field(80, ge=10, le=100, description="Qualité JPEG")
    buffer_size: int = Field(30, description="Taille du buffer")

class DetectionResponse(BaseModel):
    """Réponse de détection"""
    success: bool = Field(..., description="Succès de l'opération")
    timestamp: datetime = Field(..., description="Timestamp de la détection")
    processing_time: float = Field(..., description="Temps de traitement (ms)")
    
    # Résultats
    objects: List[ObjectDetection] = Field(default_factory=list)
    persons: List[PersonDetection] = Field(default_factory=list)
    
    # Statistiques
    total_objects: int = Field(0, description="Nombre total d'objets")
    lost_objects: int = Field(0, description="Nombre d'objets perdus")
    suspect_objects: int = Field(0, description="Nombres d'objets suspects")
    
    # Métadonnées de l'image
    image_info: Dict[str, Any] = Field(default_factory=dict)
    model_used: str = Field("", description="Modèle utilisé")

class LostObjectAlert(BaseModel):
    """Alerte d'objet perdu"""
    alert_id: str = Field(..., description="ID unique de l'alerte")
    object_detection: ObjectDetection = Field(..., description="Objet perdu")
    alert_level: str = Field(..., description="Niveau d'alerte (WARNING, CRITICAL)")
    message: str = Field(..., description="Message d'alerte")
    recommended_actions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(..., description="Création de l'alerte")
    
class StreamFrame(BaseModel):
    """Frame de streaming"""
    frame_id: str = Field(..., description="ID unique du frame")
    timestamp: datetime = Field(..., description="Timestamp du frame")
    detection_result: DetectionResponse = Field(..., description="Résultat de détection")
    frame_data: Optional[str] = Field(None, description="Données image encodées base64")
    alerts: List[LostObjectAlert] = Field(default_factory=list)

class StreamStatus(BaseModel):
    """État du streaming"""
    client_id: str = Field(..., description="ID du client")
    is_active: bool = Field(..., description="Stream actif")
    fps: float = Field(..., description="FPS actuel")
    frames_processed: int = Field(..., description="Frames traités")
    alerts_generated: int = Field(..., description="Alertes générées")
    connected_since: datetime = Field(..., description="Connecté depuis")
    last_frame: Optional[datetime] = Field(None, description="Dernier frame")

class ModelInfo(BaseModel):
    """Informations sur un modèle"""
    name: str = Field(..., description="Nom du modèle")
    version: str = Field(..., description="Version")
    num_classes: int = Field(..., description="Nombre de classes")
    image_size: tuple = Field(..., description="Taille d'image")
    is_loaded: bool = Field(..., description="Modèle chargé")
    memory_usage: float = Field(..., description="Usage mémoire (MB)")
    performance_stats: Dict[str, float] = Field(default_factory=dict)

class HealthStatus(BaseModel):
    """État de santé du service"""
    status: str = Field(..., description="healthy, degraded, unhealthy")
    timestamp: datetime = Field(..., description="Timestamp du check")
    models_loaded: List[str] = Field(default_factory=list)
    gpu_available: bool = Field(..., description="GPU disponible")
    memory_usage: Dict[str, float] = Field(default_factory=dict)
    active_streams: int = Field(0, description="Streams actifs")
    errors: List[str] = Field(default_factory=list)

class ServiceStats(BaseModel):
    """Statistiques du service"""
    uptime: float = Field(..., description="Temps de fonctionnement (secondes)")
    total_detections: int = Field(0, description="Détections totales")
    total_alerts: int = Field(0, description="Alertes totales")
    average_processing_time: float = Field(0.0, description="Temps moyen (ms)")
    models_performance: Dict[str, Dict] = Field(default_factory=dict)
    resource_usage: Dict[str, float] = Field(default_factory=dict)