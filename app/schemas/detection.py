"""
Schémas Pydantic pour la détection d'objets perdus
Définit tous les modèles de données utilisés pour la détection, le tracking et le matching
"""
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
from datetime import datetime
import uuid
import re

# === ENUMS ===

class ObjectClass(str, Enum):
    """Classes d'objets détectés"""
    VALISE = "valise"
    SAC = "sac"
    TELEPHONE = "telephone"
    SAC_A_DOS = "sac_a_dos"
    PORTEFEUILLE = "portefeuille"
    CLES = "cles"

class DetectionStatus(str, Enum):
    """Statut de détection"""
    DETECTED = "detected"
    TRACKING = "tracking"
    LOST = "lost"
    FOUND = "found"
    EXPIRED = "expired"

class TrackingStatus(str, Enum):
    """Statut de tracking"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOST = "lost"
    TERMINATED = "terminated"

class MatchingMethod(str, Enum):
    """Méthodes de matching"""
    ORB = "orb"
    SIFT = "sift"
    CNN = "cnn"
    HYBRID = "hybrid"

class LostObjectStatus(str, Enum):
    """Statut des objets perdus"""
    REPORTED = "reported"
    SEARCHING = "searching"
    FOUND = "found"
    CLAIMED = "claimed"
    EXPIRED = "expired"

class SearchStatus(str, Enum):
    """Statut des recherches"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# === MODÈLES DE BASE ===

class BoundingBox(BaseModel):
    """Boîte englobante d'un objet détecté"""
    x: float = Field(..., ge=0, description="Position X (coin supérieur gauche)")
    y: float = Field(..., ge=0, description="Position Y (coin supérieur gauche)")
    width: float = Field(..., gt=0, description="Largeur de la boîte")
    height: float = Field(..., gt=0, description="Hauteur de la boîte")
    
    @property
    def x2(self) -> float:
        """Coordonnée X du coin inférieur droit"""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Coordonnée Y du coin inférieur droit"""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Centre de la boîte"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """Aire de la boîte"""
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calcule l'IoU avec une autre boîte"""
        # Intersection
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.area + other.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

class Detection(BaseModel):
    """Détection d'un objet"""
    bbox: BoundingBox = Field(..., description="Boîte englobante")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de détection")
    class_id: int = Field(..., ge=0, le=5, description="ID de la classe (0-5)")
    class_name: ObjectClass = Field(..., description="Nom de la classe")
    track_id: Optional[int] = Field(None, description="ID de tracking si disponible")
    timestamp: Optional[float] = Field(None, description="Timestamp de détection")
    features: Optional[List[float]] = Field(None, description="Features extraites pour matching")
    
    @validator('class_id', 'class_name')
    def validate_class_consistency(cls, v, values):
        """Valide la cohérence entre class_id et class_name"""
        class_mapping = {
            0: ObjectClass.VALISE,
            1: ObjectClass.SAC,
            2: ObjectClass.TELEPHONE,
            3: ObjectClass.SAC_A_DOS,
            4: ObjectClass.PORTEFEUILLE,
            5: ObjectClass.CLES
        }
        
        if 'class_id' in values and 'class_name' in values:
            expected_name = class_mapping.get(values['class_id'])
            if v != expected_name:
                raise ValueError(f"class_name '{v}' ne correspond pas à class_id {values['class_id']}")
        
        return v

class ObjectTrack(BaseModel):
    """Track d'un objet suivi"""
    track_id: int = Field(..., description="ID unique du track")
    detections: List[Detection] = Field(..., description="Historique des détections")
    first_seen: float = Field(..., description="Timestamp première détection")
    last_seen: float = Field(..., description="Timestamp dernière détection") 
    status: TrackingStatus = Field(TrackingStatus.ACTIVE, description="Statut du track")
    lost_duration: float = Field(0.0, description="Durée depuis dernière détection")
    confidence_avg: float = Field(..., ge=0.0, le=1.0, description="Confiance moyenne")
    
    @property
    def duration(self) -> float:
        """Durée totale du track"""
        return self.last_seen - self.first_seen
    
    @property
    def is_lost(self) -> bool:
        """True si l'objet est considéré comme perdu"""
        return self.status == TrackingStatus.LOST
    
    @property
    def current_detection(self) -> Optional[Detection]:
        """Détection la plus récente"""
        return self.detections[-1] if self.detections else None

# === MODÈLES DE RÉSULTATS ===

class DetectionResult(BaseModel):
    """Résultat d'une détection"""
    detections: List[Detection] = Field([], description="Liste des détections")
    image_size: Tuple[int, int] = Field(..., description="Taille de l'image (largeur, hauteur)")
    processing_time: float = Field(..., ge=0, description="Temps de traitement en secondes")
    model_name: str = Field(..., description="Nom du modèle utilisé")
    confidence_threshold: float = Field(..., description="Seuil de confiance utilisé")
    timestamp: float = Field(..., description="Timestamp du traitement")
    
    @property
    def detection_count(self) -> int:
        """Nombre de détections"""
        return len(self.detections)
    
    @property
    def has_detections(self) -> bool:
        """True si des objets ont été détectés"""
        return len(self.detections) > 0

class TrackingResult(BaseModel):
    """Résultat du tracking"""
    tracks: List[ObjectTrack] = Field([], description="Liste des tracks actifs")
    lost_tracks: List[ObjectTrack] = Field([], description="Liste des tracks perdus")
    frame_count: int = Field(..., ge=0, description="Numéro de frame")
    processing_time: float = Field(..., ge=0, description="Temps de traitement")
    timestamp: float = Field(..., description="Timestamp du frame")
    
    @property
    def active_track_count(self) -> int:
        """Nombre de tracks actifs"""
        return len([t for t in self.tracks if t.status == TrackingStatus.ACTIVE])
    
    @property
    def lost_object_count(self) -> int:
        """Nombre d'objets perdus"""
        return len(self.lost_tracks)

# === MODÈLES DE REQUÊTE ===

class ImageDetectionRequest(BaseModel):
    """Requête de détection sur image"""
    image_data: Optional[bytes] = Field(None, description="Données de l'image (base64)")
    image_url: Optional[str] = Field(None, description="URL de l'image")
    model_name: Optional[str] = Field(None, description="Modèle à utiliser")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_image: bool = Field(True, description="Retourner l'image annotée")
    extract_features: bool = Field(False, description="Extraire les features pour matching")
    
    @root_validator
    def validate_image_source(cls, values):
        """Valide qu'une source d'image est fournie"""
        if not values.get('image_data') and not values.get('image_url'):
            raise ValueError("image_data ou image_url doit être fourni")
        return values

class VideoDetectionRequest(BaseModel):
    """Requête de détection sur vidéo"""
    video_data: Optional[bytes] = Field(None, description="Données de la vidéo")
    video_url: Optional[str] = Field(None, description="URL de la vidéo")
    model_name: Optional[str] = Field(None, description="Modèle à utiliser")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_tracking: bool = Field(True, description="Activer le tracking")
    output_video: bool = Field(False, description="Générer vidéo annotée")
    frame_interval: int = Field(1, ge=1, description="Interval entre les frames")
    
    @root_validator
    def validate_video_source(cls, values):
        """Valide qu'une source vidéo est fournie"""
        if not values.get('video_data') and not values.get('video_url'):
            raise ValueError("video_data ou video_url doit être fourni")
        return values

class StreamDetectionRequest(BaseModel):
    """Requête de détection en streaming"""
    stream_url: str = Field(..., description="URL du stream")
    model_name: Optional[str] = Field(None, description="Modèle à utiliser")
    enable_tracking: bool = Field(True, description="Activer le tracking")
    lost_threshold: int = Field(30, ge=1, description="Seuils objets perdus (secondes)")
    alert_lost_objects: bool = Field(True, description="Alerter objets perdus")
    
class BatchDetectionRequest(BaseModel):
    """Requête de détection en lot"""
    image_urls: List[str] = Field(..., min_items=1, max_items=50)
    model_name: Optional[str] = Field(None, description="Modèle à utiliser")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    parallel_processing: bool = Field(True, description="Traitement parallèle")

# === MODÈLES DE CONFIGURATION ===

class DetectionConfig(BaseModel):
    """Configuration de détection"""
    model_name: str = Field(..., description="Nom du modèle")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.4, ge=0.0, le=1.0)
    max_detections: int = Field(100, ge=1, le=1000)
    input_size: Tuple[int, int] = Field((640, 640))
    device: str = Field("auto", description="Device de calcul")

class TrackingConfig(BaseModel):
    """Configuration de tracking"""
    track_threshold: float = Field(0.5, ge=0.0, le=1.0)
    match_threshold: float = Field(0.8, ge=0.0, le=1.0)
    lost_time_threshold: int = Field(30, ge=1, description="Temps avant objet perdu (s)")
    max_disappeared: int = Field(50, ge=1, description="Frames max sans détection")
    track_buffer: int = Field(30, ge=1, description="Buffer pour tracking")

class ModelInfo(BaseModel):
    """Informations sur un modèle"""
    name: str = Field(..., description="Nom du modèle")
    type: str = Field(..., description="Type de modèle")
    version: str = Field(..., description="Version du modèle")
    classes: List[str] = Field(..., description="Classes supportées")
    input_size: Tuple[int, int] = Field(..., description="Taille d'entrée")
    accuracy: Optional[float] = Field(None, description="Précision du modèle")
    speed: Optional[float] = Field(None, description="Vitesse (FPS)")
    file_size: Optional[int] = Field(None, description="Taille du fichier (bytes)")

# === MODÈLES DE MATCHING ===

class MatchingFeatures(BaseModel):
    """Features extraites pour le matching"""
    keypoints: List[Tuple[float, float]] = Field([], description="Points clés détectés")
    descriptors: List[List[float]] = Field([], description="Descripteurs des keypoints")
    features_vector: Optional[List[float]] = Field(None, description="Vecteur de features CNN")
    extraction_method: MatchingMethod = Field(..., description="Méthode d'extraction")
    extraction_time: float = Field(..., ge=0, description="Temps d'extraction")

class SimilarityResult(BaseModel):
    """Résultat de calcul de similarité"""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Score de similarité")
    distance: float = Field(..., ge=0.0, description="Distance entre features")
    matches_count: int = Field(..., ge=0, description="Nombre de matches trouvés")
    is_match: bool = Field(..., description="True si similarité suffisante")
    method_used: MatchingMethod = Field(..., description="Méthode utilisée")
    processing_time: float = Field(..., ge=0, description="Temps de calcul")

class MatchingRequest(BaseModel):
    """Requête de matching entre deux images"""
    image1_data: Optional[bytes] = Field(None, description="Première image")
    image1_url: Optional[str] = Field(None, description="URL première image")
    image2_data: Optional[bytes] = Field(None, description="Deuxième image")
    image2_url: Optional[str] = Field(None, description="URL deuxième image")
    method: MatchingMethod = Field(MatchingMethod.ORB, description="Méthode de matching")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    return_matches_image: bool = Field(False, description="Retourner image avec matches")
    
    @root_validator
    def validate_images(cls, values):
        """Valide que les deux images sont fournies"""
        img1_ok = values.get('image1_data') or values.get('image1_url')
        img2_ok = values.get('image2_data') or values.get('image2_url')
        
        if not img1_ok or not img2_ok:
            raise ValueError("Les deux images doivent être fournies")
        return values

class MatchingResult(BaseModel):
    """Résultat complet de matching"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    similarity: SimilarityResult = Field(..., description="Résultat de similarité")
    features1: MatchingFeatures = Field(..., description="Features image 1")
    features2: MatchingFeatures = Field(..., description="Features image 2")
    matches_image_url: Optional[str] = Field(None, description="URL image avec matches")
    total_processing_time: float = Field(..., ge=0, description="Temps total")
    timestamp: float = Field(..., description="Timestamp du matching")

# === MODÈLES MÉTIER ===

class ContactInfo(BaseModel):
    """Informations de contact"""
    email: Optional[str] = Field(None, description="Adresse email")
    phone: Optional[str] = Field(None, description="Numéro de téléphone")
    name: Optional[str] = Field(None, description="Nom de la personne")
    
    @validator('email')
    def validate_email(cls, v):
        if v and not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
            raise ValueError('Format email invalide')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^[\+]?[1-9][\d]{3,14}$', v):
            raise ValueError('Format téléphone invalide')
        return v

class LostObject(BaseModel):
    """Objet perdu dans le système"""
    id: str = Field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:8]}")
    object_class: ObjectClass = Field(..., description="Classe de l'objet")
    description: str = Field(..., min_length=10, max_length=500, description="Description détaillée")
    location: str = Field(..., min_length=5, max_length=200, description="Lieu de perte")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Date/heure de perte")
    image_url: Optional[str] = Field(None, description="URL de l'image de l'objet")
    detection_data: Optional[Detection] = Field(None, description="Données de détection")
    features: Optional[MatchingFeatures] = Field(None, description="Features pour matching")
    status: LostObjectStatus = Field(LostObjectStatus.REPORTED, description="Statut actuel")
    contact_info: ContactInfo = Field(..., description="Informations de contact")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées additionnelles")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LostObjectCreate(BaseModel):
    """Modèle pour créer un objet perdu"""
    object_class: ObjectClass = Field(..., description="Classe de l'objet")
    description: str = Field(..., min_length=10, max_length=500)
    location: str = Field(..., min_length=5, max_length=200)
    contact_info: ContactInfo = Field(..., description="Informations de contact")
    image_data: Optional[bytes] = Field(None, description="Image de l'objet")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LostObjectUpdate(BaseModel):
    """Modèle pour mettre à jour un objet perdu"""
    description: Optional[str] = Field(None, min_length=10, max_length=500)
    location: Optional[str] = Field(None, min_length=5, max_length=200)
    status: Optional[LostObjectStatus] = Field(None)
    contact_info: Optional[ContactInfo] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)

class SearchRequest(BaseModel):
    """Demande de recherche d'objet"""
    id: str = Field(default_factory=lambda: f"search_{uuid.uuid4().hex[:8]}")
    requester_email: str = Field(..., description="Email du demandeur")
    object_description: str = Field(..., min_length=10, max_length=500)
    search_image_url: Optional[str] = Field(None, description="Image de recherche")
    location_hint: Optional[str] = Field(None, description="Indice de localisation")
    contact_phone: Optional[str] = Field(None, description="Téléphone de contact")
    status: SearchStatus = Field(SearchStatus.PENDING, description="Statut de la recherche")
    matching_results: List[str] = Field([], description="IDs des objets matchés")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Date d'expiration")

class SearchRequestCreate(BaseModel):
    """Modèle pour créer une demande de recherche"""
    requester_email: str = Field(..., description="Email du demandeur")
    object_description: str = Field(..., min_length=10, max_length=500)
    search_image_data: Optional[bytes] = Field(None, description="Image de recherche")
    location_hint: Optional[str] = Field(None, max_length=200)
    contact_phone: Optional[str] = Field(None)
    
    @validator('requester_email')
    def validate_email(cls, v):
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
            raise ValueError('Format email invalide')
        return v

# === MODÈLES DE NOTIFICATION ===

class AlertInfo(BaseModel):
    """Information d'alerte"""
    alert_id: str = Field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    alert_type: str = Field(..., description="Type d'alerte")
    message: str = Field(..., description="Message d'alerte")
    severity: str = Field("info", description="Sévérité (info, warning, error)")
    object_id: Optional[str] = Field(None, description="ID de l'objet concerné")
    location: Optional[str] = Field(None, description="Localisation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(False, description="Alerte acquittée")

class LostObjectAlert(AlertInfo):
    """Alerte objet perdu"""
    alert_type: str = Field("lost_object", description="Type d'alerte")
    track_data: ObjectTrack = Field(..., description="Données du track perdu")
    lost_duration: float = Field(..., description="Durée depuis perte")
    confidence: float = Field(..., description="Confiance de la détection")

# === MODÈLES DE STATISTIQUES ===

class DetectionStats(BaseModel):
    """Statistiques de détection"""
    total_detections: int = Field(0, ge=0)
    detections_by_class: Dict[ObjectClass, int] = Field(default_factory=dict)
    average_confidence: float = Field(0.0, ge=0.0, le=1.0)
    processing_times: List[float] = Field(default_factory=list)
    period_start: datetime = Field(default_factory=datetime.utcnow)
    period_end: datetime = Field(default_factory=datetime.utcnow)

# === VALIDATION FINALE ===
def validate_detection_schemas():
    """Valide tous les schémas de détection"""
    try:
        # Test BoundingBox
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.area == 15000
        
        # Test Detection
        detection = Detection(
            bbox=bbox,
            confidence=0.85,
            class_id=0,
            class_name=ObjectClass.VALISE
        )
        assert detection.class_name == ObjectClass.VALISE
        
        # Test LostObject
        contact = ContactInfo(email="test@example.com", phone="+33123456789")
        lost_obj = LostObjectCreate(
            object_class=ObjectClass.SAC,
            description="Sac noir en cuir",
            location="Gare de Lyon",
            contact_info=contact
        )
        assert lost_obj.object_class == ObjectClass.SAC
        
        return True
        
    except Exception as e:
        print(f"Erreur validation: {e}")
        return False