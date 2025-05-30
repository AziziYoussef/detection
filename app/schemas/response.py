"""
Schémas Pydantic pour les réponses API du service de détection d'objets perdus
Définit tous les modèles de réponse standardisés de l'API REST
"""
from typing import List, Optional, Dict, Any, Union, Generic, TypeVar
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel
from enum import Enum
from datetime import datetime
import uuid

# Import des schémas de détection
from .detection import (
    Detection, DetectionResult, TrackingResult, MatchingResult,
    LostObject, SearchRequest, ObjectTrack, SimilarityResult,
    DetectionStats, LostObjectAlert, ModelInfo
)

# === ENUMS DE RÉPONSE ===

class ResponseStatus(str, Enum):
    """Statut des réponses API"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"

class ErrorCode(str, Enum):
    """Codes d'erreur standardisés"""
    # Erreurs générales
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    
    # Erreurs de fichiers
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    
    # Erreurs de traitement
    DETECTION_FAILED = "DETECTION_FAILED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    
    # Erreurs de matching
    MATCHING_FAILED = "MATCHING_FAILED"
    NO_FEATURES_FOUND = "NO_FEATURES_FOUND"
    SIMILARITY_TOO_LOW = "SIMILARITY_TOO_LOW"
    
    # Erreurs métier
    OBJECT_NOT_FOUND = "OBJECT_NOT_FOUND"
    SEARCH_EXPIRED = "SEARCH_EXPIRED"
    INVALID_CONTACT_INFO = "INVALID_CONTACT_INFO"

class MessageType(str, Enum):
    """Types de messages WebSocket"""
    DETECTION = "detection"
    TRACKING = "tracking"
    LOST_OBJECT = "lost_object"
    ALERT = "alert"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

# === MODÈLES DE RÉPONSE DE BASE ===

DataType = TypeVar('DataType')

class BaseResponse(GenericModel, Generic[DataType]):
    """Réponse de base générique"""
    success: bool = Field(..., description="Indique si la requête a réussi")
    message: str = Field(..., description="Message descriptif")
    data: Optional[DataType] = Field(None, description="Données de la réponse")
    timestamp: float = Field(..., description="Timestamp de la réponse")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unique de la requête")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Opération réalisée avec succès",
                "data": {},
                "timestamp": 1640995200.0,
                "request_id": "req_abc123def456"
            }
        }

class SuccessResponse(BaseResponse[DataType]):
    """Réponse de succès"""
    success: bool = Field(True, description="Toujours True pour une réponse de succès")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="Statut de la réponse")

class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée"""
    success: bool = Field(False, description="Toujours False pour une erreur")
    error_code: ErrorCode = Field(..., description="Code d'erreur standardisé")
    message: str = Field(..., description="Message d'erreur")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails supplémentaires")
    timestamp: float = Field(..., description="Timestamp de l'erreur")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
class PartialResponse(BaseResponse[DataType]):
    """Réponse partielle (succès avec avertissements)"""
    success: bool = Field(True, description="Succès partiel")
    status: ResponseStatus = Field(ResponseStatus.PARTIAL)
    warnings: List[str] = Field([], description="Liste des avertissements")

# === MODÈLES AVEC MÉTADONNÉES ===

class ResponseMetadata(BaseModel):
    """Métadonnées de réponse"""
    processing_time: float = Field(..., ge=0, description="Temps de traitement (secondes)")
    model_used: Optional[str] = Field(None, description="Modèle utilisé")
    version: str = Field("1.0.0", description="Version de l'API")
    environment: Optional[str] = Field(None, description="Environnement d'exécution")
    server_id: Optional[str] = Field(None, description="ID du serveur")

class MetadataResponse(BaseResponse[DataType]):
    """Réponse avec métadonnées"""
    metadata: ResponseMetadata = Field(..., description="Métadonnées de traitement")

# === RÉPONSES DE DÉTECTION ===

class DetectionResponseData(BaseModel):
    """Données de réponse pour la détection"""
    detections: List[Detection] = Field([], description="Liste des objets détectés")
    detection_count: int = Field(..., ge=0, description="Nombre d'objets détectés")
    image_size: Optional[tuple] = Field(None, description="Taille de l'image analysée")
    annotated_image_url: Optional[str] = Field(None, description="URL de l'image annotée")
    processing_stats: Optional[Dict[str, float]] = Field(None, description="Statistiques de traitement")

class DetectionResponse(MetadataResponse[DetectionResponseData]):
    """Réponse pour la détection d'objets"""
    data: DetectionResponseData = Field(..., description="Données de détection")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Détection réalisée avec succès",
                "data": {
                    "detections": [
                        {
                            "bbox": {"x": 100, "y": 150, "width": 200, "height": 180},
                            "confidence": 0.85,
                            "class_id": 0,
                            "class_name": "valise",
                            "track_id": 42
                        }
                    ],
                    "detection_count": 1,
                    "image_size": [640, 480],
                    "annotated_image_url": "/api/v1/images/annotated_123.jpg"
                },
                "metadata": {
                    "processing_time": 0.156,
                    "model_used": "stable_model"
                },
                "timestamp": 1640995200.0
            }
        }

class VideoDetectionResponseData(BaseModel):
    """Données de réponse pour la détection vidéo"""
    total_frames: int = Field(..., ge=0, description="Nombre total de frames")
    processed_frames: int = Field(..., ge=0, description="Frames traitées")
    detections_by_frame: Dict[int, List[Detection]] = Field({}, description="Détections par frame")
    tracking_results: Optional[List[TrackingResult]] = Field(None, description="Résultats de tracking")
    lost_objects: List[ObjectTrack] = Field([], description="Objets perdus détectés")
    output_video_url: Optional[str] = Field(None, description="URL de la vidéo annotée")
    summary: DetectionStats = Field(..., description="Statistiques globales")

class VideoDetectionResponse(MetadataResponse[VideoDetectionResponseData]):
    """Réponse pour la détection vidéo"""
    data: VideoDetectionResponseData = Field(..., description="Données de détection vidéo")

class StreamDetectionResponseData(BaseModel):
    """Données de réponse pour le streaming"""
    stream_id: str = Field(..., description="ID unique du stream")
    status: str = Field(..., description="Statut du stream")
    frames_processed: int = Field(0, ge=0, description="Frames traitées")
    active_tracks: int = Field(0, ge=0, description="Tracks actifs")
    lost_objects_count: int = Field(0, ge=0, description="Objets perdus")
    websocket_url: str = Field(..., description="URL WebSocket pour temps réel")

class StreamDetectionResponse(SuccessResponse[StreamDetectionResponseData]):
    """Réponse pour le streaming en temps réel"""
    data: StreamDetectionResponseData = Field(..., description="Données de streaming")

class BatchDetectionResponseData(BaseModel):
    """Données de réponse pour le traitement en lot"""
    total_images: int = Field(..., ge=0, description="Nombre total d'images")
    processed_images: int = Field(..., ge=0, description="Images traitées")
    failed_images: int = Field(0, ge=0, description="Images en échec")
    results: List[DetectionResult] = Field([], description="Résultats par image")
    batch_summary: DetectionStats = Field(..., description="Statistiques du lot")
    errors: List[Dict[str, Any]] = Field([], description="Erreurs rencontrées")

class BatchDetectionResponse(MetadataResponse[BatchDetectionResponseData]):
    """Réponse pour la détection en lot"""
    data: BatchDetectionResponseData = Field(..., description="Données de traitement en lot")

# === RÉPONSES DE MATCHING ===

class MatchingResponseData(BaseModel):
    """Données de réponse pour le matching"""
    matching_result: MatchingResult = Field(..., description="Résultat du matching")
    is_match: bool = Field(..., description="True si similarité suffisante")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Score de similarité")
    matches_visualization_url: Optional[str] = Field(None, description="Visualisation des matches")

class MatchingResponse(MetadataResponse[MatchingResponseData]):
    """Réponse pour le matching photo-photo"""
    data: MatchingResponseData = Field(..., description="Données de matching")

class SimilaritySearchResponseData(BaseModel):
    """Données de réponse pour la recherche de similarité"""
    query_image_id: str = Field(..., description="ID de l'image de recherche")
    matches: List[Dict[str, Any]] = Field([], description="Objets similaires trouvés")
    total_matches: int = Field(..., ge=0, description="Nombre total de matches")
    search_time: float = Field(..., ge=0, description="Temps de recherche")
    similarity_threshold: float = Field(..., description="Seuil utilisé")

class SimilaritySearchResponse(SuccessResponse[SimilaritySearchResponseData]):
    """Réponse pour la recherche de similarité"""
    data: SimilaritySearchResponseData = Field(..., description="Résultats de recherche")

# === RÉPONSES MÉTIER ===

class LostObjectResponseData(BaseModel):
    """Données de réponse pour un objet perdu"""
    lost_object: LostObject = Field(..., description="Données de l'objet perdu")
    matching_suggestions: List[Dict[str, Any]] = Field([], description="Suggestions de matching")
    status_history: List[Dict[str, Any]] = Field([], description="Historique des statuts")

class LostObjectResponse(SuccessResponse[LostObjectResponseData]):
    """Réponse pour un objet perdu"""
    data: LostObjectResponseData = Field(..., description="Données de l'objet perdu")

class LostObjectListResponseData(BaseModel):
    """Données de réponse pour une liste d'objets perdus"""
    objects: List[LostObject] = Field([], description="Liste des objets perdus")
    total_count: int = Field(..., ge=0, description="Nombre total d'objets")
    page: int = Field(1, ge=1, description="Page actuelle")
    page_size: int = Field(20, ge=1, description="Taille de la page")
    total_pages: int = Field(..., ge=0, description="Nombre total de pages")
    filters_applied: Dict[str, Any] = Field({}, description="Filtres appliqués")

class LostObjectListResponse(SuccessResponse[LostObjectListResponseData]):
    """Réponse pour une liste d'objets perdus"""
    data: LostObjectListResponseData = Field(..., description="Liste des objets perdus")

class SearchRequestResponseData(BaseModel):
    """Données de réponse pour une demande de recherche"""
    search_request: SearchRequest = Field(..., description="Demande de recherche")
    initial_matches: List[str] = Field([], description="Matches initiaux trouvés")
    estimated_processing_time: Optional[float] = Field(None, description="Temps estimé")

class SearchRequestResponse(SuccessResponse[SearchRequestResponseData]):
    """Réponse pour une demande de recherche"""
    data: SearchRequestResponseData = Field(..., description="Données de la demande")

class SearchResultsResponseData(BaseModel):
    """Données de réponse pour les résultats de recherche"""
    search_id: str = Field(..., description="ID de la recherche")
    matches_found: List[Dict[str, Any]] = Field([], description="Objets correspondants")
    confidence_scores: List[float] = Field([], description="Scores de confiance")
    total_matches: int = Field(..., ge=0, description="Nombre de matches")
    search_completed: bool = Field(..., description="Recherche terminée")
    next_actions: List[str] = Field([], description="Actions recommandées")

class SearchResultsResponse(SuccessResponse[SearchResultsResponseData]):
    """Réponse pour les résultats de recherche"""
    data: SearchResultsResponseData = Field(..., description="Résultats de recherche")

# === RÉPONSES SYSTÈME ===

class HealthResponseData(BaseModel):
    """Données de réponse pour le health check"""
    status: str = Field(..., description="Statut général (healthy/degraded/unhealthy)")
    version: str = Field(..., description="Version du service")
    uptime: float = Field(..., ge=0, description="Temps de fonctionnement (secondes)")
    environment: str = Field(..., description="Environnement d'exécution")
    database_status: str = Field("unknown", description="Statut de la base de données")
    models_status: Dict[str, str] = Field({}, description="Statut des modèles")
    system_resources: Dict[str, float] = Field({}, description="Ressources système")
    last_check: datetime = Field(..., description="Dernière vérification")

class HealthResponse(BaseResponse[HealthResponseData]):
    """Réponse pour le health check"""
    data: HealthResponseData = Field(..., description="Données de santé")

class MetricsResponseData(BaseModel):
    """Données de réponse pour les métriques"""
    requests_total: int = Field(..., ge=0, description="Nombre total de requêtes")
    requests_per_second: float = Field(..., ge=0, description="Requêtes par seconde")
    average_response_time: float = Field(..., ge=0, description="Temps de réponse moyen")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Taux d'erreur")
    detection_stats: DetectionStats = Field(..., description="Statistiques de détection")
    model_performance: Dict[str, Dict[str, float]] = Field({}, description="Performance des modèles")
    system_metrics: Dict[str, float] = Field({}, description="Métriques système")
    time_range: Dict[str, datetime] = Field({}, description="Période des métriques")

class MetricsResponse(SuccessResponse[MetricsResponseData]):
    """Réponse pour les métriques"""
    data: MetricsResponseData = Field(..., description="Données de métriques")

class ModelStatusResponseData(BaseModel):
    """Données de réponse pour le statut des modèles"""
    available_models: List[ModelInfo] = Field([], description="Modèles disponibles")
    active_model: str = Field(..., description="Modèle actif")
    model_performance: Dict[str, Dict[str, float]] = Field({}, description="Performance des modèles")
    loading_status: Dict[str, str] = Field({}, description="Statut de chargement")
    last_updated: datetime = Field(..., description="Dernière mise à jour")

class ModelStatusResponse(SuccessResponse[ModelStatusResponseData]):
    """Réponse pour le statut des modèles"""
    data: ModelStatusResponseData = Field(..., description="Statut des modèles")

# === RÉPONSES WEBSOCKET ===

class WebSocketMessage(BaseModel):
    """Message WebSocket générique"""
    type: MessageType = Field(..., description="Type de message")
    payload: Dict[str, Any] = Field(..., description="Données du message")
    timestamp: float = Field(..., description="Timestamp du message")
    session_id: str = Field(..., description="ID de la session WebSocket")

class StreamFrame(BaseModel):
    """Frame de streaming temps réel"""
    frame_id: int = Field(..., ge=0, description="ID de la frame")
    timestamp: float = Field(..., description="Timestamp de la frame")
    detections: List[Detection] = Field([], description="Détections dans cette frame")
    tracking_results: List[ObjectTrack] = Field([], description="Résultats de tracking")
    frame_url: Optional[str] = Field(None, description="URL de la frame annotée")
    processing_time: float = Field(..., ge=0, description="Temps de traitement")

class DetectionEvent(BaseModel):
    """Événement de détection"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Type d'événement")
    detection: Detection = Field(..., description="Détection concernée")
    track_info: Optional[ObjectTrack] = Field(None, description="Info de tracking")
    alert_level: str = Field("info", description="Niveau d'alerte")
    location: Optional[str] = Field(None, description="Localisation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    auto_generated: bool = Field(True, description="Événement auto-généré")

class LostObjectEventData(BaseModel):
    """Données d'événement objet perdu"""
    alert: LostObjectAlert = Field(..., description="Alerte objet perdu")
    recommended_actions: List[str] = Field([], description="Actions recommandées")
    similar_objects: List[str] = Field([], description="Objets similaires dans la base")

class LostObjectEvent(WebSocketMessage):
    """Événement WebSocket pour objet perdu"""
    type: MessageType = Field(MessageType.LOST_OBJECT, description="Type d'événement")
    payload: LostObjectEventData = Field(..., description="Données de l'alerte")

# === RÉPONSES DE PAGINATION ===

class PaginationInfo(BaseModel):
    """Informations de pagination"""
    page: int = Field(1, ge=1, description="Page actuelle")
    page_size: int = Field(20, ge=1, le=100, description="Taille de la page")
    total_items: int = Field(..., ge=0, description="Nombre total d'éléments")
    total_pages: int = Field(..., ge=0, description="Nombre total de pages")
    has_next: bool = Field(..., description="Page suivante disponible")
    has_previous: bool = Field(..., description="Page précédente disponible")
    
    @validator('total_pages', pre=True, always=True)
    def calculate_total_pages(cls, v, values):
        total_items = values.get('total_items', 0)
        page_size = values.get('page_size', 20)
        return max(1, (total_items + page_size - 1) // page_size)

class PaginatedResponse(BaseResponse[DataType]):
    """Réponse paginée"""
    data: DataType = Field(..., description="Données de la page")
    pagination: PaginationInfo = Field(..., description="Informations de pagination")

# === RÉPONSES D'UPLOAD ===

class UploadResponseData(BaseModel):
    """Données de réponse pour un upload"""
    file_id: str = Field(..., description="ID unique du fichier")
    filename: str = Field(..., description="Nom du fichier")
    file_size: int = Field(..., ge=0, description="Taille du fichier (bytes)")
    content_type: str = Field(..., description="Type MIME")
    upload_url: str = Field(..., description="URL d'accès au fichier")
    processing_status: str = Field("pending", description="Statut de traitement")
    expires_at: Optional[datetime] = Field(None, description="Date d'expiration")

class UploadResponse(SuccessResponse[UploadResponseData]):
    """Réponse pour un upload de fichier"""
    data: UploadResponseData = Field(..., description="Informations du fichier uploadé")

# === UTILITAIRES DE CRÉATION DE RÉPONSES ===

def create_success_response(
    data: Any = None,
    message: str = "Opération réalisée avec succès",
    metadata: Optional[ResponseMetadata] = None
) -> Dict[str, Any]:
    """Crée une réponse de succès standardisée"""
    response = {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().timestamp(),
        "request_id": str(uuid.uuid4())
    }
    
    if metadata:
        response["metadata"] = metadata.dict()
    
    return response

def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crée une réponse d'erreur standardisée"""
    return {
        "success": False,
        "error_code": error_code.value,
        "message": message,
        "details": details or {},
        "timestamp": datetime.utcnow().timestamp(),
        "request_id": str(uuid.uuid4())
    }

def create_detection_response(
    detections: List[Detection],
    processing_time: float,
    model_name: str,
    image_size: Optional[tuple] = None,
    annotated_image_url: Optional[str] = None
) -> DetectionResponse:
    """Crée une réponse de détection"""
    data = DetectionResponseData(
        detections=detections,
        detection_count=len(detections),
        image_size=image_size,
        annotated_image_url=annotated_image_url
    )
    
    metadata = ResponseMetadata(
        processing_time=processing_time,
        model_used=model_name
    )
    
    return DetectionResponse(
        success=True,
        message=f"{len(detections)} objet(s) détecté(s)" if detections else "Aucun objet détecté",
        data=data,
        metadata=metadata,
        timestamp=datetime.utcnow().timestamp()
    )

# === VALIDATION DES SCHÉMAS DE RÉPONSE ===

def validate_response_schemas():
    """Valide tous les schémas de réponse"""
    try:
        # Test BaseResponse
        response = create_success_response({"test": "data"})
        assert response["success"] is True
        
        # Test ErrorResponse
        error = create_error_response(
            ErrorCode.DETECTION_FAILED,
            "Test error"
        )
        assert error["success"] is False
        
        # Test DetectionResponse
        from .detection import Detection, BoundingBox, ObjectClass
        
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        detection = Detection(
            bbox=bbox,
            confidence=0.85,
            class_id=0,
            class_name=ObjectClass.VALISE
        )
        
        det_response = create_detection_response(
            detections=[detection],
            processing_time=0.156,
            model_name="stable_model"
        )
        assert det_response.data.detection_count == 1
        
        return True
        
    except Exception as e:
        print(f"Erreur validation réponses: {e}")
        return False