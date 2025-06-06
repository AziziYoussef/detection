"""
Schémas Pydantic centralisés pour le service de détection d'objets perdus
Ce module expose tous les schémas de données utilisés dans l'API
"""
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

# === IMPORTS DES SCHÉMAS DE DÉTECTION ===
try:
    from .detection import (
        # Modèles de base
        BoundingBox,
        Detection,
        DetectionResult,
        ObjectTrack,
        TrackingResult,
        
        # Modèles d'entrée
        ImageDetectionRequest,
        VideoDetectionRequest,
        StreamDetectionRequest,
        BatchDetectionRequest,
        
        # Modèles de configuration
        DetectionConfig,
        TrackingConfig,
        ModelInfo,
        
        # Modèles de matching
        MatchingRequest,
        MatchingFeatures,
        SimilarityResult,
        MatchingResult,
        
        # Modèles métier
        LostObject,
        LostObjectCreate,
        LostObjectUpdate,
        SearchRequest,
        SearchRequestCreate,
        
        # Enums
        ObjectClass,
        DetectionStatus,
        TrackingStatus,
        MatchingMethod,
        LostObjectStatus
    )
    logger.info("✅ Schémas de détection importés")
    
except ImportError as e:
    logger.error(f"❌ Erreur import detection.py: {e}")
    # Définitions minimales par défaut
    class BoundingBox: pass
    class Detection: pass
    class DetectionResult: pass
    ObjectClass = None
    DetectionStatus = None

# === IMPORTS DES SCHÉMAS DE RÉPONSE ===
try:
    from .response import (
        # Réponses de base
        BaseResponse,
        SuccessResponse,
        ErrorResponse,
        
        # Réponses de détection
        DetectionResponse,
        VideoDetectionResponse,
        StreamDetectionResponse,
        BatchDetectionResponse,
        
        # Réponses de matching
        MatchingResponse,
        SimilaritySearchResponse,
        
        # Réponses métier
        LostObjectResponse,
        LostObjectListResponse,
        SearchRequestResponse,
        SearchResultsResponse,
        
        # Réponses système
        HealthResponse,
        MetricsResponse,
        ModelStatusResponse,
        
        # Réponses WebSocket
        WebSocketMessage,
        StreamFrame,
        DetectionEvent,
        
        # Types de réponse
        ResponseStatus,
        ErrorCode,
        MessageType
    )
    logger.info("✅ Schémas de réponse importés")
    
except ImportError as e:
    logger.error(f"❌ Erreur import response.py: {e}")
    # Définitions minimales par défaut
    class BaseResponse: pass
    class SuccessResponse: pass
    class ErrorResponse: pass
    ResponseStatus = None
    ErrorCode = None

# === VALIDATION DES SCHÉMAS ===
def validate_schemas() -> bool:
    """
    Valide que tous les schémas sont correctement définis
    Returns:
        bool: True si validation OK, False sinon
    """
    errors = []
    
    # Validation schémas de détection
    required_detection_schemas = [
        'BoundingBox', 'Detection', 'DetectionResult',
        'LostObject', 'SearchRequest'
    ]
    
    for schema_name in required_detection_schemas:
        if schema_name not in globals() or globals()[schema_name] is None:
            errors.append(f"Schéma de détection manquant: {schema_name}")
    
    # Validation schémas de réponse
    required_response_schemas = [
        'BaseResponse', 'DetectionResponse', 'ErrorResponse'
    ]
    
    for schema_name in required_response_schemas:
        if schema_name not in globals() or globals()[schema_name] is None:
            errors.append(f"Schéma de réponse manquant: {schema_name}")
    
    if errors:
        logger.error("❌ Erreurs de validation des schémas:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✅ Validation des schémas réussie")
    return True

# === UTILITAIRES POUR LES SCHÉMAS ===
def create_detection_example() -> dict:
    """Crée un exemple de détection pour la documentation"""
    return {
        "bbox": {
            "x": 100,
            "y": 150,
            "width": 200,
            "height": 180
        },
        "confidence": 0.85,
        "class_id": 0,
        "class_name": "valise",
        "track_id": 42,
        "timestamp": 1640995200.0
    }

def create_lost_object_example() -> dict:
    """Crée un exemple d'objet perdu pour la documentation"""
    return {
        "id": "obj_123456",
        "object_class": "valise",
        "description": "Valise noire avec roulettes",
        "location": "Gare centrale - Quai 3",
        "timestamp": "2025-05-28T10:30:00Z",
        "image_url": "/api/v1/images/obj_123456.jpg",
        "status": "lost",
        "contact_info": {
            "email": "contact@example.com",
            "phone": "+33123456789"
        }
    }

def create_search_request_example() -> dict:
    """Crée un exemple de demande de recherche"""
    return {
        "id": "search_789012",
        "requester_email": "user@example.com",
        "object_description": "Mon téléphone Samsung noir",
        "search_image_url": "/api/v1/images/search_789012.jpg",
        "location_hint": "Café du centre-ville",
        "timestamp": "2025-05-28T14:15:00Z",
        "status": "searching"
    }

# === CONFIGURATION DES EXEMPLES POUR OPENAPI ===
OPENAPI_EXAMPLES = {
    "detection": create_detection_example(),
    "lost_object": create_lost_object_example(),
    "search_request": create_search_request_example(),
    "detection_response": {
        "success": True,
        "message": "Détection réalisée avec succès",
        "data": {
            "detections": [create_detection_example()],
            "detection_count": 1,
            "processing_time": 0.156,
            "model_used": "stable_model"
        },
        "metadata": {
            "image_size": [640, 480],
            "timestamp": 1640995200.0,
            "request_id": "req_abc123"
        }
    },
    "error_response": {
        "success": False,
        "error_code": "DETECTION_FAILED",
        "message": "Erreur lors de la détection d'objets",
        "details": {
            "error_type": "ProcessingError",
            "file_info": "image.jpg (2.3MB)"
        },
        "timestamp": 1640995200.0
    }
}

# === MAPPING DES CLASSES D'OBJETS ===
# Import depuis la configuration si disponible
try:
    from app.config import DETECTED_CLASSES, CLASS_COLORS
    
    OBJECT_CLASS_MAPPING = DETECTED_CLASSES
    OBJECT_CLASS_COLORS = CLASS_COLORS
    
except ImportError:
    # Définition locale si config non disponible
    OBJECT_CLASS_MAPPING = {
        "valise": 0,
        "sac": 1,
        "telephone": 2,
        "sac_a_dos": 3,
        "portefeuille": 4,
        "cles": 5
    }
    
    OBJECT_CLASS_COLORS = {
        0: (255, 0, 0),    # Rouge
        1: (0, 255, 0),    # Vert
        2: (0, 0, 255),    # Bleu
        3: (255, 255, 0),  # Jaune
        4: (255, 0, 255),  # Magenta
        5: (0, 255, 255)   # Cyan
    }

# === FONCTIONS UTILITAIRES ===
def get_class_name(class_id: int) -> str:
    """
    Retourne le nom de la classe à partir de son ID
    Args:
        class_id: ID de la classe (0-5)
    Returns:
        str: Nom de la classe ou "unknown"
    """
    for name, id_val in OBJECT_CLASS_MAPPING.items():
        if id_val == class_id:
            return name
    return "unknown"

def get_class_id(class_name: str) -> int:
    """
    Retourne l'ID de la classe à partir de son nom
    Args:
        class_name: Nom de la classe
    Returns:
        int: ID de la classe ou -1 si non trouvée
    """
    return OBJECT_CLASS_MAPPING.get(class_name.lower(), -1)

def get_class_color(class_id: int) -> tuple:
    """
    Retourne la couleur RGB d'une classe
    Args:
        class_id: ID de la classe
    Returns:
        tuple: Couleur RGB (r, g, b)
    """
    return OBJECT_CLASS_COLORS.get(class_id, (128, 128, 128))

def is_valid_class(class_name: str) -> bool:
    """
    Vérifie si une classe est valide
    Args:
        class_name: Nom de la classe
    Returns:
        bool: True si classe valide
    """
    return class_name.lower() in OBJECT_CLASS_MAPPING

# === EXPORT PRINCIPAL ===
__all__ = [
    # Schémas de détection
    "BoundingBox",
    "Detection", 
    "DetectionResult",
    "ObjectTrack",
    "TrackingResult",
    
    # Requêtes de détection
    "ImageDetectionRequest",
    "VideoDetectionRequest", 
    "StreamDetectionRequest",
    "BatchDetectionRequest",
    
    # Configuration
    "DetectionConfig",
    "TrackingConfig",
    "ModelInfo",
    
    # Matching
    "MatchingRequest",
    "MatchingFeatures",
    "SimilarityResult", 
    "MatchingResult",
    
    # Modèles métier
    "LostObject",
    "LostObjectCreate",
    "LostObjectUpdate",
    "SearchRequest",
    "SearchRequestCreate",
    
    # Enums
    "ObjectClass",
    "DetectionStatus",
    "TrackingStatus",
    "MatchingMethod",
    "LostObjectStatus",
    
    # Réponses
    "BaseResponse",
    "SuccessResponse",
    "ErrorResponse",
    "DetectionResponse",
    "VideoDetectionResponse",
    "StreamDetectionResponse",
    "BatchDetectionResponse",
    "MatchingResponse",
    "SimilaritySearchResponse",
    "LostObjectResponse",
    "LostObjectListResponse",
    "SearchRequestResponse",
    "SearchResultsResponse",
    "HealthResponse",
    "MetricsResponse",
    "ModelStatusResponse",
    
    # WebSocket
    "WebSocketMessage",
    "StreamFrame",
    "DetectionEvent",
    
    # Types
    "ResponseStatus",
    "ErrorCode", 
    "MessageType",
    
    # Utilitaires
    "validate_schemas",
    "create_detection_example",
    "create_lost_object_example",
    "create_search_request_example",
    "get_class_name",
    "get_class_id",
    "get_class_color",
    "is_valid_class",
    
    # Constantes
    "OPENAPI_EXAMPLES",
    "OBJECT_CLASS_MAPPING",
    "OBJECT_CLASS_COLORS"
]

# === VALIDATION AUTOMATIQUE ===
try:
    validate_schemas()
except Exception as e:
    logger.error(f"❌ Erreur validation automatique: {e}")

# === LOG DE FIN D'IMPORT ===
logger.info("📦 Module de schémas chargé")
logger.info(f"🎯 {len(OBJECT_CLASS_MAPPING)} classes d'objets supportées")
logger.info(f"📊 {len(__all__)} schémas et utilitaires disponibles")

# === TESTS DE VALIDATION ===
if __name__ == "__main__":
    print("🧪 Tests de validation des schémas...")
    
    # Test mapping des classes
    print(f"✅ Classes supportées: {list(OBJECT_CLASS_MAPPING.keys())}")
    
    # Test des utilitaires
    print(f"✅ Classe 'valise' -> ID: {get_class_id('valise')}")
    print(f"✅ ID 2 -> Classe: {get_class_name(2)}")
    print(f"✅ Couleur classe 0: {get_class_color(0)}")
    print(f"✅ 'telephone' valide: {is_valid_class('telephone')}")
    
    # Test exemples
    detection_ex = create_detection_example()
    print(f"✅ Exemple détection: {detection_ex['class_name']}")
    
    lost_obj_ex = create_lost_object_example() 
    print(f"✅ Exemple objet perdu: {lost_obj_ex['object_class']}")
    
    # Validation finale
    if validate_schemas():
        print("🎉 Tous les schémas sont valides!")
    else:
        print("❌ Erreurs de validation détectées")
    
    print(f"📊 {len(__all__)} éléments exportés")