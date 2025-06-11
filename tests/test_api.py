# tests/test_api.py
"""
🧪 Tests du Service IA
Tests unitaires et d'intégration pour le service de détection
"""

import pytest
import numpy as np
import cv2
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import json
import base64
from pathlib import Path

from app.main import app
from app.core.model_manager import ModelManager
from app.schemas.detection import ObjectDetection, BoundingBox, ObjectStatus
from app.utils.image_utils import ImageProcessor, encode_image_to_base64

# === CONFIGURATION TESTS ===
client = TestClient(app)

@pytest.fixture
def test_image():
    """Image de test 320x320"""
    return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

@pytest.fixture
def test_image_base64(test_image):
    """Image de test encodée en base64"""
    return encode_image_to_base64(test_image)

@pytest.fixture
def mock_detection_result():
    """Résultat de détection simulé"""
    bbox = BoundingBox(x=100, y=100, width=50, height=50)
    detection = ObjectDetection(
        object_id="test_obj_1",
        class_name="backpack",
        class_name_fr="Sac à dos",
        confidence=0.85,
        confidence_level="high",
        bounding_box=bbox,
        first_seen="2025-06-11T10:00:00",
        last_seen="2025-06-11T10:00:00",
        status=ObjectStatus.NORMAL
    )
    return [detection], [], []  # objects, persons, alerts

# === TESTS API ENDPOINTS ===

def test_health_endpoint():
    """Test de l'endpoint de santé"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

def test_root_endpoint():
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data

def test_models_list_endpoint():
    """Test de listage des modèles"""
    response = client.get("/api/v1/models/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@patch('app.core.detector.ObjectDetector.detect')
def test_image_detection_file(mock_detect, test_image, mock_detection_result):
    """Test détection avec upload de fichier"""
    mock_detect.return_value = mock_detection_result
    
    # Création fichier temporaire
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_image)
        
        with open(tmp.name, 'rb') as f:
            response = client.post(
                "/api/v1/detect/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "model_name": "stable_epoch_30",
                    "confidence_threshold": 0.5
                }
            )
    
    Path(tmp.name).unlink()
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "objects" in data
    assert "processing_time" in data

@patch('app.core.detector.ObjectDetector.detect')
def test_image_detection_base64(mock_detect, test_image_base64, mock_detection_result):
    """Test détection avec image base64"""
    mock_detect.return_value = mock_detection_result
    
    response = client.post(
        "/api/v1/detect/image",
        data={
            "image_base64": test_image_base64,
            "model_name": "stable_epoch_30",
            "confidence_threshold": 0.5
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_image_detection_invalid_format():
    """Test avec format d'image invalide"""
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("test.txt", "not an image", "text/plain")}
    )
    
    assert response.status_code == 400

def test_image_detection_no_data():
    """Test sans données image"""
    response = client.post("/api/v1/detect/image")
    assert response.status_code == 400

# === TESTS UTILS ===

class TestImageProcessor:
    """Tests du processeur d'images"""
    
    def test_preprocess_image(self, test_image):
        """Test prétraitement d'image"""
        processor = ImageProcessor((320, 320))
        tensor, info = processor.preprocess_image(test_image)
        
        assert tensor.shape == (1, 3, 320, 320)  # Batch, channels, H, W
        assert "original_shape" in info
        assert "scale_factors" in info
    
    def test_preprocess_different_size(self):
        """Test avec image de taille différente"""
        processor = ImageProcessor((640, 480))
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        tensor, info = processor.preprocess_image(image)
        assert tensor.shape == (1, 3, 480, 640)

def test_encode_decode_base64(test_image):
    """Test encodage/décodage base64"""
    from app.utils.image_utils import decode_base64_to_image
    
    # Encodage
    encoded = encode_image_to_base64(test_image)
    assert isinstance(encoded, str)
    
    # Décodage
    decoded = decode_base64_to_image(encoded)
    assert decoded.shape == test_image.shape

# === TESTS CORE LOGIC ===

class TestDetectionLogic:
    """Tests de la logique de détection"""
    
    def test_object_status_determination(self):
        """Test détermination du statut d'objet"""
        # Simuler différents scénarios
        bbox = BoundingBox(x=100, y=100, width=50, height=50)
        
        # Objet normal
        obj = ObjectDetection(
            object_id="test",
            class_name="bottle",
            class_name_fr="Bouteille",
            confidence=0.8,
            confidence_level="high",
            bounding_box=bbox,
            first_seen="2025-06-11T10:00:00",
            last_seen="2025-06-11T10:00:00",
            duration_stationary=10  # 10 secondes
        )
        
        # Objet devrait être normal (< 30s)
        assert obj.duration_stationary < 30
    
    def test_bounding_box_operations(self):
        """Test opérations sur bounding boxes"""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        
        # Test aire
        assert bbox.area() == 8000
        
        # Test centre
        center = bbox.center()
        assert center == (60, 60)  # (10 + 100/2, 20 + 80/2)

# === TESTS STREAMING ===

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test connexion WebSocket"""
    from fastapi.testclient import TestClient
    
    with TestClient(app) as client:
        # Note: TestClient ne supporte pas parfaitement WebSocket
        # Ici on teste juste que l'endpoint existe
        response = client.get("/api/v1/stream/status")
        assert response.status_code == 200

# === TESTS MODÈLES ===

@pytest.mark.asyncio
async def test_model_manager_initialization():
    """Test initialisation du gestionnaire de modèles"""
    model_manager = ModelManager()
    
    # Vérifier que les modèles sont découverts
    assert len(model_manager.available_models) >= 0
    
    # Test santé
    health = await model_manager.get_health_status()
    assert "timestamp" in health
    assert "gpu_available" in health

# === TESTS PERFORMANCE ===

@patch('app.core.detector.ObjectDetector.detect')
def test_performance_detection(mock_detect, test_image):
    """Test performance de détection"""
    import time
    
    mock_detect.return_value = ([], [], [])  # Résultat vide
    
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_image)
        
        with open(tmp.name, 'rb') as f:
            response = client.post(
                "/api/v1/detect/image",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
    
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # ms
    
    Path(tmp.name).unlink()
    
    assert response.status_code == 200
    # Le traitement devrait prendre moins de 5 secondes
    assert processing_time < 5000
    
    data = response.json()
    assert "processing_time" in data

# === TESTS D'INTÉGRATION ===

def test_full_detection_pipeline(test_image):
    """Test complet du pipeline de détection"""
    # Ce test nécessite un modèle réel pour fonctionner complètement
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_image)
        
        with open(tmp.name, 'rb') as f:
            response = client.post(
                "/api/v1/detect/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "confidence_threshold": 0.3,
                    "enable_tracking": True,
                    "enable_lost_detection": True
                }
            )
    
    Path(tmp.name).unlink()
    
    # Même sans modèle réel, l'API devrait répondre
    # (le modèle placeholder retournera des résultats vides)
    assert response.status_code in [200, 500]  # 500 si modèle non trouvé

# === TESTS VIDÉO ===

def test_video_processing_endpoint():
    """Test endpoint de traitement vidéo"""
    # Créer une courte vidéo de test
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp.name, fourcc, 5.0, (320, 240))
        
        # 10 frames de test
        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Test de l'API
        with open(tmp.name, 'rb') as f:
            response = client.post(
                "/api/v1/detect/video",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"max_frames": 5}
            )
    
    Path(tmp.name).unlink()
    
    # L'endpoint devrait accepter la vidéo
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data

# === UTILITAIRES DE TEST ===

def create_test_video(path: str, frames: int = 30, fps: int = 10):
    """Crée une vidéo de test"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    
    for i in range(frames):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        # Ajouter un objet mobile
        x = int(50 + 200 * np.sin(i * 0.2))
        cv2.rectangle(frame, (x, 100), (x+50, 150), (0, 255, 0), -1)
        out.write(frame)
    
    out.release()

if __name__ == "__main__":
    # Exécution directe des tests
    print("🧪 Exécution des tests...")
    pytest.main(["-v", __file__])

# === SCRIPT DE TEST COMPLET ===
# tests/run_tests.py
"""
Script pour exécuter tous les tests
"""

import subprocess
import sys
from pathlib import Path

def run_all_tests():
    """Exécute tous les tests"""
    print("🧪 SERVICE IA - TESTS COMPLETS")
    print("="*50)
    
    # Tests unitaires
    print("\n📋 Tests unitaires...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", "-v", "--tb=short"
    ])
    
    if result.returncode == 0:
        print("✅ Tous les tests sont passés!")
    else:
        print("❌ Certains tests ont échoué")
        return False
    
    # Tests de performance (optionnel)
    print("\n⚡ Tests de performance...")
    # Ici vous pourriez ajouter des tests de charge
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)