"""
Pytest Configuration and Fixtures for Lost Objects Detection Service
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import numpy as np
import cv2
import torch

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requiring full service)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )

# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "models_dir": "tests/fixtures/models",
        "cache_dir": "tests/fixtures/cache", 
        "temp_dir": "tests/fixtures/temp",
        "test_data_dir": "tests/fixtures/data",
        "test_images_dir": "tests/fixtures/images",
        "test_videos_dir": "tests/fixtures/videos"
    }

# Temporary directories
@pytest.fixture(scope="session")
def temp_storage_dir():
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="lost_objects_test_")
    
    # Create subdirectories
    (Path(temp_dir) / "models").mkdir()
    (Path(temp_dir) / "cache").mkdir()
    (Path(temp_dir) / "temp").mkdir()
    (Path(temp_dir) / "logs").mkdir()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

# Mock services
@pytest.fixture
def mock_model_service():
    """Mock ModelService"""
    service = Mock()
    service.initialize = AsyncMock()
    service.cleanup = AsyncMock()
    service.get_model = AsyncMock()
    service.get_available_models = AsyncMock(return_value={
        "test_model": {
            "name": "test_model",
            "type": "test",
            "loaded": True,
            "memory_usage_mb": 100
        }
    })
    service.get_service_stats = AsyncMock(return_value={
        "total_inferences": 0,
        "average_inference_time": 0.1,
        "error_count": 0
    })
    service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "default_model_loaded": True
    })
    return service

@pytest.fixture
def mock_stream_service():
    """Mock StreamService"""
    service = Mock()
    service.start_service = AsyncMock()
    service.stop_service = AsyncMock()
    service.connect_client = AsyncMock(return_value=True)
    service.disconnect_client = AsyncMock()
    service.get_service_stats = AsyncMock(return_value={
        "service": {"active_connections": 0},
        "processing": {"processing_active": True}
    })
    return service

@pytest.fixture
def mock_video_service():
    """Mock VideoService"""
    service = Mock()
    service.start_video_processing = AsyncMock(return_value="test_job_id")
    service.get_job_status = AsyncMock(return_value={
        "status": "completed",
        "progress": 1.0
    })
    return service

@pytest.fixture
def mock_detector():
    """Mock LostObjectsDetector"""
    detector = Mock()
    detector.detect_objects.return_value = {
        "detections": [
            {
                "class": "backpack",
                "confidence": 0.85,
                "bbox": [50, 50, 150, 150]
            }
        ],
        "processing_time": 0.1
    }
    detector.detect_lost_objects.return_value = {
        "all_detections": [
            {
                "class": "backpack", 
                "confidence": 0.85,
                "bbox": [50, 50, 150, 150]
            }
        ],
        "lost_objects": [],
        "suspect_objects": [],
        "processing_time": 0.1
    }
    detector.get_stats.return_value = {
        "total_detections": 1,
        "avg_processing_time": 0.1
    }
    return detector

# Test data fixtures
@pytest.fixture
def test_image():
    """Create test image"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue
    cv2.rectangle(image, (200, 100), (300, 200), (0, 255, 0), -1)  # Green
    cv2.rectangle(image, (400, 50), (500, 150), (0, 0, 255), -1)  # Red
    
    # Add text
    cv2.putText(image, "Test Image", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

@pytest.fixture
def test_image_bytes(test_image):
    """Convert test image to bytes"""
    _, buffer = cv2.imencode('.jpg', test_image)
    return buffer.tobytes()

@pytest.fixture
def test_images():
    """Create multiple test images"""
    images = []
    for i in range(3):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Different colored rectangle for each image
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
        cv2.rectangle(image, (50 + i*20, 50 + i*20), (150 + i*20, 150 + i*20), color, -1)
        cv2.putText(image, f"Test {i}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        images.append(image)
    return images

@pytest.fixture
def test_video_frames(test_images):
    """Create test video frames"""
    return test_images

@pytest.fixture
def mock_model_weights():
    """Create mock model weights"""
    # Create a simple mock model state dict
    state_dict = {
        'backbone.conv1.weight': torch.randn(64, 3, 7, 7),
        'backbone.conv1.bias': torch.randn(64),
        'fpn.conv1.weight': torch.randn(256, 64, 3, 3),
        'fpn.conv1.bias': torch.randn(256),
        'prediction_head.cls_head.weight': torch.randn(28, 256),
        'prediction_head.cls_head.bias': torch.randn(28),
        'prediction_head.reg_head.weight': torch.randn(4, 256),
        'prediction_head.reg_head.bias': torch.randn(4)
    }
    return state_dict

# Database fixtures
@pytest.fixture
def mock_database():
    """Mock database connection"""
    db = Mock()
    db.connect = Mock()
    db.disconnect = Mock()
    db.execute = Mock()
    db.fetch = Mock()
    return db

# Test client fixtures
@pytest.fixture
def test_client():
    """Create test client for API testing"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)

@pytest.fixture
async def async_test_client():
    """Create async test client"""
    from httpx import AsyncClient
    from app.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

# Mock external dependencies
@pytest.fixture
def mock_redis():
    """Mock Redis connection"""
    redis = Mock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=False)
    return redis

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.send_bytes = AsyncMock()
    websocket.receive = AsyncMock()
    websocket.close = AsyncMock()
    return websocket

# Performance fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# Environment fixtures
@pytest.fixture
def test_environment():
    """Set up test environment variables"""
    import os
    
    original_env = dict(os.environ)
    
    # Set test environment variables
    test_vars = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "debug",
        "MODELS_DIR": "tests/fixtures/models",
        "CACHE_DIR": "tests/fixtures/cache",
        "TEMP_DIR": "tests/fixtures/temp"
    }
    
    os.environ.update(test_vars)
    
    yield test_vars
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# GPU fixtures
@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing"""
    return torch.cuda.is_available()

@pytest.fixture
def mock_gpu_model():
    """Mock GPU model for testing"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = Mock()
    model.to = Mock(return_value=model)
    model.device = device
    model.eval = Mock()
    return model

# Async utilities
@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test"""
    yield
    
    # Cleanup any temporary files created during tests
    temp_patterns = [
        "test_*",
        "*.tmp",
        "*.test"
    ]
    
    import glob
    for pattern in temp_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                if Path(file).is_file():
                    Path(file).unlink()
                elif Path(file).is_dir():
                    shutil.rmtree(file)
            except:
                pass

# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests"""
    import logging
    
    # Reduce log level for tests
    logging.getLogger("app").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    
    yield
    
    # Reset logging after tests
    logging.getLogger("app").setLevel(logging.INFO)

# Pytest hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Add integration marker for integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker for performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add GPU marker for GPU tests
        if "gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Add slow marker for slow tests
        if any(keyword in item.nodeid for keyword in ["slow", "batch", "video"]):
            item.add_marker(pytest.mark.slow)

def pytest_runtest_setup(item):
    """Setup for each test"""
    # Skip GPU tests if GPU not available
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("GPU not available")

# Custom assertions
class CustomAssertions:
    """Custom assertions for testing"""
    
    @staticmethod
    def assert_detection_result(result):
        """Assert detection result format"""
        assert isinstance(result, dict)
        assert "detections" in result
        assert "processing_time" in result
        assert isinstance(result["detections"], list)
        assert isinstance(result["processing_time"], (int, float))
    
    @staticmethod
    def assert_bbox_format(bbox):
        """Assert bounding box format"""
        assert isinstance(bbox, (list, tuple))
        assert len(bbox) == 4
        assert all(isinstance(coord, (int, float)) for coord in bbox)
        assert bbox[0] < bbox[2]  # x1 < x2
        assert bbox[1] < bbox[3]  # y1 < y2

@pytest.fixture
def assert_helpers():
    """Provide custom assertion helpers"""
    return CustomAssertions()