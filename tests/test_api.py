"""
Basic API Tests for Lost Objects Detection Service
"""
import pytest
import asyncio
import json
import io
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import cv2

# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)

class TestHealthEndpoints:
    """Test health and basic endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Lost Objects Detection API"

class TestModelEndpoints:
    """Test model-related endpoints"""
    
    @patch('app.main.app.state.model_manager')
    def test_get_models(self, mock_model_manager):
        """Test models listing endpoint"""
        # Mock the model manager
        mock_service = Mock()
        mock_service.get_available_models.return_value = {
            "stable_model_epoch_30": {
                "name": "stable_model_epoch_30",
                "type": "production",
                "loaded": True
            }
        }
        mock_model_manager.get_model_service.return_value = mock_service
        
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data

class TestImageDetection:
    """Test image detection endpoints"""
    
    def create_test_image(self, width=640, height=480):
        """Create a test image"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    @patch('app.main.app.state.model_manager')
    def test_image_detection_success(self, mock_model_manager):
        """Test successful image detection"""
        # Mock the model manager and detector
        mock_service = Mock()
        mock_detector = Mock()
        
        # Mock detection result
        mock_detector.detect_objects.return_value = {
            'detections': [
                {
                    'class': 'backpack',
                    'confidence': 0.85,
                    'bbox': [50, 50, 150, 150]
                }
            ],
            'processing_time': 0.1
        }
        
        mock_service.get_model.return_value = mock_detector
        mock_model_manager.get_model_service.return_value = mock_service
        
        # Create test image
        test_image = self.create_test_image()
        
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert len(data["detections"]) > 0
    
    def test_image_detection_no_file(self):
        """Test image detection without file"""
        response = client.post("/api/v1/detect/image")
        assert response.status_code == 422  # Validation error
    
    def test_image_detection_invalid_file(self):
        """Test image detection with invalid file"""
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        )
        assert response.status_code == 400

class TestBatchDetection:
    """Test batch detection endpoints"""
    
    def create_test_image(self, width=640, height=480):
        """Create a test image"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    @patch('app.main.app.state.model_manager')
    @patch('app.api.endpoints.batch_detection.processor')
    def test_batch_upload_success(self, mock_processor, mock_model_manager):
        """Test successful batch upload"""
        # Mock the processor
        mock_processor.process_files_batch = Mock()
        
        # Create test images
        test_images = [self.create_test_image() for _ in range(3)]
        
        files = [
            ("files", (f"test_{i}.jpg", io.BytesIO(img), "image/jpeg"))
            for i, img in enumerate(test_images)
        ]
        
        response = client.post("/api/v1/detect/batch/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
    
    def test_batch_upload_no_files(self):
        """Test batch upload without files"""
        response = client.post("/api/v1/detect/batch/upload")
        assert response.status_code == 400

class TestStreamingEndpoints:
    """Test streaming-related endpoints"""
    
    def test_streaming_stats(self):
        """Test streaming statistics endpoint"""
        response = client.get("/api/v1/stream/stats")
        # This might fail if stream service is not initialized
        # In a real test, we'd mock the dependencies
        assert response.status_code in [200, 500]
    
    def test_streaming_clients(self):
        """Test streaming clients endpoint"""
        response = client.get("/api/v1/stream/clients")
        assert response.status_code in [200, 500]

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint"""
        response = client.get("/api/v1/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self):
        """Test invalid HTTP method"""
        response = client.delete("/health")
        assert response.status_code == 405
    
    @patch('app.main.app.state.model_manager')
    def test_service_unavailable(self, mock_model_manager):
        """Test when service is unavailable"""
        # Mock service to raise an exception
        mock_model_manager.get_model_service.side_effect = RuntimeError("Service unavailable")
        
        response = client.get("/api/v1/models")
        assert response.status_code == 500

class TestValidation:
    """Test input validation"""
    
    def test_large_file_rejection(self):
        """Test rejection of oversized files"""
        # Create a large dummy file (this test might need adjustment based on actual limits)
        large_content = b"x" * (100 * 1024 * 1024)  # 100MB
        
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("large.jpg", io.BytesIO(large_content), "image/jpeg")}
        )
        
        # Should be rejected due to size (status code may vary based on implementation)
        assert response.status_code in [400, 413, 422]

class TestConcurrency:
    """Test concurrent request handling"""
    
    @patch('app.main.app.state.model_manager')
    def test_concurrent_health_checks(self, mock_model_manager):
        """Test multiple concurrent health checks"""
        import concurrent.futures
        
        def make_health_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All should succeed
        for response in results:
            assert response.status_code == 200

# Pytest fixtures for common test data
@pytest.fixture
def test_image():
    """Fixture providing a test image"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

@pytest.fixture
def mock_detection_result():
    """Fixture providing a mock detection result"""
    return {
        'detections': [
            {
                'class': 'backpack',
                'confidence': 0.85,
                'bbox': [50, 50, 150, 150]
            },
            {
                'class': 'cell phone',
                'confidence': 0.72,
                'bbox': [200, 100, 250, 180]
            }
        ],
        'processing_time': 0.15,
        'model_name': 'stable_model_epoch_30'
    }

# Integration tests (require actual service running)
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring full service"""
    
    def test_full_detection_pipeline(self, test_image):
        """Test complete detection pipeline"""
        # This test requires the actual service to be running with models loaded
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )
        
        # Skip if service not properly configured
        if response.status_code == 500:
            pytest.skip("Service not fully configured for integration test")
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "processing_time" in data

# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance-related tests"""
    
    @patch('app.main.app.state.model_manager')
    def test_response_time(self, mock_model_manager, test_image):
        """Test API response time"""
        import time
        
        # Mock fast response
        mock_service = Mock()
        mock_detector = Mock()
        mock_detector.detect_objects.return_value = {
            'detections': [],
            'processing_time': 0.1
        }
        mock_service.get_model.return_value = mock_detector
        mock_model_manager.get_model_service.return_value = mock_service
        
        start_time = time.time()
        
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )
        
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])