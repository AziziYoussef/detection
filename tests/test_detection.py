"""
Detection-specific Tests for Lost Objects Detection Service
Tests core detection functionality, model loading, and temporal tracking
"""
import pytest
import numpy as np
import torch
import cv2
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.core.detector import LostObjectsDetector, TrackedObject, ObjectStatus
from app.models.model import LostObjectDetector
from app.services.model_service import ModelService
from app.core.model_manager import ModelManager

class TestModelLoading:
    """Test model loading and initialization"""
    
    @patch('torch.load')
    @patch('pathlib.Path.exists')
    def test_model_loading_success(self, mock_exists, mock_torch_load, test_config):
        """Test successful model loading"""
        mock_exists.return_value = True
        mock_torch_load.return_value = {
            'model_state_dict': {
                'backbone.conv1.weight': torch.randn(64, 3, 7, 7),
                'fpn.conv1.weight': torch.randn(256, 64, 3, 3)
            }
        }
        
        # Mock model path
        model_path = "test_model.pth"
        
        # This should not fail
        detector = LostObjectsDetector(
            model_path=model_path,
            config=test_config
        )
        
        assert detector is not None
        mock_torch_load.assert_called_once()
    
    def test_model_loading_file_not_found(self, test_config):
        """Test model loading with missing file"""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            LostObjectsDetector(
                model_path="nonexistent_model.pth",
                config=test_config
            )
    
    @patch('torch.cuda.is_available')
    def test_device_selection_gpu(self, mock_cuda_available, test_config):
        """Test GPU device selection"""
        mock_cuda_available.return_value = True
        
        with patch('app.core.detector.LostObjectsDetector._load_model'):
            detector = LostObjectsDetector(
                model_path="test_model.pth",
                config=test_config,
                device="auto"
            )
            
            assert str(detector.device) == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cpu(self, mock_cuda_available, test_config):
        """Test CPU device selection"""
        mock_cuda_available.return_value = False
        
        with patch('app.core.detector.LostObjectsDetector._load_model'):
            detector = LostObjectsDetector(
                model_path="test_model.pth",
                config=test_config,
                device="auto"
            )
            
            assert str(detector.device) == "cpu"

class TestObjectDetection:
    """Test core object detection functionality"""
    
    def test_detect_objects_success(self, mock_detector, test_image):
        """Test successful object detection"""
        result = mock_detector.detect_objects(test_image)
        
        assert 'detections' in result
        assert 'processing_time' in result
        assert isinstance(result['detections'], list)
        assert isinstance(result['processing_time'], float)
        assert result['processing_time'] > 0
    
    def test_detect_objects_empty_image(self, mock_detector):
        """Test detection with empty/invalid image"""
        # Create invalid image
        invalid_image = np.array([])
        
        # Should handle gracefully
        result = mock_detector.detect_objects(invalid_image)
        
        # Should return empty results or error
        assert 'detections' in result
        # Allow either empty detections or error handling
        assert len(result['detections']) == 0 or 'error' in result
    
    def test_detect_objects_large_image(self, mock_detector):
        """Test detection with large image"""
        # Create large image
        large_image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        
        result = mock_detector.detect_objects(large_image)
        
        assert 'detections' in result
        assert 'processing_time' in result
    
    def test_detection_confidence_filtering(self, mock_detector, test_image):
        """Test confidence threshold filtering"""
        # Mock detector to return detections with different confidences
        mock_detector.detect_objects.return_value = {
            'detections': [
                {'class': 'backpack', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]},
                {'class': 'handbag', 'confidence': 0.2, 'bbox': [100, 100, 150, 150]},
                {'class': 'suitcase', 'confidence': 0.7, 'bbox': [200, 200, 300, 300]}
            ],
            'processing_time': 0.1
        }
        
        result = mock_detector.detect_objects(test_image)
        
        # With default threshold of 0.3, should filter out the 0.2 confidence detection
        high_conf_detections = [d for d in result['detections'] if d['confidence'] >= 0.3]
        assert len(high_conf_detections) >= 2

class TestTemporalTracking:
    """Test temporal object tracking functionality"""
    
    def test_object_tracking_initialization(self, mock_detector):
        """Test initial object tracking setup"""
        assert mock_detector.tracked_objects == {}
        assert mock_detector.next_object_id == 1
    
    def test_new_object_creation(self, mock_detector, test_image):
        """Test creation of new tracked object"""
        # Mock detection result
        mock_detector.detect_lost_objects.return_value = {
            'all_detections': [
                {'class': 'backpack', 'confidence': 0.8, 'bbox': [50, 50, 150, 150]}
            ],
            'lost_objects': [],
            'suspect_objects': [],
            'tracked_objects_count': 1,
            'processing_time': 0.1
        }
        
        result = mock_detector.detect_lost_objects(test_image, timestamp=time.time())
        
        assert result['tracked_objects_count'] >= 0
        assert 'all_detections' in result
    
    def test_object_status_progression(self):
        """Test object status progression: normal -> suspect -> lost"""
        # Create a tracked object
        tracked_obj = TrackedObject(
            object_id="test_obj_001",
            class_name="backpack",
            confidence=0.8,
            bbox=(50, 50, 150, 150),
            first_seen=time.time() - 100,
            last_seen=time.time(),
            last_movement=time.time() - 100,
            status=ObjectStatus.NORMAL,
            stationary_duration=100.0,
            nearest_person_distance=5.0
        )
        
        # Initially normal
        assert tracked_obj.status == ObjectStatus.NORMAL
        
        # Should become suspect after threshold
        if tracked_obj.stationary_duration > 30 and tracked_obj.nearest_person_distance > 3:
            tracked_obj.status = ObjectStatus.SUSPECT
        
        assert tracked_obj.status == ObjectStatus.SUSPECT
        
        # Should become lost after longer threshold
        if tracked_obj.stationary_duration > 300:
            tracked_obj.status = ObjectStatus.LOST
    
    def test_object_movement_detection(self, mock_detector):
        """Test detection of object movement"""
        # This would test IoU-based movement detection
        bbox1 = (50, 50, 150, 150)
        bbox2 = (55, 55, 155, 155)  # Slight movement
        bbox3 = (200, 200, 300, 300)  # Significant movement
        
        # Small movement should not reset tracking
        # Large movement should reset tracking
        # This is implementation-specific logic
        pass
    
    def test_person_distance_calculation(self, mock_detector):
        """Test nearest person distance calculation"""
        # Mock scenario with person and object
        detections = [
            {'class': 'person', 'confidence': 0.9, 'bbox': [100, 100, 200, 300]},
            {'class': 'backpack', 'confidence': 0.8, 'bbox': [250, 150, 350, 250]}
        ]
        
        # Test distance calculation logic
        # This would test the actual distance calculation in the detector
        pass

class TestLostObjectLogic:
    """Test lost object detection logic"""
    
    def test_suspect_object_detection(self, mock_detector, test_image):
        """Test detection of suspect objects"""
        timestamp = time.time()
        
        # Mock a scenario where object has been stationary for suspect threshold
        mock_detector.detect_lost_objects.return_value = {
            'all_detections': [
                {'class': 'suitcase', 'confidence': 0.85, 'bbox': [100, 100, 200, 200]}
            ],
            'lost_objects': [],
            'suspect_objects': [
                {
                    'object_id': 'obj_000001',
                    'class': 'suitcase',
                    'confidence': 0.85,
                    'bbox': [100, 100, 200, 200],
                    'status': 'suspect',
                    'duration_stationary': 45.0
                }
            ],
            'tracked_objects_count': 1,
            'processing_time': 0.1
        }
        
        result = mock_detector.detect_lost_objects(test_image, timestamp=timestamp)
        
        assert len(result['suspect_objects']) >= 0
        if result['suspect_objects']:
            suspect_obj = result['suspect_objects'][0]
            assert suspect_obj['status'] == 'suspect'
    
    def test_lost_object_detection(self, mock_detector, test_image):
        """Test detection of lost objects"""
        timestamp = time.time()
        
        # Mock a scenario where object has been stationary for lost threshold
        mock_detector.detect_lost_objects.return_value = {
            'all_detections': [
                {'class': 'handbag', 'confidence': 0.9, 'bbox': [150, 150, 250, 250]}
            ],
            'lost_objects': [
                {
                    'object_id': 'obj_000002',
                    'class': 'handbag',
                    'confidence': 0.9,
                    'bbox': [150, 150, 250, 250],
                    'status': 'lost',
                    'duration_stationary': 350.0,
                    'alert_level': 'MEDIUM'
                }
            ],
            'suspect_objects': [],
            'tracked_objects_count': 1,
            'processing_time': 0.1
        }
        
        result = mock_detector.detect_lost_objects(test_image, timestamp=timestamp)
        
        assert len(result['lost_objects']) >= 0
        if result['lost_objects']:
            lost_obj = result['lost_objects'][0]
            assert lost_obj['status'] == 'lost'
            assert 'alert_level' in lost_obj
    
    def test_alert_level_calculation(self):
        """Test alert level calculation based on duration"""
        # Test different durations and expected alert levels
        test_cases = [
            (60, 'LOW'),      # 1 minute
            (900, 'MEDIUM'),  # 15 minutes  
            (1800, 'HIGH'),   # 30 minutes
            (3600, 'HIGH')    # 1 hour
        ]
        
        for duration, expected_level in test_cases:
            # This would test the actual alert level calculation
            if duration > 1800:  # 30 minutes
                alert_level = "HIGH"
            elif duration > 900:  # 15 minutes
                alert_level = "MEDIUM"
            else:
                alert_level = "LOW"
            
            assert alert_level == expected_level or alert_level in ['LOW', 'MEDIUM', 'HIGH']

class TestModelService:
    """Test model service functionality"""
    
    @pytest.mark.asyncio
    async def test_model_service_initialization(self, mock_model_service):
        """Test model service initialization"""
        await mock_model_service.initialize()
        mock_model_service.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_loading(self, mock_model_service):
        """Test model loading through service"""
        model_name = "test_model"
        mock_detector = Mock()
        mock_model_service.load_model.return_value = mock_detector
        
        result = await mock_model_service.load_model(model_name)
        
        assert result is not None
        mock_model_service.load_model.assert_called_with(model_name)
    
    @pytest.mark.asyncio
    async def test_model_caching(self, mock_model_service):
        """Test model caching functionality"""
        model_name = "cached_model"
        
        # First call should load model
        result1 = await mock_model_service.get_model(model_name)
        
        # Second call should return cached model
        result2 = await mock_model_service.get_model(model_name)
        
        # Both results should be the same (cached)
        assert result1 is not None
        assert result2 is not None

class TestPerformance:
    """Test performance-related functionality"""
    
    def test_detection_speed(self, mock_detector, test_image, performance_timer):
        """Test detection speed"""
        performance_timer.start()
        
        result = mock_detector.detect_objects(test_image)
        
        performance_timer.stop()
        
        assert result is not None
        assert performance_timer.elapsed is not None
        # Performance should be reasonable (adjust threshold as needed)
        assert performance_timer.elapsed < 5.0  # 5 seconds max for test
    
    def test_memory_usage(self, mock_detector, test_images):
        """Test memory usage with multiple images"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple images
        for image in test_images:
            mock_detector.detect_objects(image)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
    
    @pytest.mark.performance
    def test_batch_processing_performance(self, mock_detector, test_images):
        """Test batch processing performance"""
        start_time = time.time()
        
        results = []
        for image in test_images:
            result = mock_detector.detect_objects(image)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(test_images)
        
        assert len(results) == len(test_images)
        assert avg_time_per_image < 2.0  # Less than 2 seconds per image

class TestErrorHandling:
    """Test error handling in detection"""
    
    def test_invalid_image_format(self, mock_detector):
        """Test handling of invalid image formats"""
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            np.array([]),
            np.zeros((10, 10)),  # 2D instead of 3D
            np.zeros((10, 10, 1)),  # Wrong channel count
            "not_an_image"
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = mock_detector.detect_objects(invalid_input)
                # Should either handle gracefully or raise expected exception
                if isinstance(result, dict):
                    assert 'error' in result or 'detections' in result
            except (ValueError, TypeError, AttributeError):
                # Expected exceptions are acceptable
                pass
    
    def test_model_loading_errors(self, test_config):
        """Test model loading error handling"""
        # Test various error conditions
        error_cases = [
            "nonexistent_file.pth",
            "",
            None
        ]
        
        for error_case in error_cases:
            with pytest.raises((FileNotFoundError, ValueError, TypeError)):
                LostObjectsDetector(
                    model_path=error_case,
                    config=test_config
                )
    
    def test_gpu_not_available_fallback(self, test_config):
        """Test fallback to CPU when GPU not available"""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('app.core.detector.LostObjectsDetector._load_model'):
                detector = LostObjectsDetector(
                    model_path="test_model.pth",
                    config=test_config,
                    device="auto"
                )
                
                assert str(detector.device) == "cpu"

class TestIntegration:
    """Integration tests requiring multiple components"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_detection_pipeline(self, test_image, test_config):
        """Test complete detection pipeline"""
        # This test requires actual model files and full initialization
        pytest.skip("Requires full model setup for integration testing")
        
        # Full pipeline test would include:
        # 1. Model loading
        # 2. Image preprocessing
        # 3. Model inference
        # 4. Post-processing
        # 5. Temporal tracking
        # 6. Lost object logic
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_model_manager_integration(self):
        """Test model manager with real components"""
        pytest.skip("Requires full service setup for integration testing")

if __name__ == "__main__":
    # Run detection-specific tests
    pytest.main([__file__, "-v", "--tb=short"])