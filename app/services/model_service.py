"""
Model Management Service for Lost Objects Detection
Handles model loading, caching, and performance optimization
"""
import torch
import asyncio
import os
import threading
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import time
import psutil
from collections import OrderedDict
from dataclasses import dataclass

from ..core.detector import LostObjectsDetector
from ..models.model import LostObjectDetector
from ..config.config import config
from ..config.model_config import MODEL_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information dataclass"""
    name: str
    path: str
    model_type: str
    loaded: bool
    load_time: float
    memory_usage: int  # in MB
    performance_stats: Dict
    last_used: float
    version: str = "1.0"
    description: str = ""

class ModelCache:
    """LRU Cache for models with memory management"""
    
    def __init__(self, max_models: int = 3, max_memory_mb: int = 2048):
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.models: OrderedDict[str, LostObjectsDetector] = OrderedDict()
        self.model_info: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()
    
    def get(self, model_name: str) -> Optional[LostObjectsDetector]:
        """Get model from cache"""
        with self._lock:
            if model_name in self.models:
                # Move to end (most recently used)
                self.models.move_to_end(model_name)
                self.model_info[model_name].last_used = time.time()
                return self.models[model_name]
            return None
    
    def put(self, model_name: str, model: LostObjectsDetector, model_info: ModelInfo):
        """Add model to cache"""
        with self._lock:
            # Remove if already exists
            if model_name in self.models:
                del self.models[model_name]
            
            # Add new model
            self.models[model_name] = model
            self.model_info[model_name] = model_info
            
            # Enforce cache limits
            self._enforce_limits()
    
    def remove(self, model_name: str):
        """Remove model from cache"""
        with self._lock:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_info[model_name]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _enforce_limits(self):
        """Enforce cache size and memory limits"""
        # Remove oldest models if exceeding max_models
        while len(self.models) > self.max_models:
            oldest_model = next(iter(self.models))
            logger.info(f"Removing model {oldest_model} from cache (max models limit)")
            del self.models[oldest_model]
            del self.model_info[oldest_model]
        
        # Check memory usage
        total_memory = sum(info.memory_usage for info in self.model_info.values())
        
        while total_memory > self.max_memory_mb and len(self.models) > 1:
            # Remove least recently used model
            lru_model = min(
                self.model_info.items(),
                key=lambda x: x[1].last_used
            )[0]
            
            logger.info(f"Removing model {lru_model} from cache (memory limit)")
            total_memory -= self.model_info[lru_model].memory_usage
            del self.models[lru_model]
            del self.model_info[lru_model]
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_memory = sum(info.memory_usage for info in self.model_info.values())
            return {
                'loaded_models': len(self.models),
                'max_models': self.max_models,
                'total_memory_mb': total_memory,
                'max_memory_mb': self.max_memory_mb,
                'models': {
                    name: {
                        'loaded': True,
                        'memory_mb': info.memory_usage,
                        'last_used': info.last_used
                    }
                    for name, info in self.model_info.items()
                }
            }

class ModelService:
    """
    Model Management Service
    
    Provides high-level interface for model operations including:
    - Model loading and caching
    - Performance optimization
    - Multi-model support
    - Health monitoring
    """
    
    def __init__(self):
        self.models_dir = Path(config.get('models_dir', 'storage/models'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model cache
        self.cache = ModelCache(
            max_models=config.get('max_cached_models', 3),
            max_memory_mb=config.get('max_cache_memory_mb', 2048)
        )
        
        # Available models registry
        self.available_models: Dict[str, ModelInfo] = {}
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'total_inference_time': 0.0,
            'model_load_times': {},
            'error_count': 0
        }
        
        # Default model
        self.default_model_name = config.get('default_model', 'stable_model_epoch_30')
        
        logger.info(f"ModelService initialized on {self.device}")
    
    async def initialize(self):
        """Initialize model service and discover available models"""
        try:
            await self._discover_models()
            
            # Preload default model
            if self.default_model_name in self.available_models:
                await self.load_model(self.default_model_name)
                logger.info(f"Default model '{self.default_model_name}' preloaded")
            
            logger.info("ModelService initialization complete")
            
        except Exception as e:
            logger.error(f"ModelService initialization failed: {e}")
            raise
    
    async def _discover_models(self):
        """Discover available models in models directory"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        model_configs = {
            'stable_model_epoch_30': {
                'type': 'production',
                'description': 'Stable production model - 28 classes',
                'version': '1.0'
            },
            'best_extended_model': {
                'type': 'extended',
                'description': 'Extended model with enhanced features',
                'version': '1.1'
            },
            'fast_stream_model': {
                'type': 'streaming',
                'description': 'Optimized for real-time streaming',
                'version': '1.0'
            },
            'mobile_model': {
                'type': 'mobile',
                'description': 'Lightweight model for edge deployment',
                'version': '1.0'
            }
        }
        
        for model_file in self.models_dir.glob("*.pth"):
            model_name = model_file.stem
            
            # Get model configuration
            model_config = model_configs.get(model_name, {
                'type': 'custom',
                'description': f'Custom model: {model_name}',
                'version': '1.0'
            })
            
            model_info = ModelInfo(
                name=model_name,
                path=str(model_file),
                model_type=model_config['type'],
                loaded=False,
                load_time=0.0,
                memory_usage=0,
                performance_stats={},
                last_used=0.0,
                version=model_config['version'],
                description=model_config['description']
            )
            
            self.available_models[model_name] = model_info
            
        logger.info(f"Discovered {len(self.available_models)} models")
    
    async def load_model(self, model_name: str) -> LostObjectsDetector:
        """
        Load model by name
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded detector instance
        """
        # Check cache first
        cached_model = self.cache.get(model_name)
        if cached_model:
            logger.debug(f"Model {model_name} loaded from cache")
            return cached_model
        
        # Check if model exists
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_info = self.available_models[model_name]
        
        try:
            start_time = time.time()
            
            # Load model
            detector = LostObjectsDetector(
                model_path=model_info.path,
                config=config,
                device=str(self.device)
            )
            
            load_time = time.time() - start_time
            
            # Estimate memory usage
            memory_usage = self._estimate_model_memory(detector)
            
            # Update model info
            model_info.loaded = True
            model_info.load_time = load_time
            model_info.memory_usage = memory_usage
            model_info.last_used = time.time()
            
            # Add to cache
            self.cache.put(model_name, detector, model_info)
            
            # Update performance stats
            self.performance_stats['model_load_times'][model_name] = load_time
            
            logger.info(f"Model {model_name} loaded in {load_time:.2f}s, {memory_usage}MB")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.performance_stats['error_count'] += 1
            raise
    
    def _estimate_model_memory(self, detector: LostObjectsDetector) -> int:
        """Estimate model memory usage in MB"""
        try:
            total_params = sum(p.numel() * p.element_size() for p in detector.model.parameters())
            # Add overhead for activations and gradients
            estimated_mb = int((total_params * 3) / (1024 * 1024))  # 3x for params + gradients + activations
            return max(estimated_mb, 100)  # Minimum 100MB
        except:
            return 200  # Default estimate
    
    async def get_model(self, model_name: Optional[str] = None) -> LostObjectsDetector:
        """
        Get model instance (load if necessary)
        
        Args:
            model_name: Model name (default: default model)
            
        Returns:
            Detector instance
        """
        if model_name is None:
            model_name = self.default_model_name
        
        return await self.load_model(model_name)
    
    async def unload_model(self, model_name: str):
        """Unload model from cache"""
        self.cache.remove(model_name)
        if model_name in self.available_models:
            self.available_models[model_name].loaded = False
        logger.info(f"Model {model_name} unloaded")
    
    async def predict(
        self,
        image: Union[torch.Tensor, list],
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Run prediction using specified model
        
        Args:
            image: Input image(s)
            model_name: Model to use (default: default model)
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        try:
            detector = await self.get_model(model_name)
            
            # Run prediction
            if hasattr(image, 'shape') and len(image.shape) == 4:
                # Batch processing
                results = []
                for i in range(image.shape[0]):
                    result = detector.detect_objects(image[i], **kwargs)
                    results.append(result)
                return {'batch_results': results}
            else:
                # Single image
                result = detector.detect_objects(image, **kwargs)
                
                # Update performance stats
                inference_time = time.time() - start_time
                self.performance_stats['total_inferences'] += 1
                self.performance_stats['total_inference_time'] += inference_time
                
                return result
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            self.performance_stats['error_count'] += 1
            raise
    
    async def predict_lost_objects(
        self,
        image: torch.Tensor,
        model_name: Optional[str] = None,
        timestamp: Optional[float] = None,
        location: str = "",
        **kwargs
    ) -> Dict:
        """
        Predict lost objects with temporal tracking
        
        Args:
            image: Input image
            model_name: Model to use
            timestamp: Current timestamp
            location: Location identifier
            **kwargs: Additional parameters
            
        Returns:
            Lost objects detection results
        """
        try:
            detector = await self.get_model(model_name)
            
            result = detector.detect_lost_objects(
                image=image,
                timestamp=timestamp,
                location=location,
                **kwargs
            )
            
            # Update performance stats
            self.performance_stats['total_inferences'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Lost objects prediction failed: {e}")
            self.performance_stats['error_count'] += 1
            raise
    
    async def get_available_models(self) -> Dict[str, Dict]:
        """Get information about available models"""
        return {
            name: {
                'name': info.name,
                'type': info.model_type,
                'description': info.description,
                'version': info.version,
                'loaded': info.loaded,
                'memory_usage_mb': info.memory_usage,
                'load_time': info.load_time,
                'last_used': info.last_used
            }
            for name, info in self.available_models.items()
        }
    
    async def get_model_performance(self, model_name: str) -> Dict:
        """Get performance statistics for a specific model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_info = self.available_models[model_name]
        cached_model = self.cache.get(model_name)
        
        perf_stats = {}
        if cached_model:
            perf_stats = cached_model.get_stats()
        
        return {
            'model_name': model_name,
            'model_info': {
                'type': model_info.model_type,
                'version': model_info.version,
                'description': model_info.description
            },
            'performance': perf_stats,
            'load_time': model_info.load_time,
            'memory_usage_mb': model_info.memory_usage,
            'last_used': model_info.last_used
        }
    
    async def get_service_stats(self) -> Dict:
        """Get overall service statistics"""
        # System memory info
        memory = psutil.virtual_memory()
        
        # GPU info if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_available': True,
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            }
        
        # Calculate average inference time
        avg_inference_time = 0.0
        if self.performance_stats['total_inferences'] > 0:
            avg_inference_time = (
                self.performance_stats['total_inference_time'] / 
                self.performance_stats['total_inferences']
            )
        
        return {
            'service_info': {
                'device': str(self.device),
                'models_directory': str(self.models_dir),
                'default_model': self.default_model_name
            },
            'performance': {
                'total_inferences': self.performance_stats['total_inferences'],
                'average_inference_time': avg_inference_time,
                'error_count': self.performance_stats['error_count'],
                'model_load_times': self.performance_stats['model_load_times']
            },
            'cache': self.cache.get_cache_info(),
            'system': {
                'memory_usage_percent': memory.percent,
                'available_memory_gb': memory.available / 1024**3,
                **gpu_info
            },
            'models': {
                'total_available': len(self.available_models),
                'currently_loaded': len([m for m in self.available_models.values() if m.loaded])
            }
        }
    
    async def optimize_cache(self):
        """Optimize model cache performance"""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache optimized")
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        try:
            # Check if default model can be loaded
            detector = await self.get_model()
            
            # Test inference with dummy data
            dummy_image = torch.randn(3, 320, 320)
            _ = detector.detect_objects(dummy_image.numpy())
            
            return {
                'status': 'healthy',
                'default_model_loaded': True,
                'inference_test': 'passed',
                'cache_info': self.cache.get_cache_info()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'cache_info': self.cache.get_cache_info()
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        # Clear cache
        for model_name in list(self.cache.models.keys()):
            self.cache.remove(model_name)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ModelService cleanup complete")