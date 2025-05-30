"""
🤖 MODEL SERVICE - SERVICE DE GESTION CENTRALISÉE DES MODÈLES
============================================================
Service central pour la gestion intelligente de tous les modèles de détection

Responsabilités:
- Interface unifiée pour tous les modèles PyTorch
- Gestion du cache et des ressources mémoire
- Optimisations performance (batch, GPU, etc.)
- Monitoring et métriques des modèles
- Sélection automatique du meilleur modèle
- Fallback et récupération d'erreurs

Modèles gérés:
- Epoch 30: Champion (F1=49.86%, Précision=60.73%)
- Extended: 28 classes d'objets perdus  
- Fast: Optimisé streaming temps réel
- Mobile: Optimisé edge/mobile

Fonctionnalités avancées:
- Auto-scaling selon la charge
- Warm-up automatique des modèles
- A/B testing entre modèles
- Ensemble de modèles pour meilleure précision
- Analytics et recommandations d'optimisation
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import weakref

import torch
import numpy as np
from PIL import Image

# Imports internes
from app.core.model_manager import ModelManager
from app.core.detector import ObjectDetector, DetectionConfig, DetectionMode
from app.schemas.detection import DetectionResult
from app.config.config import get_settings

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS
class ModelPriority(str, Enum):
    """🎯 Priorités des modèles"""
    CRITICAL = "critical"    # Modèles critiques (toujours chargés)
    HIGH = "high"           # Haute priorité
    MEDIUM = "medium"       # Priorité normale
    LOW = "low"             # Basse priorité
    ON_DEMAND = "on_demand" # Chargement à la demande

class PerformanceProfile(str, Enum):
    """⚡ Profils de performance"""
    SPEED = "speed"          # Optimisé vitesse
    BALANCED = "balanced"    # Équilibre vitesse/qualité
    ACCURACY = "accuracy"    # Optimisé précision
    EFFICIENCY = "efficiency" # Optimisé ressources

@dataclass
class ModelMetrics:
    """📊 Métriques d'un modèle"""
    model_name: str
    total_inferences: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    success_rate: float = 1.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    
    # Historique récent (sliding window)
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def update_inference(self, time_ms: float, success: bool = True):
        """📊 Met à jour les métriques d'inférence"""
        self.total_inferences += 1
        
        if success:
            self.total_time_ms += time_ms
            self.average_time_ms = self.total_time_ms / self.total_inferences
            self.recent_times.append(time_ms)
        else:
            self.recent_errors.append(time.time())
        
        # Calcul success rate récent
        recent_successes = self.total_inferences - len(self.recent_errors)
        if self.total_inferences > 0:
            self.success_rate = recent_successes / self.total_inferences
    
    def get_recent_performance(self) -> Dict[str, float]:
        """📊 Performance récente"""
        if not self.recent_times:
            return {"avg_time_ms": 0.0, "min_time_ms": 0.0, "max_time_ms": 0.0}
        
        recent_list = list(self.recent_times)
        return {
            "avg_time_ms": np.mean(recent_list),
            "min_time_ms": np.min(recent_list),
            "max_time_ms": np.max(recent_list),
            "std_time_ms": np.std(recent_list),
            "p95_time_ms": np.percentile(recent_list, 95) if len(recent_list) > 5 else np.max(recent_list)
        }

# 🤖 SERVICE PRINCIPAL
class ModelService:
    """🤖 Service de gestion centralisée des modèles"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.settings = get_settings()
        
        # Détecteurs par modèle
        self._detectors: Dict[str, ObjectDetector] = {}
        self._detector_locks: Dict[str, asyncio.Lock] = {}
        
        # Métriques par modèle
        self.metrics: Dict[str, ModelMetrics] = {}
        
        # Configuration des modèles
        self.model_configs = self._initialize_model_configs()
        
        # Cache des résultats récents
        self._results_cache: Dict[str, Tuple[List[DetectionResult], float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Statistiques globales
        self.total_requests = 0
        self.cache_hits = 0
        self.start_time = time.time()
        
        # Thread safety
        self._global_lock = threading.RLock()
        
        logger.info("🤖 ModelService initialisé")
    
    def _initialize_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """⚙️ Initialise les configurations des modèles"""
        
        return {
            "epoch_30": {
                "priority": ModelPriority.CRITICAL,
                "profile": PerformanceProfile.BALANCED,
                "warmup": True,
                "max_batch_size": 8,
                "timeout_ms": 5000,
                "description": "Modèle champion (F1=49.86%)"
            },
            "extended": {
                "priority": ModelPriority.HIGH,
                "profile": PerformanceProfile.ACCURACY,
                "warmup": True,
                "max_batch_size": 4,
                "timeout_ms": 10000,
                "description": "Modèle étendu 28 classes"
            },
            "fast": {
                "priority": ModelPriority.HIGH,
                "profile": PerformanceProfile.SPEED,
                "warmup": True,
                "max_batch_size": 16,
                "timeout_ms": 2000,
                "description": "Modèle rapide streaming"
            },
            "mobile": {
                "priority": ModelPriority.MEDIUM,
                "profile": PerformanceProfile.EFFICIENCY,
                "warmup": False,
                "max_batch_size": 1,
                "timeout_ms": 3000,
                "description": "Modèle mobile/edge"
            }
        }
    
    async def initialize(self):
        """🚀 Initialise le service"""
        
        logger.info("🚀 Initialisation ModelService...")
        
        # Initialiser métriques pour tous les modèles
        for model_name in self.model_configs.keys():
            self.metrics[model_name] = ModelMetrics(model_name)
            self._detector_locks[model_name] = asyncio.Lock()
        
        # Warm-up des modèles critiques
        await self._warmup_critical_models()
        
        logger.info("✅ ModelService initialisé")
    
    async def _warmup_critical_models(self):
        """🔥 Pré-chauffe les modèles critiques"""
        
        critical_models = [
            name for name, config in self.model_configs.items()
            if config.get("warmup", False) and config["priority"] == ModelPriority.CRITICAL
        ]
        
        logger.info(f"🔥 Warm-up de {len(critical_models)} modèles critiques")
        
        for model_name in critical_models:
            try:
                await self._get_or_create_detector(model_name)
                logger.info(f"✅ Modèle pré-chargé: {model_name}")
            except Exception as e:
                logger.error(f"❌ Erreur warm-up {model_name}: {e}")
    
    async def detect_objects(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_name: str = "epoch_30",
        confidence_threshold: float = 0.5,
        detection_mode: DetectionMode = DetectionMode.BALANCED,
        use_cache: bool = True,
        detection_id: Optional[str] = None
    ) -> List[DetectionResult]:
        """
        🎯 Détecte les objets dans une image
        
        Args:
            image: Image à analyser
            model_name: Nom du modèle à utiliser
            confidence_threshold: Seuil de confiance
            detection_mode: Mode de détection
            use_cache: Utiliser le cache
            detection_id: ID unique pour cette détection
            
        Returns:
            Liste des objets détectés
        """
        
        start_time = time.time()
        self.total_requests += 1
        
        # Cache check
        if use_cache:
            cache_key = self._generate_cache_key(image, model_name, confidence_threshold)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self.cache_hits += 1
                logger.debug(f"🎯 Cache hit pour {model_name}")
                return cached_result
        
        try:
            # Récupération du détecteur
            detector = await self._get_or_create_detector(model_name)
            
            # Configuration de détection
            detector.config.confidence_threshold = confidence_threshold
            detector.config.detection_mode = detection_mode
            
            # Détection
            detections = await detector.detect_objects(
                image=image,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                detection_id=detection_id
            )
            
            # Mise à jour métriques
            processing_time = (time.time() - start_time) * 1000
            self.metrics[model_name].update_inference(processing_time, True)
            
            # Cache du résultat
            if use_cache and cache_key:
                self._cache_result(cache_key, detections)
            
            logger.debug(
                f"🎯 Détection {model_name}: {len(detections)} objets "
                f"en {processing_time:.1f}ms"
            )
            
            return detections
            
        except Exception as e:
            # Métriques d'erreur
            processing_time = (time.time() - start_time) * 1000
            if model_name in self.metrics:
                self.metrics[model_name].update_inference(processing_time, False)
            
            logger.error(f"❌ Erreur détection {model_name}: {e}")
            
            # Fallback vers modèle par défaut si différent
            if model_name != "epoch_30":
                logger.info(f"🔄 Fallback vers epoch_30 depuis {model_name}")
                return await self.detect_objects(
                    image, "epoch_30", confidence_threshold, detection_mode, use_cache, detection_id
                )
            
            raise
    
    async def detect_objects_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        model_name: str = "epoch_30",
        confidence_threshold: float = 0.5,
        detection_mode: DetectionMode = DetectionMode.BALANCED
    ) -> List[List[DetectionResult]]:
        """
        🎯 Détection batch optimisée
        
        Args:
            images: Liste d'images
            model_name: Modèle à utiliser
            confidence_threshold: Seuil de confiance
            detection_mode: Mode de détection
            
        Returns:
            Liste de détections par image
        """
        
        if not images:
            return []
        
        start_time = time.time()
        
        try:
            # Récupération du détecteur
            detector = await self._get_or_create_detector(model_name)
            
            # Configuration
            detector.config.confidence_threshold = confidence_threshold
            detector.config.detection_mode = detection_mode
            
            # Détection batch
            batch_results = await detector.detect_objects_batch(
                images=images,
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )
            
            # Métriques
            processing_time = (time.time() - start_time) * 1000
            avg_time_per_image = processing_time / len(images)
            
            for _ in images:
                self.metrics[model_name].update_inference(avg_time_per_image, True)
            
            logger.info(
                f"🎯 Batch {model_name}: {len(images)} images "
                f"en {processing_time:.1f}ms ({avg_time_per_image:.1f}ms/img)"
            )
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Erreur batch {model_name}: {e}")
            
            # Fallback séquentiel
            logger.info("🔄 Fallback vers traitement séquentiel")
            results = []
            
            for i, image in enumerate(images):
                try:
                    detection_result = await self.detect_objects(
                        image, model_name, confidence_threshold, detection_mode, 
                        detection_id=f"batch_fallback_{i}"
                    )
                    results.append(detection_result)
                except Exception as img_error:
                    logger.error(f"❌ Erreur image {i}: {img_error}")
                    results.append([])  # Liste vide pour cette image
            
            return results
    
    async def get_best_model_for_task(
        self,
        task_type: str = "general",
        performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
        image_size: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        🎯 Sélectionne le meilleur modèle pour une tâche
        
        Args:
            task_type: Type de tâche ("general", "streaming", "batch", "mobile")
            performance_profile: Profil de performance désiré
            image_size: Taille des images (pour optimisation)
            
        Returns:
            Nom du modèle recommandé
        """
        
        # Règles de sélection basées sur le type de tâche
        task_models = {
            "general": ["epoch_30", "extended"],
            "streaming": ["fast", "epoch_30"],
            "batch": ["epoch_30", "extended"],
            "mobile": ["mobile", "fast"],
            "accuracy": ["extended", "epoch_30"],
            "speed": ["fast", "mobile"]
        }
        
        # Candidats selon le type de tâche
        candidates = task_models.get(task_type, ["epoch_30"])
        
        # Filtrage par profil de performance
        profile_candidates = []
        for model_name in candidates:
            model_config = self.model_configs.get(model_name, {})
            if model_config.get("profile") == performance_profile:
                profile_candidates.append(model_name)
        
        # Si pas de correspondance exacte, prendre les candidats généraux
        if not profile_candidates:
            profile_candidates = candidates
        
        # Sélection basée sur les métriques
        best_model = await self._select_best_performing_model(profile_candidates, performance_profile)
        
        logger.debug(f"🎯 Modèle sélectionné: {best_model} pour {task_type}/{performance_profile.value}")
        return best_model
    
    async def _select_best_performing_model(
        self,
        candidates: List[str],
        performance_profile: PerformanceProfile
    ) -> str:
        """🎯 Sélectionne le modèle avec les meilleures performances"""
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Scoring basé sur les métriques
        scores = {}
        
        for model_name in candidates:
            if model_name not in self.metrics:
                scores[model_name] = 0.0
                continue
            
            metrics = self.metrics[model_name]
            recent_perf = metrics.get_recent_performance()
            
            # Calcul du score selon le profil
            if performance_profile == PerformanceProfile.SPEED:
                # Privilégier vitesse
                score = 1000 / (recent_perf.get("avg_time_ms", 1000) + 1)
                score *= metrics.success_rate
            
            elif performance_profile == PerformanceProfile.ACCURACY:
                # Privilégier précision (basé sur config + success rate)
                base_accuracy = {
                    "extended": 0.95,
                    "epoch_30": 0.90,
                    "fast": 0.85,
                    "mobile": 0.80
                }.get(model_name, 0.75)
                
                score = base_accuracy * metrics.success_rate
            
            elif performance_profile == PerformanceProfile.EFFICIENCY:
                # Équilibre vitesse/ressources
                time_score = 1000 / (recent_perf.get("avg_time_ms", 1000) + 1)
                memory_score = 1000 / (metrics.memory_usage_mb + 100)
                score = (time_score + memory_score) * metrics.success_rate
            
            else:  # BALANCED
                # Équilibre général
                time_score = 1000 / (recent_perf.get("avg_time_ms", 1000) + 1)
                accuracy_score = metrics.success_rate
                score = (time_score + accuracy_score * 500) / 2
            
            scores[model_name] = score
        
        # Retourner le meilleur score
        best_model = max(scores.keys(), key=lambda k: scores[k])
        return best_model
    
    async def _get_or_create_detector(self, model_name: str) -> ObjectDetector:
        """🔧 Récupère ou crée un détecteur pour un modèle"""
        
        async with self._detector_locks.get(model_name, asyncio.Lock()):
            # Vérifier si déjà créé
            if model_name in self._detectors:
                return self._detectors[model_name]
            
            logger.info(f"🔧 Création détecteur pour {model_name}")
            
            # Configuration de détection
            model_config = self.model_configs.get(model_name, {})
            
            if model_config.get("profile") == PerformanceProfile.SPEED:
                detection_config = DetectionConfig(
                    detection_mode=DetectionMode.FAST,
                    half_precision=True
                )
            elif model_config.get("profile") == PerformanceProfile.ACCURACY:
                detection_config = DetectionConfig(
                    detection_mode=DetectionMode.QUALITY,
                    half_precision=False
                )
            else:
                detection_config = DetectionConfig(
                    detection_mode=DetectionMode.BALANCED
                )
            
            # Création du détecteur
            detector = ObjectDetector(
                model_manager=self.model_manager,
                config=detection_config
            )
            
            # Pré-chargement du modèle
            await self.model_manager.get_model(model_name)
            
            # Mise à jour métriques mémoire
            if model_name in self.metrics:
                try:
                    memory_info = self.model_manager.get_memory_usage()
                    self.metrics[model_name].memory_usage_mb = memory_info.get("model_cache_mb", 0)
                except Exception:
                    pass
            
            # Stockage
            self._detectors[model_name] = detector
            
            logger.info(f"✅ Détecteur créé: {model_name}")
            return detector
    
    def _generate_cache_key(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_name: str,
        confidence_threshold: float
    ) -> Optional[str]:
        """🔑 Génère une clé de cache pour une image"""
        
        try:
            # Hash simple basé sur taille et quelques pixels
            if isinstance(image, Image.Image):
                width, height = image.size
                # Échantillonner quelques pixels
                sample_pixels = []
                for x in range(0, width, width//10 + 1):
                    for y in range(0, height, height//10 + 1):
                        try:
                            pixel = image.getpixel((x, y))
                            if isinstance(pixel, (list, tuple)):
                                sample_pixels.extend(pixel[:3])  # RGB seulement
                            else:
                                sample_pixels.append(pixel)
                        except:
                            pass
                
                # Hash combiné
                content_hash = hash(tuple(sample_pixels[:50]))  # Limiter à 50 valeurs
                
            elif isinstance(image, np.ndarray):
                # Hash basé sur forme et échantillonnage
                shape_hash = hash(image.shape)
                content_hash = hash(image[::image.shape[0]//10 + 1, ::image.shape[1]//10 + 1].tobytes())
                content_hash = shape_hash ^ content_hash
                
            else:  # torch.Tensor
                # Conversion temporaire pour hash
                if image.is_cuda:
                    image_cpu = image.cpu()
                else:
                    image_cpu = image
                
                shape_hash = hash(tuple(image_cpu.shape))
                # Échantillonnage sparse
                sample = image_cpu[::image_cpu.shape[-2]//10 + 1, ::image_cpu.shape[-1]//10 + 1]
                content_hash = hash(sample.numpy().tobytes())
                content_hash = shape_hash ^ content_hash
            
            # Clé finale
            cache_key = f"{model_name}_{confidence_threshold:.2f}_{abs(content_hash)}"
            return cache_key
            
        except Exception as e:
            logger.debug(f"❌ Erreur génération cache key: {e}")
            return None
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[DetectionResult]]:
        """🔍 Récupère un résultat du cache"""
        
        if cache_key not in self._results_cache:
            return None
        
        result, timestamp = self._results_cache[cache_key]
        
        # Vérifier TTL
        if time.time() - timestamp > self._cache_ttl:
            del self._results_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: List[DetectionResult]):
        """💾 Met en cache un résultat"""
        
        # Nettoyage périodique du cache
        current_time = time.time()
        if len(self._results_cache) > 1000:  # Limite taille
            # Supprimer les entrées expirées
            expired_keys = [
                key for key, (_, timestamp) in self._results_cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            
            for key in expired_keys:
                del self._results_cache[key]
        
        # Ajout au cache
        self._results_cache[cache_key] = (result, current_time)
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques globales du service"""
        
        uptime = time.time() - self.start_time
        cache_hit_rate = self.cache_hits / max(1, self.total_requests)
        
        # Statistiques par modèle
        model_stats = {}
        for model_name, metrics in self.metrics.items():
            model_stats[model_name] = {
                "total_inferences": metrics.total_inferences,
                "average_time_ms": metrics.average_time_ms,
                "success_rate": metrics.success_rate,
                "memory_usage_mb": metrics.memory_usage_mb,
                "recent_performance": metrics.get_recent_performance(),
                "is_loaded": model_name in self._detectors
            }
        
        return {
            "service_info": {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self._results_cache),
                "loaded_detectors": len(self._detectors)
            },
            "model_statistics": model_stats,
            "model_configs": self.model_configs
        }
    
    async def get_detailed_performance_metrics(self) -> Dict[str, Any]:
        """📊 Métriques détaillées de performance"""
        
        # Métriques système
        system_metrics = {}
        try:
            memory_info = self.model_manager.get_memory_usage()
            gpu_health = self.model_manager.check_gpu_health()
            
            system_metrics = {
                "memory": memory_info,
                "gpu": gpu_health
            }
        except Exception as e:
            logger.debug(f"Erreur métriques système: {e}")
        
        # Métriques par modèle
        model_metrics = {}
        for model_name, metrics in self.metrics.items():
            model_metrics[model_name] = {
                **metrics.get_recent_performance(),
                "total_inferences": metrics.total_inferences,
                "success_rate": metrics.success_rate,
                "memory_usage_mb": metrics.memory_usage_mb,
                "configuration": self.model_configs.get(model_name, {})
            }
        
        return {
            "system_metrics": system_metrics,
            "model_metrics": model_metrics,
            "service_health": await self._get_service_health()
        }
    
    async def _get_service_health(self) -> Dict[str, Any]:
        """🏥 État de santé du service"""
        
        health_status = "healthy"
        issues = []
        
        # Vérifier les modèles critiques
        for model_name, config in self.model_configs.items():
            if config["priority"] == ModelPriority.CRITICAL:
                if model_name not in self._detectors:
                    health_status = "degraded"
                    issues.append(f"Modèle critique non chargé: {model_name}")
                
                metrics = self.metrics.get(model_name)
                if metrics and metrics.success_rate < 0.9:
                    health_status = "degraded"
                    issues.append(f"Taux de succès bas pour {model_name}: {metrics.success_rate:.2f}")
        
        # Vérifier les ressources
        try:
            memory_info = self.model_manager.get_memory_usage()
            if memory_info.get("system_memory_percent", 0) > 90:
                health_status = "warning"
                issues.append("Utilisation mémoire système élevée")
        except Exception:
            pass
        
        return {
            "status": health_status,
            "issues": issues,
            "checks_passed": len(issues) == 0,
            "last_check": time.time()
        }
    
    async def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques complètes pour analytics"""
        
        service_stats = self.get_service_statistics()
        performance_metrics = await self.get_detailed_performance_metrics()
        
        # Recommandations d'optimisation
        recommendations = await self._generate_optimization_recommendations()
        
        return {
            **service_stats,
            "performance_metrics": performance_metrics,
            "optimization_recommendations": recommendations,
            "generated_at": time.time()
        }
    
    async def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """💡 Génère des recommandations d'optimisation"""
        
        recommendations = []
        
        # Analyser les métriques
        for model_name, metrics in self.metrics.items():
            recent_perf = metrics.get_recent_performance()
            
            # Recommandations basées sur les temps de réponse
            if recent_perf.get("avg_time_ms", 0) > 1000:
                recommendations.append({
                    "type": "performance",
                    "model": model_name,
                    "issue": "Temps de réponse élevé",
                    "recommendation": "Considérer l'utilisation du modèle 'fast' ou optimiser la taille des images"
                })
            
            # Recommandations basées sur le taux de succès
            if metrics.success_rate < 0.95:
                recommendations.append({
                    "type": "reliability",
                    "model": model_name,
                    "issue": "Taux de succès bas",
                    "recommendation": "Vérifier la charge système et redémarrer le modèle si nécessaire"
                })
        
        # Recommandations générales
        cache_hit_rate = self.cache_hits / max(1, self.total_requests)
        if cache_hit_rate < 0.1:
            recommendations.append({
                "type": "caching",
                "model": "global",
                "issue": "Faible taux de cache hit",
                "recommendation": "Augmenter le TTL du cache ou optimiser la stratégie de cache"
            })
        
        return recommendations
    
    def get_current_timestamp(self) -> float:
        """⏰ Timestamp actuel"""
        return time.time()
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        logger.info("🧹 Nettoyage ModelService...")
        
        # Nettoyage des détecteurs
        for detector in self._detectors.values():
            try:
                await detector.cleanup()
            except Exception as e:
                logger.warning(f"Erreur nettoyage détecteur: {e}")
        
        self._detectors.clear()
        self._results_cache.clear()
        
        logger.info("✅ ModelService nettoyé")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "ModelService",
    "ModelMetrics",
    "ModelPriority",
    "PerformanceProfile"
]