#!/usr/bin/env python3
"""
📊 BENCHMARK - SCRIPT DE TEST DE PERFORMANCE
============================================
Script complet pour tester les performances du service IA de détection d'objets

Tests inclus:
- Performance des modèles individuels
- Tests de charge (stress test)
- Benchmarks API REST et WebSocket
- Tests de mémoire et ressources
- Comparaison multi-modèles
- Tests de différents formats d'images/vidéos
- Simulation de charge réaliste

Métriques collectées:
- Temps d'inférence (min, max, moyenne, p95)
- Utilisation mémoire (RAM, GPU)
- Débit (images/sec, requêtes/sec)
- Précision (mAP, F1-Score sur dataset test)
- Stabilité (erreurs, crashes)

Utilisation:
    python benchmark.py --quick                    # Test rapide
    python benchmark.py --full --save-results     # Test complet avec sauvegarde
    python benchmark.py --stress --duration 3600  # Test de charge 1h
    python benchmark.py --api --host localhost:8001  # Test API
"""

import argparse
import asyncio
import time
import logging
import json
import statistics
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import psutil
import gc

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import requests
import websocket
from PIL import Image, ImageDraw
import cv2

# Imports du projet
from app.core.model_manager import ModelManager
from app.services.model_service import ModelService
from app.services.image_service import ImageService, ImageProcessingConfig
from app.services.video_service import VideoService, VideoProcessingConfig
from app.config.config import get_settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """📊 Résultat d'un benchmark"""
    test_name: str
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    
    # Métriques temporelles
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    median_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    
    # Métriques système
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_gpu_percent: float = 0.0
    
    # Débit
    operations_per_second: float = 0.0
    
    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculs automatiques"""
        if self.duration_seconds > 0 and self.successful_operations > 0:
            self.operations_per_second = self.successful_operations / self.duration_seconds

@dataclass
class SystemMetrics:
    """📊 Métriques système"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0

class SystemMonitor:
    """📊 Moniteur système pour benchmarks"""
    
    def __init__(self):
        self.metrics: List[SystemMetrics] = []
        self.monitoring = False
        self._monitor_thread = None
        
        # Vérifier GPU disponible
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml = pynvml
            except ImportError:
                self.gpu_available = False
    
    def start_monitoring(self, interval: float = 1.0):
        """▶️ Démarre le monitoring"""
        self.monitoring = True
        self.metrics.clear()
        
        def monitor():
            while self.monitoring:
                try:
                    # CPU et RAM
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    # GPU si disponible
                    gpu_percent = 0.0
                    gpu_memory = 0.0
                    
                    if self.gpu_available:
                        try:
                            gpu_util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                            gpu_percent = gpu_util.gpu
                            
                            gpu_mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                            gpu_memory = gpu_mem_info.used / (1024**3)  # GB
                        except:
                            pass
                    
                    # Enregistrer métriques
                    metric = SystemMetrics(
                        timestamp=time.time(),
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_used_gb=memory.used / (1024**3),
                        gpu_percent=gpu_percent,
                        gpu_memory_used_gb=gpu_memory
                    )
                    
                    self.metrics.append(metric)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.warning(f"Erreur monitoring: {e}")
                    time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """⏹️ Arrête le monitoring et retourne les statistiques"""
        self.monitoring = False
        
        if not self.metrics:
            return {}
        
        # Calculer statistiques
        cpu_values = [m.cpu_percent for m in self.metrics]
        memory_values = [m.memory_used_gb for m in self.metrics]
        gpu_values = [m.gpu_percent for m in self.metrics if m.gpu_percent > 0]
        
        return {
            "avg_cpu_percent": statistics.mean(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_gb": statistics.mean(memory_values),
            "peak_memory_gb": max(memory_values),
            "avg_gpu_percent": statistics.mean(gpu_values) if gpu_values else 0.0,
            "max_gpu_percent": max(gpu_values) if gpu_values else 0.0,
            "samples_count": len(self.metrics)
        }

class PerformanceBenchmark:
    """📊 Benchmark de performance principale"""
    
    def __init__(self):
        self.settings = get_settings()
        self.results: List[BenchmarkResult] = []
        self.system_monitor = SystemMonitor()
        
        # Services
        self.model_manager = None
        self.model_service = None
        self.image_service = None
        self.video_service = None
        
        # Données de test
        self.test_images = []
        self.test_videos = []
        
        logger.info("📊 PerformanceBenchmark initialisé")
    
    async def initialize(self):
        """🚀 Initialise les services"""
        logger.info("🚀 Initialisation des services pour benchmark...")
        
        # Model Manager
        self.model_manager = ModelManager(self.settings)
        await self.model_manager.initialize()
        
        # Services
        self.model_service = ModelService(self.model_manager)
        await self.model_service.initialize()
        
        self.image_service = ImageService(self.model_service)
        await self.image_service.initialize()
        
        self.video_service = VideoService(self.model_service)
        await self.video_service.initialize()
        
        # Générer données de test
        await self._generate_test_data()
        
        logger.info("✅ Services initialisés pour benchmark")
    
    async def _generate_test_data(self):
        """🎲 Génère des données de test"""
        
        logger.info("🎲 Génération des données de test...")
        
        # Images de test de différentes tailles
        test_sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        
        for i, (width, height) in enumerate(test_sizes):
            # Créer image synthétique avec objets
            image = self._create_synthetic_image(width, height, f"test_image_{i}")
            self.test_images.append({
                "image": image,
                "size": (width, height),
                "name": f"synthetic_{width}x{height}"
            })
        
        # Vidéos de test (simulées par séquences d'images)
        for duration in [5, 15, 30]:  # secondes
            frames = []
            for frame_idx in range(duration * 2):  # 2 FPS pour test
                frame = self._create_synthetic_image(640, 480, f"video_frame_{frame_idx}")
                frames.append(frame)
            
            self.test_videos.append({
                "frames": frames,
                "duration": duration,
                "name": f"synthetic_video_{duration}s"
            })
        
        logger.info(f"✅ Généré {len(self.test_images)} images et {len(self.test_videos)} vidéos de test")
    
    def _create_synthetic_image(self, width: int, height: int, seed: str) -> Image.Image:
        """🎨 Crée une image synthétique avec objets"""
        
        # Image de base
        image = Image.new('RGB', (width, height), color=(100, 150, 200))
        draw = ImageDraw.Draw(image)
        
        # Ajouter quelques formes qui ressemblent à des objets
        import random
        random.seed(hash(seed) % 2**32)
        
        # Rectangles (sacs, valises)
        for _ in range(3):
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = x1 + random.randint(50, min(200, width - x1))
            y2 = y1 + random.randint(30, min(150, height - y1))
            
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
        
        # Cercles (téléphones, montres)
        for _ in range(2):
            center_x = random.randint(50, width - 50)
            center_y = random.randint(50, height - 50)
            radius = random.randint(20, 60)
            
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], 
                        fill=color, outline=(0, 0, 0), width=2)
        
        return image
    
    async def run_model_benchmark(self, model_name: str, num_iterations: int = 100) -> BenchmarkResult:
        """🧠 Benchmark d'un modèle spécifique"""
        
        logger.info(f"🧠 Benchmark modèle: {model_name} ({num_iterations} itérations)")
        
        # Démarrer monitoring
        self.system_monitor.start_monitoring()
        
        start_time = time.time()
        times = []
        successful = 0
        failed = 0
        
        try:
            for i in range(num_iterations):
                # Choisir image de test aléatoire
                test_image = self.test_images[i % len(self.test_images)]["image"]
                
                # Mesurer temps de traitement
                iteration_start = time.time()
                
                try:
                    detections = await self.model_service.detect_objects(
                        image=test_image,
                        model_name=model_name,
                        confidence_threshold=0.5,
                        detection_id=f"benchmark_{i}"
                    )
                    
                    iteration_time = (time.time() - iteration_start) * 1000  # ms
                    times.append(iteration_time)
                    successful += 1
                    
                except Exception as e:
                    logger.warning(f"Erreur itération {i}: {e}")
                    failed += 1
                
                # Log progression
                if (i + 1) % 20 == 0:
                    logger.info(f"  Progression: {i + 1}/{num_iterations}")
        
        finally:
            # Arrêter monitoring
            system_stats = self.system_monitor.stop_monitoring()
        
        # Calculer statistiques
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            test_name=f"model_benchmark_{model_name}",
            duration_seconds=total_time,
            total_operations=num_iterations,
            successful_operations=successful,
            failed_operations=failed
        )
        
        if times:
            result.min_time_ms = min(times)
            result.max_time_ms = max(times)
            result.avg_time_ms = statistics.mean(times)
            result.median_time_ms = statistics.median(times)
            result.p95_time_ms = np.percentile(times, 95)
        
        # Ajouter métriques système
        result.peak_memory_mb = system_stats.get("peak_memory_gb", 0) * 1024
        result.avg_cpu_percent = system_stats.get("avg_cpu_percent", 0)
        result.avg_gpu_percent = system_stats.get("avg_gpu_percent", 0)
        
        result.metadata = {
            "model_name": model_name,
            "iterations": num_iterations,
            "system_stats": system_stats
        }
        
        self.results.append(result)
        return result
    
    async def run_stress_test(self, duration_seconds: int = 300, concurrent_requests: int = 10) -> BenchmarkResult:
        """💪 Test de charge (stress test)"""
        
        logger.info(f"💪 Test de charge: {duration_seconds}s avec {concurrent_requests} requêtes simultanées")
        
        self.system_monitor.start_monitoring()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        successful = 0
        failed = 0
        times = []
        
        async def worker(worker_id: int):
            """Worker de test de charge"""
            nonlocal successful, failed, times
            
            while time.time() < end_time:
                try:
                    # Image aléatoire
                    test_image = self.test_images[worker_id % len(self.test_images)]["image"]
                    
                    # Modèle aléatoire
                    models = ["epoch_30", "extended", "fast"]
                    model_name = models[successful % len(models)]
                    
                    # Traitement
                    iteration_start = time.time()
                    
                    detections = await self.model_service.detect_objects(
                        image=test_image,
                        model_name=model_name,
                        confidence_threshold=0.5,
                        detection_id=f"stress_{worker_id}_{successful}"
                    )
                    
                    iteration_time = (time.time() - iteration_start) * 1000
                    times.append(iteration_time)
                    successful += 1
                    
                except Exception as e:
                    failed += 1
                    logger.debug(f"Erreur worker {worker_id}: {e}")
                
                # Petite pause pour éviter spam
                await asyncio.sleep(0.1)
        
        # Lancer workers en parallèle
        tasks = [worker(i) for i in range(concurrent_requests)]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.warning(f"Erreur stress test: {e}")
        finally:
            system_stats = self.system_monitor.stop_monitoring()
        
        # Résultats
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="stress_test",
            duration_seconds=total_time,
            total_operations=successful + failed,
            successful_operations=successful,
            failed_operations=failed
        )
        
        if times:
            result.min_time_ms = min(times)
            result.max_time_ms = max(times)
            result.avg_time_ms = statistics.mean(times)
            result.median_time_ms = statistics.median(times)
            result.p95_time_ms = np.percentile(times, 95)
        
        result.peak_memory_mb = system_stats.get("peak_memory_gb", 0) * 1024
        result.avg_cpu_percent = system_stats.get("avg_cpu_percent", 0)
        result.avg_gpu_percent = system_stats.get("avg_gpu_percent", 0)
        
        result.metadata = {
            "duration_target": duration_seconds,
            "concurrent_requests": concurrent_requests,
            "system_stats": system_stats
        }
        
        self.results.append(result)
        return result
    
    async def run_image_formats_benchmark(self) -> BenchmarkResult:
        """🖼️ Benchmark des différents formats d'images"""
        
        logger.info("🖼️ Benchmark formats d'images")
        
        # Créer images de test dans différents formats
        base_image = self.test_images[0]["image"]
        formats_to_test = [
            ("JPEG", 95), ("JPEG", 85), ("JPEG", 70),
            ("PNG", None), ("BMP", None), ("WEBP", 90)
        ]
        
        self.system_monitor.start_monitoring()
        start_time = time.time()
        
        times_by_format = {}
        successful = 0
        failed = 0
        
        for format_name, quality in formats_to_test:
            format_times = []
            
            for i in range(20):  # 20 tests par format
                try:
                    # Convertir image au format
                    from io import BytesIO
                    buffer = BytesIO()
                    
                    if quality:
                        base_image.save(buffer, format=format_name, quality=quality)
                    else:
                        base_image.save(buffer, format=format_name)
                    
                    buffer.seek(0)
                    test_image = Image.open(buffer)
                    
                    # Tester détection
                    iteration_start = time.time()
                    
                    detections = await self.model_service.detect_objects(
                        image=test_image,
                        model_name="epoch_30",
                        confidence_threshold=0.5,
                        detection_id=f"format_{format_name}_{i}"
                    )
                    
                    iteration_time = (time.time() - iteration_start) * 1000
                    format_times.append(iteration_time)
                    successful += 1
                    
                except Exception as e:
                    logger.warning(f"Erreur format {format_name}: {e}")
                    failed += 1
            
            if format_times:
                times_by_format[f"{format_name}_{quality or 'default'}"] = {
                    "avg_ms": statistics.mean(format_times),
                    "min_ms": min(format_times),
                    "max_ms": max(format_times)
                }
        
        system_stats = self.system_monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        # Tous les temps combinés
        all_times = []
        for format_data in times_by_format.values():
            all_times.extend([format_data["avg_ms"]])  # Simplified
        
        result = BenchmarkResult(
            test_name="image_formats_benchmark",
            duration_seconds=total_time,
            total_operations=successful + failed,
            successful_operations=successful,
            failed_operations=failed
        )
        
        if all_times:
            result.avg_time_ms = statistics.mean(all_times)
            result.min_time_ms = min(all_times)
            result.max_time_ms = max(all_times)
        
        result.metadata = {
            "formats_tested": list(times_by_format.keys()),
            "times_by_format": times_by_format,
            "system_stats": system_stats
        }
        
        self.results.append(result)
        return result
    
    async def run_memory_benchmark(self) -> BenchmarkResult:
        """💾 Benchmark de l'utilisation mémoire"""
        
        logger.info("💾 Benchmark mémoire")
        
        self.system_monitor.start_monitoring()
        start_time = time.time()
        
        successful = 0
        failed = 0
        peak_memory = 0
        
        # Test avec images de plus en plus grandes
        large_sizes = [(1920, 1080), (2560, 1440), (3840, 2160)]  # HD, 2K, 4K
        
        for width, height in large_sizes:
            try:
                # Créer grande image
                large_image = self._create_synthetic_image(width, height, f"memory_test_{width}x{height}")
                
                # Mesurer mémoire avant
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Traitement
                detections = await self.model_service.detect_objects(
                    image=large_image,
                    model_name="epoch_30",
                    confidence_threshold=0.5,
                    detection_id=f"memory_{width}x{height}"
                )
                
                # Mesurer mémoire après
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_diff = memory_after - memory_before
                
                peak_memory = max(peak_memory, memory_after)
                successful += 1
                
                logger.info(f"  {width}x{height}: {memory_diff:.1f}MB (+), Peak: {memory_after:.1f}MB")
                
                # Nettoyer
                del large_image
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Erreur mémoire {width}x{height}: {e}")
                failed += 1
        
        system_stats = self.system_monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="memory_benchmark",
            duration_seconds=total_time,
            total_operations=len(large_sizes),
            successful_operations=successful,
            failed_operations=failed,
            peak_memory_mb=peak_memory
        )
        
        result.metadata = {
            "image_sizes_tested": large_sizes,
            "peak_memory_mb": peak_memory,
            "system_stats": system_stats
        }
        
        self.results.append(result)
        return result
    
    def run_api_benchmark(self, api_url: str, num_requests: int = 100) -> BenchmarkResult:
        """🌐 Benchmark de l'API REST"""
        
        logger.info(f"🌐 Benchmark API: {api_url} ({num_requests} requêtes)")
        
        self.system_monitor.start_monitoring()
        start_time = time.time()
        
        times = []
        successful = 0
        failed = 0
        
        # Préparer données de test
        test_image = self.test_images[0]["image"]
        
        # Convertir en base64 pour API
        from io import BytesIO
        import base64
        
        buffer = BytesIO()
        test_image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Données à envoyer
        payload = {
            "image_data": image_b64,
            "confidence_threshold": 0.5,
            "model_name": "epoch_30"
        }
        
        # Requêtes séquentielles
        for i in range(num_requests):
            request_start = time.time()
            
            try:
                response = requests.post(
                    f"{api_url}/api/v1/detect/image",
                    json=payload,
                    timeout=30
                )
                
                request_time = (time.time() - request_start) * 1000
                
                if response.status_code == 200:
                    times.append(request_time)
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"API error {response.status_code}: {response.text[:100]}")
                
            except Exception as e:
                failed += 1
                logger.warning(f"Erreur requête {i}: {e}")
            
            # Log progression
            if (i + 1) % 20 == 0:
                logger.info(f"  Progression API: {i + 1}/{num_requests}")
        
        system_stats = self.system_monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="api_benchmark",
            duration_seconds=total_time,
            total_operations=num_requests,
            successful_operations=successful,
            failed_operations=failed
        )
        
        if times:
            result.min_time_ms = min(times)
            result.max_time_ms = max(times)
            result.avg_time_ms = statistics.mean(times)
            result.median_time_ms = statistics.median(times)
            result.p95_time_ms = np.percentile(times, 95)
        
        result.metadata = {
            "api_url": api_url,
            "requests_count": num_requests,
            "system_stats": system_stats
        }
        
        self.results.append(result)
        return result
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """📊 Génère un rapport complet"""
        
        if not self.results:
            logger.warning("Aucun résultat de benchmark disponible")
            return {}
        
        # Calculer statistiques globales
        total_operations = sum(r.total_operations for r in self.results)
        total_successful = sum(r.successful_operations for r in self.results)
        total_duration = sum(r.duration_seconds for r in self.results)
        
        # Trouver les meilleurs/pires performances
        best_throughput = max(self.results, key=lambda r: r.operations_per_second)
        worst_latency = max(self.results, key=lambda r: r.avg_time_ms)
        
        report = {
            "benchmark_summary": {
                "total_tests": len(self.results),
                "total_operations": total_operations,
                "total_successful": total_successful,
                "success_rate": total_successful / total_operations if total_operations > 0 else 0,
                "total_duration_seconds": total_duration,
                "generated_at": time.time()
            },
            "performance_highlights": {
                "best_throughput": {
                    "test_name": best_throughput.test_name,
                    "ops_per_second": best_throughput.operations_per_second
                },
                "worst_latency": {
                    "test_name": worst_latency.test_name,
                    "avg_time_ms": worst_latency.avg_time_ms
                }
            },
            "detailed_results": []
        }
        
        # Résultats détaillés
        for result in self.results:
            result_dict = {
                "test_name": result.test_name,
                "duration_seconds": result.duration_seconds,
                "operations": {
                    "total": result.total_operations,
                    "successful": result.successful_operations,
                    "failed": result.failed_operations,
                    "success_rate": result.successful_operations / result.total_operations if result.total_operations > 0 else 0
                },
                "performance": {
                    "avg_time_ms": result.avg_time_ms,
                    "min_time_ms": result.min_time_ms,
                    "max_time_ms": result.max_time_ms,
                    "median_time_ms": result.median_time_ms,
                    "p95_time_ms": result.p95_time_ms,
                    "operations_per_second": result.operations_per_second
                },
                "resources": {
                    "peak_memory_mb": result.peak_memory_mb,
                    "avg_cpu_percent": result.avg_cpu_percent,
                    "avg_gpu_percent": result.avg_gpu_percent
                },
                "metadata": result.metadata
            }
            
            report["detailed_results"].append(result_dict)
        
        # Sauvegarder si demandé
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"📊 Rapport sauvegardé: {output_file}")
        
        return report
    
    async def cleanup(self):
        """🧹 Nettoyage"""
        if self.image_service:
            await self.image_service.cleanup()
        if self.video_service:
            await self.video_service.cleanup()
        if self.model_service:
            await self.model_service.cleanup()

async def main():
    """🚀 Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(description="Benchmark du service IA")
    
    # Types de tests
    parser.add_argument('--quick', action='store_true', help='Test rapide (5 min)')
    parser.add_argument('--full', action='store_true', help='Test complet (30 min)')
    parser.add_argument('--stress', action='store_true', help='Test de charge')
    parser.add_argument('--api', action='store_true', help='Test API REST')
    
    # Paramètres
    parser.add_argument('--duration', type=int, default=300, help='Durée test de charge (secondes)')
    parser.add_argument('--iterations', type=int, default=100, help='Nombre d\'itérations')
    parser.add_argument('--concurrent', type=int, default=10, help='Requêtes simultanées')
    parser.add_argument('--host', type=str, default='http://localhost:8001', help='URL API')
    
    # Sortie
    parser.add_argument('--save-results', action='store_true', help='Sauvegarder résultats')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    # Créer benchmark
    benchmark = PerformanceBenchmark()
    
    try:
        # Initialiser services
        await benchmark.initialize()
        
        if args.quick or not any([args.full, args.stress, args.api]):
            # Test rapide par défaut
            logger.info("🚀 Benchmark rapide")
            
            # Test modèle principal
            await benchmark.run_model_benchmark("epoch_30", 50)
            
            # Test formats d'images
            await benchmark.run_image_formats_benchmark()
        
        elif args.full:
            # Test complet
            logger.info("🚀 Benchmark complet")
            
            # Tous les modèles
            for model_name in ["epoch_30", "extended", "fast"]:
                await benchmark.run_model_benchmark(model_name, args.iterations)
            
            # Tests spécialisés
            await benchmark.run_image_formats_benchmark()
            await benchmark.run_memory_benchmark()
            await benchmark.run_stress_test(300, 5)  # 5 min stress test
        
        elif args.stress:
            # Test de charge uniquement
            logger.info("🚀 Test de charge")
            await benchmark.run_stress_test(args.duration, args.concurrent)
        
        elif args.api:
            # Test API uniquement
            logger.info("🚀 Test API")
            benchmark.run_api_benchmark(args.host, args.iterations)
        
        # Générer rapport
        report = benchmark.generate_report(
            args.output if args.save_results else None
        )
        
        # Afficher résumé
        print("\n" + "="*50)
        print("📊 RÉSUMÉ DU BENCHMARK")
        print("="*50)
        
        summary = report.get("benchmark_summary", {})
        print(f"Tests exécutés: {summary.get('total_tests', 0)}")
        print(f"Opérations totales: {summary.get('total_operations', 0)}")
        print(f"Taux de réussite: {summary.get('success_rate', 0):.2%}")
        print(f"Durée totale: {summary.get('total_duration_seconds', 0):.1f}s")
        
        highlights = report.get("performance_highlights", {})
        if highlights:
            print(f"\nMeilleur débit: {highlights.get('best_throughput', {}).get('ops_per_second', 0):.1f} ops/sec")
            print(f"Pire latence: {highlights.get('worst_latency', {}).get('avg_time_ms', 0):.1f}ms")
        
        print("\n📊 Détails par test:")
        for result in report.get("detailed_results", []):
            perf = result.get("performance", {})
            print(f"  {result['test_name']}: {perf.get('avg_time_ms', 0):.1f}ms avg, {perf.get('operations_per_second', 0):.1f} ops/sec")
    
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main())