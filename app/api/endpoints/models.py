# app/api/endpoints/models.py
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Optional
import logging
from datetime import datetime

from app.schemas.detection import ModelInfo, HealthStatus, ServiceStats
from app.core.model_manager import ModelManager

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_model_manager(request: Request) -> ModelManager:
    """Récupère le gestionnaire de modèles"""
    return request.app.state.model_manager

@router.get("/", response_model=List[Dict])
async def list_available_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    📋 Liste tous les modèles disponibles
    
    Retourne la liste des modèles avec leurs informations:
    - Nom et description
    - État de chargement
    - Performance
    - Usage mémoire
    """
    
    try:
        models_info = []
        
        for name, model_info in model_manager.available_models.items():
            info = {
                "name": name,
                "description": model_info.config.get('description', 'Modèle de détection'),
                "file_path": str(model_info.path),
                "is_loaded": model_info.is_loaded,
                "performance": model_info.config.get('performance', 'unknown'),
                "speed": model_info.config.get('speed', 'unknown'),
                "memory_usage_mb": model_info.memory_usage,
                "inference_count": model_info.inference_count,
                "last_used": model_info.last_used.isoformat() if model_info.last_used else None,
                "load_time": model_info.load_time.isoformat() if model_info.load_time else None,
                "file_exists": model_info.path.exists()
            }
            
            # Calcul du temps d'inférence moyen
            if model_info.inference_count > 0:
                avg_time = model_info.total_inference_time / model_info.inference_count
                info["avg_inference_time_ms"] = avg_time
            else:
                info["avg_inference_time_ms"] = None
            
            models_info.append(info)
        
        # Tri par nom
        models_info.sort(key=lambda x: x['name'])
        
        return models_info
        
    except Exception as e:
        logger.error(f"Erreur listage modèles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des modèles: {str(e)}"
        )

@router.get("/{model_name}")
async def get_model_info(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🔍 Récupère les informations détaillées d'un modèle
    
    Informations incluses:
    - Configuration technique
    - Statistiques d'usage
    - Performance
    - État de chargement
    """
    
    if model_name not in model_manager.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' non trouvé"
        )
    
    try:
        model_info = model_manager.available_models[model_name]
        
        # Informations détaillées
        detailed_info = {
            "name": model_name,
            "file_path": str(model_info.path),
            "file_exists": model_info.path.exists(),
            "file_size_mb": model_info.path.stat().st_size / (1024*1024) if model_info.path.exists() else 0,
            "config": model_info.config,
            "is_loaded": model_info.is_loaded,
            "memory_usage_mb": model_info.memory_usage,
            "load_time": model_info.load_time.isoformat() if model_info.load_time else None,
            "last_used": model_info.last_used.isoformat() if model_info.last_used else None,
            "usage_stats": {
                "inference_count": model_info.inference_count,
                "total_inference_time": model_info.total_inference_time,
                "avg_inference_time_ms": (
                    model_info.total_inference_time / model_info.inference_count
                    if model_info.inference_count > 0 else 0
                )
            }
        }
        
        # Si le modèle est chargé, ajouter des infos sur le modèle PyTorch
        if model_info.is_loaded and model_info.model:
            try:
                param_count = sum(p.numel() for p in model_info.model.parameters())
                detailed_info["model_stats"] = {
                    "parameter_count": param_count,
                    "parameter_count_readable": f"{param_count/1e6:.1f}M" if param_count > 1e6 else f"{param_count/1e3:.1f}K",
                    "device": str(next(model_info.model.parameters()).device),
                    "dtype": str(next(model_info.model.parameters()).dtype)
                }
            except Exception as e:
                logger.warning(f"Impossible de récupérer les stats du modèle {model_name}: {e}")
        
        return detailed_info
        
    except Exception as e:
        logger.error(f"Erreur récupération info modèle {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations: {str(e)}"
        )

@router.post("/{model_name}/load")
async def load_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🔄 Charge un modèle en mémoire
    
    Force le chargement d'un modèle spécifique en mémoire.
    Utile pour le pré-chargement ou le changement de modèle.
    """
    
    if model_name not in model_manager.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' non trouvé"
        )
    
    try:
        start_time = datetime.now()
        
        # Chargement du modèle
        model_info = await model_manager.load_model(model_name)
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modèle {model_name} chargé manuellement en {load_time:.2f}s")
        
        return {
            "success": True,
            "message": f"Modèle '{model_name}' chargé avec succès",
            "model_name": model_name,
            "load_time_seconds": load_time,
            "memory_usage_mb": model_info.memory_usage,
            "is_loaded": model_info.is_loaded
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erreur chargement modèle {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement: {str(e)}"
        )

@router.post("/{model_name}/unload")
async def unload_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🗑️ Décharge un modèle de la mémoire
    
    Libère la mémoire occupée par un modèle.
    Le modèle pourra être rechargé automatiquement si nécessaire.
    """
    
    if model_name not in model_manager.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' non trouvé"
        )
    
    try:
        model_info = model_manager.available_models[model_name]
        
        if not model_info.is_loaded:
            return {
                "success": True,
                "message": f"Modèle '{model_name}' déjà déchargé",
                "was_loaded": False
            }
        
        # Déchargement via le cache LRU
        if model_name in model_manager.models_cache.cache:
            model_manager.models_cache._unload_model(model_info)
            del model_manager.models_cache.cache[model_name]
        
        logger.info(f"Modèle {model_name} déchargé manuellement")
        
        return {
            "success": True,
            "message": f"Modèle '{model_name}' déchargé avec succès",
            "model_name": model_name,
            "was_loaded": True
        }
        
    except Exception as e:
        logger.error(f"Erreur déchargement modèle {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du déchargement: {str(e)}"
        )

@router.get("/cache/status")
async def get_cache_status(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🧠 Récupère l'état du cache des modèles
    
    Informations sur:
    - Modèles actuellement en cache
    - Usage mémoire
    - Statistiques d'accès
    """
    
    try:
        cache = model_manager.models_cache
        cache_info = []
        
        total_memory = 0
        for name, model_info in cache.cache.items():
            info = {
                "model_name": name,
                "is_loaded": model_info.is_loaded,
                "memory_usage_mb": model_info.memory_usage,
                "last_used": model_info.last_used.isoformat() if model_info.last_used else None,
                "inference_count": model_info.inference_count,
                "load_time": model_info.load_time.isoformat() if model_info.load_time else None
            }
            cache_info.append(info)
            total_memory += model_info.memory_usage
        
        # Tri par dernière utilisation
        cache_info.sort(key=lambda x: x['last_used'] or '', reverse=True)
        
        return {
            "cache_size": len(cache.cache),
            "max_cache_size": cache.max_size,
            "total_memory_usage_mb": total_memory,
            "models_in_cache": cache_info,
            "cache_stats": model_manager.stats
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération cache status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération du statut: {str(e)}"
        )

@router.post("/cache/clear")
async def clear_cache(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🧹 Vide le cache des modèles
    
    Décharge tous les modèles actuellement en mémoire.
    Utile pour libérer la mémoire ou forcer un rechargement.
    """
    
    try:
        cache = model_manager.models_cache
        models_count = len(cache.cache)
        
        # Déchargement de tous les modèles
        for model_info in list(cache.cache.values()):
            cache._unload_model(model_info)
        
        # Vidage du cache
        cache.cache.clear()
        
        logger.info(f"Cache vidé: {models_count} modèles déchargés")
        
        return {
            "success": True,
            "message": f"Cache vidé - {models_count} modèles déchargés",
            "models_unloaded": models_count
        }
        
    except Exception as e:
        logger.error(f"Erreur vidage cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du vidage du cache: {str(e)}"
        )

@router.get("/performance/benchmark")
async def benchmark_models(
    test_iterations: int = 10,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🏃‍♂️ Benchmark des performances des modèles
    
    Teste la vitesse d'inférence de tous les modèles disponibles
    sur une image de test.
    """
    
    try:
        import numpy as np
        import time
        
        # Image de test (320x320x3)
        test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        results = {}
        
        for model_name in model_manager.available_models.keys():
            try:
                logger.info(f"Benchmark du modèle {model_name}...")
                
                # Chargement du modèle
                start_load = time.time()
                detector = await model_manager.get_detector(model_name)
                load_time = (time.time() - start_load) * 1000
                
                # Warm-up (1 inférence)
                detector.detect(test_image, enable_tracking=False, enable_lost_detection=False)
                
                # Test de performance
                inference_times = []
                for i in range(test_iterations):
                    start_time = time.time()
                    detector.detect(test_image, enable_tracking=False, enable_lost_detection=False)
                    inference_time = (time.time() - start_time) * 1000
                    inference_times.append(inference_time)
                
                # Statistiques
                avg_time = np.mean(inference_times)
                min_time = np.min(inference_times)
                max_time = np.max(inference_times)
                std_time = np.std(inference_times)
                
                results[model_name] = {
                    "load_time_ms": load_time,
                    "avg_inference_time_ms": avg_time,
                    "min_inference_time_ms": min_time,
                    "max_inference_time_ms": max_time,
                    "std_inference_time_ms": std_time,
                    "fps_estimate": 1000 / avg_time if avg_time > 0 else 0,
                    "iterations": test_iterations,
                    "memory_usage_mb": model_manager.available_models[model_name].memory_usage
                }
                
            except Exception as e:
                logger.error(f"Erreur benchmark {model_name}: {e}")
                results[model_name] = {
                    "error": str(e),
                    "success": False
                }
        
        # Tri par performance (temps moyen)
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        if successful_results:
            best_model = min(successful_results.keys(), 
                           key=lambda x: successful_results[x]["avg_inference_time_ms"])
            
            return {
                "success": True,
                "benchmark_completed": True,
                "test_image_size": test_image.shape,
                "iterations_per_model": test_iterations,
                "best_performing_model": best_model,
                "results": results,
                "summary": {
                    "models_tested": len(results),
                    "successful_tests": len(successful_results),
                    "failed_tests": len(results) - len(successful_results)
                }
            }
        else:
            return {
                "success": False,
                "message": "Aucun modèle n'a pu être testé avec succès",
                "results": results
            }
        
    except Exception as e:
        logger.error(f"Erreur benchmark: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du benchmark: {str(e)}"
        )

@router.get("/health")
async def models_health_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🏥 Vérification de santé des modèles
    
    Vérifie l'état de tous les modèles et du système
    """
    
    try:
        health_status = await model_manager.get_health_status()
        
        # Vérifications additionnelles
        health_checks = []
        
        # Vérifier que les fichiers de modèles existent
        for name, model_info in model_manager.available_models.items():
            if model_info.path.exists():
                health_checks.append(f"✅ Modèle {name}: fichier OK")
            else:
                health_checks.append(f"❌ Modèle {name}: fichier manquant")
        
        # Vérifier qu'au moins un modèle est chargé
        loaded_models = [name for name, info in model_manager.available_models.items() if info.is_loaded]
        if loaded_models:
            health_checks.append(f"✅ {len(loaded_models)} modèle(s) chargé(s)")
        else:
            health_checks.append("⚠️ Aucun modèle chargé")
        
        # État global
        overall_status = "healthy"
        if not loaded_models:
            overall_status = "degraded"
        
        error_count = sum(1 for check in health_checks if check.startswith("❌"))
        if error_count > 0:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": health_status["timestamp"].isoformat(),
            "models_status": health_status,
            "health_checks": health_checks,
            "loaded_models": loaded_models,
            "available_models": list(model_manager.available_models.keys()),
            "cache_usage": f"{len(model_manager.models_cache.cache)}/{model_manager.models_cache.max_size}"
        }
        
    except Exception as e:
        logger.error(f"Erreur health check modèles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vérification: {str(e)}"
        )