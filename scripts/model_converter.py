#!/usr/bin/env python3
"""
🔄 MODEL CONVERTER - SCRIPT DE CONVERSION DE MODÈLES
===================================================
Script pour convertir les modèles PyTorch vers différents formats d'optimisation

Formats supportés:
- ONNX: Interopérabilité multi-frameworks
- TorchScript: Déploiement PyTorch optimisé  
- TensorRT: Optimisation NVIDIA GPU
- OpenVINO: Optimisation Intel CPU/GPU
- CoreML: Optimisation Apple Silicon
- Quantization: Réduction taille/mémoire

Utilisation:
    python model_converter.py --input model.pth --output model.onnx --format onnx
    python model_converter.py --benchmark --all-formats
    python model_converter.py --optimize --target mobile
"""

import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Ajouter le chemin parent pour imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image

# Imports du projet
from storage.models.config_epoch_30 import get_epoch30_model_config
from storage.models.config_extended import get_extended_model_config
from app.models.model import create_epoch30_model, create_extended_model, create_fast_model

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelConverter:
    """🔄 Convertisseur de modèles PyTorch"""
    
    def __init__(self):
        self.supported_formats = [
            'onnx', 'torchscript', 'tensorrt', 'openvino', 'coreml', 'quantized'
        ]
        
        # Chemins
        self.models_dir = Path("storage/models")
        self.converted_dir = Path("storage/models/converted")
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache des modèles chargés
        self._loaded_models = {}
        
        logger.info("🔄 ModelConverter initialisé")
    
    def convert_model(
        self,
        input_path: str,
        output_path: str,
        target_format: str,
        optimization_level: str = "balanced",
        **kwargs
    ) -> bool:
        """
        🔄 Convertit un modèle vers le format cible
        
        Args:
            input_path: Chemin du modèle source (.pth)
            output_path: Chemin de sortie
            target_format: Format cible (onnx, torchscript, etc.)
            optimization_level: Niveau d'optimisation (fast, balanced, quality)
            **kwargs: Arguments spécifiques au format
            
        Returns:
            True si conversion réussie
        """
        
        logger.info(f"🔄 Conversion {input_path} -> {output_path} ({target_format})")
        
        try:
            # 1. Charger le modèle source
            model, config = self._load_pytorch_model(input_path)
            if model is None:
                return False
            
            # 2. Préparer le modèle pour conversion
            model = self._prepare_model_for_conversion(model, optimization_level)
            
            # 3. Convertir selon le format
            if target_format.lower() == 'onnx':
                success = self._convert_to_onnx(model, output_path, config, **kwargs)
            elif target_format.lower() == 'torchscript':
                success = self._convert_to_torchscript(model, output_path, config, **kwargs)
            elif target_format.lower() == 'tensorrt':
                success = self._convert_to_tensorrt(model, output_path, config, **kwargs)
            elif target_format.lower() == 'openvino':
                success = self._convert_to_openvino(model, output_path, config, **kwargs)
            elif target_format.lower() == 'coreml':
                success = self._convert_to_coreml(model, output_path, config, **kwargs)
            elif target_format.lower() == 'quantized':
                success = self._convert_to_quantized(model, output_path, config, **kwargs)
            else:
                logger.error(f"❌ Format non supporté: {target_format}")
                return False
            
            if success:
                # 4. Valider la conversion
                validation_success = self._validate_converted_model(
                    output_path, target_format, config
                )
                
                if validation_success:
                    logger.info(f"✅ Conversion réussie: {output_path}")
                    return True
                else:
                    logger.error(f"❌ Validation échouée: {output_path}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion: {e}")
            return False
    
    def _load_pytorch_model(self, model_path: str) -> Tuple[Optional[nn.Module], Optional[Dict]]:
        """📥 Charge un modèle PyTorch"""
        
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                logger.error(f"❌ Modèle non trouvé: {model_path}")
                return None, None
            
            # Charger le checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extraire la configuration
            model_config = checkpoint.get('model_config', {})
            model_name = checkpoint.get('model_name', 'unknown')
            
            # Créer le modèle selon le type
            if 'epoch_30' in model_name.lower():
                model = create_epoch30_model()
                config = get_epoch30_model_config()
            elif 'extended' in model_name.lower():
                model = create_extended_model()
                config = get_extended_model_config()
            else:
                # Modèle générique
                model = create_fast_model()  # Fallback
                config = None
            
            # Charger les poids
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"✅ Modèle chargé: {model_name}")
            return model, config
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return None, None
    
    def _prepare_model_for_conversion(self, model: nn.Module, optimization_level: str) -> nn.Module:
        """🔧 Prépare le modèle pour conversion"""
        
        # Optimisations selon le niveau
        if optimization_level == "fast":
            # Optimisations légères
            model = torch.jit.optimize_for_inference(model)
        elif optimization_level == "quality":
            # Pas d'optimisations agressives
            pass
        else:  # balanced
            # Optimisations équilibrées
            model = torch.jit.optimize_for_inference(model)
        
        return model
    
    def _convert_to_onnx(
        self,
        model: nn.Module,
        output_path: str,
        config: Optional[Dict],
        **kwargs
    ) -> bool:
        """🔄 Conversion vers ONNX"""
        
        try:
            # Paramètres par défaut
            input_size = kwargs.get('input_size', (640, 640))
            batch_size = kwargs.get('batch_size', 1)
            opset_version = kwargs.get('opset_version', 11)
            
            # Exemple d'entrée
            dummy_input = torch.randn(batch_size, 3, *input_size)
            
            # Noms des entrées/sorties
            input_names = kwargs.get('input_names', ['input'])
            output_names = kwargs.get('output_names', ['output'])
            
            # Axes dynamiques pour batch variable
            dynamic_axes = kwargs.get('dynamic_axes', {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            })
            
            # Conversion ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"✅ Conversion ONNX terminée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion ONNX: {e}")
            return False
    
    def _convert_to_torchscript(
        self,
        model: nn.Module,
        output_path: str,
        config: Optional[Dict],
        **kwargs
    ) -> bool:
        """🔄 Conversion vers TorchScript"""
        
        try:
            method = kwargs.get('method', 'trace')  # 'trace' ou 'script'
            
            if method == 'trace':
                # Tracing avec exemple d'entrée
                input_size = kwargs.get('input_size', (640, 640))
                example_input = torch.randn(1, 3, *input_size)
                
                traced_model = torch.jit.trace(model, example_input)
                traced_model.save(output_path)
                
            else:  # script
                # Scripting (plus robuste mais plus lent)
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path)
            
            logger.info(f"✅ Conversion TorchScript ({method}) terminée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion TorchScript: {e}")
            return False
    
    def _convert_to_tensorrt(
        self,
        model: nn.Module,
        output_path: str,
        config: Optional[Dict],
        **kwargs
    ) -> bool:
        """🔄 Conversion vers TensorRT"""
        
        try:
            # Vérifier TensorRT disponible
            try:
                import tensorrt as trt
                import torch_tensorrt
            except ImportError:
                logger.error("❌ TensorRT non disponible")
                return False
            
            # Paramètres TensorRT
            input_size = kwargs.get('input_size', (640, 640))
            precision = kwargs.get('precision', 'fp16')  # fp32, fp16, int8
            
            # Compilation TensorRT
            if precision == 'fp16':
                trt_model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(shape=[1, 3, *input_size])],
                    enabled_precisions=[torch.half]
                )
            elif precision == 'int8':
                trt_model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(shape=[1, 3, *input_size])],
                    enabled_precisions=[torch.int8]
                )
            else:  # fp32
                trt_model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(shape=[1, 3, *input_size])]
                )
            
            # Sauvegarder
            torch.jit.save(trt_model, output_path)
            
            logger.info(f"✅ Conversion TensorRT ({precision}) terminée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion TensorRT: {e}")
            return False
    
    def _convert_to_openvino(
        self,
        model: nn.Module,
        output_path: str,
        config: Optional[Dict],
        **kwargs
    ) -> bool:
        """🔄 Conversion vers OpenVINO"""
        
        try:
            # Vérifier OpenVINO disponible
            try:
                from openvino.tools import mo
            except ImportError:
                logger.error("❌ OpenVINO non disponible")
                return False
            
            # D'abord convertir en ONNX
            onnx_path = output_path.replace('.xml', '_temp.onnx')
            if not self._convert_to_onnx(model, onnx_path, config, **kwargs):
                return False
            
            # Puis convertir ONNX vers OpenVINO
            mo_args = {
                'input_model': onnx_path,
                'output_dir': str(Path(output_path).parent),
                'model_name': Path(output_path).stem
            }
            
            # Paramètres additionnels
            if 'input_shape' in kwargs:
                mo_args['input_shape'] = kwargs['input_shape']
            
            # Conversion
            mo.convert_model(**mo_args)
            
            # Nettoyer fichier ONNX temporaire
            Path(onnx_path).unlink(missing_ok=True)
            
            logger.info(f"✅ Conversion OpenVINO terminée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion OpenVINO: {e}")
            return False
    
    def _convert_to_coreml(
        self,
        model: nn.Module,
        output_path: str,
        config: Optional[Dict],
        **kwargs
    ) -> bool:
        """🔄 Conversion vers CoreML (Apple)"""
        
        try:
            # Vérifier CoreML disponible
            try:
                import coremltools as ct
            except ImportError:
                logger.error("❌ CoreMLTools non disponible")
                return False
            
            # Paramètres
            input_size = kwargs.get('input_size', (640, 640))
            
            # Exemple d'entrée
            example_input = torch.randn(1, 3, *input_size)
            
            # Tracer le modèle
            traced_model = torch.jit.trace(model, example_input)
            
            # Conversion CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.ImageType(shape=(1, 3, *input_size))],
                classifier_config=ct.ClassifierConfig(
                    class_labels=list(range(28))  # 28 classes
                ) if kwargs.get('add_classifier', False) else None
            )
            
            # Métadonnées
            coreml_model.short_description = "Lost Objects Detection Model"
            coreml_model.author = "AI Detection Service"
            
            # Sauvegarder
            coreml_model.save(output_path)
            
            logger.info(f"✅ Conversion CoreML terminée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion CoreML: {e}")
            return False
    
    def _convert_to_quantized(
        self,
        model: nn.Module,
        output_path: str,
        config: Optional[Dict],
        **kwargs
    ) -> bool:
        """🔄 Quantification du modèle"""
        
        try:
            quantization_type = kwargs.get('quantization', 'dynamic')  # dynamic, static, qat
            
            if quantization_type == 'dynamic':
                # Quantification dynamique (plus simple)
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Conv2d, nn.Linear},
                    dtype=torch.qint8
                )
                
            elif quantization_type == 'static':
                # Quantification statique (nécessite calibration)
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # Calibration (nécessiterait des données réelles)
                # Ici on fait une calibration factice
                input_size = kwargs.get('input_size', (640, 640))
                for _ in range(10):
                    dummy_input = torch.randn(1, 3, *input_size)
                    model(dummy_input)
                
                quantized_model = torch.quantization.convert(model, inplace=False)
                
            else:
                logger.error(f"❌ Type de quantification non supporté: {quantization_type}")
                return False
            
            # Sauvegarder le modèle quantifié
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'quantization_type': quantization_type,
                'original_config': config.__dict__ if config else {}
            }, output_path)
            
            logger.info(f"✅ Quantification ({quantization_type}) terminée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur quantification: {e}")
            return False
    
    def _validate_converted_model(
        self,
        model_path: str,
        format_type: str,
        original_config: Optional[Dict]
    ) -> bool:
        """✅ Valide le modèle converti"""
        
        try:
            model_path = Path(model_path)
            
            # Vérifier que le fichier existe
            if not model_path.exists():
                logger.error(f"❌ Fichier converti non trouvé: {model_path}")
                return False
            
            # Vérifications spécifiques au format
            if format_type == 'onnx':
                return self._validate_onnx_model(model_path)
            elif format_type == 'torchscript':
                return self._validate_torchscript_model(model_path)
            elif format_type == 'coreml':
                return self._validate_coreml_model(model_path)
            else:
                # Validation générique (taille fichier)
                file_size = model_path.stat().st_size
                if file_size < 1024:  # Moins de 1KB suspect
                    logger.error(f"❌ Fichier trop petit: {file_size} bytes")
                    return False
            
            logger.info(f"✅ Validation réussie: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")
            return False
    
    def _validate_onnx_model(self, model_path: Path) -> bool:
        """✅ Valide un modèle ONNX"""
        
        try:
            import onnx
            import onnxruntime as ort
            
            # Charger et vérifier le modèle ONNX
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
            
            # Test d'inférence
            session = ort.InferenceSession(str(model_path))
            
            # Obtenir les dimensions d'entrée
            input_shape = session.get_inputs()[0].shape
            
            # Créer données de test
            if any(dim is None or isinstance(dim, str) for dim in input_shape):
                # Dimensions dynamiques
                test_shape = [1, 3, 640, 640]
            else:
                test_shape = input_shape
            
            test_input = np.random.randn(*test_shape).astype(np.float32)
            
            # Inférence test
            outputs = session.run(None, {session.get_inputs()[0].name: test_input})
            
            if outputs and len(outputs) > 0:
                logger.info(f"✅ Test ONNX réussi - Output shape: {[o.shape for o in outputs]}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur validation ONNX: {e}")
            return False
    
    def _validate_torchscript_model(self, model_path: Path) -> bool:
        """✅ Valide un modèle TorchScript"""
        
        try:
            # Charger le modèle TorchScript
            model = torch.jit.load(str(model_path), map_location='cpu')
            model.eval()
            
            # Test d'inférence
            test_input = torch.randn(1, 3, 640, 640)
            
            with torch.no_grad():
                output = model(test_input)
            
            if output is not None:
                logger.info(f"✅ Test TorchScript réussi")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur validation TorchScript: {e}")
            return False
    
    def _validate_coreml_model(self, model_path: Path) -> bool:
        """✅ Valide un modèle CoreML"""
        
        try:
            import coremltools as ct
            
            # Charger le modèle CoreML
            model = ct.models.MLModel(str(model_path))
            
            # Vérifier les spécifications
            spec = model.get_spec()
            
            if spec and len(spec.description.input) > 0:
                logger.info(f"✅ Test CoreML réussi")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur validation CoreML: {e}")
            return False
    
    def batch_convert(
        self,
        models_dir: str,
        target_formats: List[str],
        optimization_level: str = "balanced"
    ) -> Dict[str, bool]:
        """🔄 Conversion batch de tous les modèles"""
        
        models_dir = Path(models_dir)
        results = {}
        
        # Trouver tous les modèles .pth
        model_files = list(models_dir.glob("*.pth"))
        
        if not model_files:
            logger.warning(f"⚠️ Aucun modèle trouvé dans {models_dir}")
            return results
        
        logger.info(f"🔄 Conversion batch: {len(model_files)} modèles × {len(target_formats)} formats")
        
        for model_file in model_files:
            model_name = model_file.stem
            
            for target_format in target_formats:
                # Déterminer extension de sortie
                ext_map = {
                    'onnx': '.onnx',
                    'torchscript': '.pt',
                    'tensorrt': '.trt',
                    'openvino': '.xml',
                    'coreml': '.mlmodel',
                    'quantized': '_quantized.pth'
                }
                
                output_file = self.converted_dir / f"{model_name}{ext_map.get(target_format, '.converted')}"
                
                # Conversion
                success = self.convert_model(
                    str(model_file),
                    str(output_file),
                    target_format,
                    optimization_level
                )
                
                results[f"{model_name}_{target_format}"] = success
        
        # Résumé
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"✅ Conversion batch terminée: {successful}/{total} réussies")
        
        return results
    
    def benchmark_formats(self, model_path: str, formats: List[str]) -> Dict[str, Dict[str, float]]:
        """📊 Benchmark des différents formats"""
        
        logger.info(f"📊 Benchmark des formats: {formats}")
        
        results = {}
        
        # Modèle original PyTorch
        original_model, config = self._load_pytorch_model(model_path)
        if original_model is None:
            return results
        
        # Test du modèle original
        results['pytorch'] = self._benchmark_single_model(
            original_model, 'pytorch', config
        )
        
        # Convertir et tester chaque format
        for format_name in formats:
            try:
                # Conversion temporaire
                temp_output = f"/tmp/benchmark_model.{format_name}"
                
                if self.convert_model(model_path, temp_output, format_name):
                    # Charger et tester
                    if format_name == 'onnx':
                        benchmark_result = self._benchmark_onnx_model(temp_output)
                    elif format_name == 'torchscript':
                        benchmark_result = self._benchmark_torchscript_model(temp_output)
                    else:
                        benchmark_result = {'inference_time_ms': -1, 'memory_mb': -1}
                    
                    results[format_name] = benchmark_result
                    
                    # Nettoyer
                    Path(temp_output).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"❌ Erreur benchmark {format_name}: {e}")
                results[format_name] = {'error': str(e)}
        
        return results
    
    def _benchmark_single_model(
        self,
        model: nn.Module,
        format_name: str,
        config: Optional[Dict],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """📊 Benchmark d'un modèle unique"""
        
        model.eval()
        
        # Données de test
        test_input = torch.randn(1, 3, 640, 640)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Mesures
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(test_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
        
        # Mémoire approximative
        memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return {
            'inference_time_ms': np.mean(times),
            'inference_std_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'memory_mb': memory_mb
        }
    
    def _benchmark_onnx_model(self, model_path: str) -> Dict[str, float]:
        """📊 Benchmark modèle ONNX"""
        
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            
            # Warm-up
            for _ in range(10):
                _ = session.run(None, {input_name: test_input})
            
            # Mesures
            times = []
            for _ in range(100):
                start_time = time.time()
                _ = session.run(None, {input_name: test_input})
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            return {
                'inference_time_ms': np.mean(times),
                'inference_std_ms': np.std(times),
                'memory_mb': Path(model_path).stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_torchscript_model(self, model_path: str) -> Dict[str, float]:
        """📊 Benchmark modèle TorchScript"""
        
        try:
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            
            test_input = torch.randn(1, 3, 640, 640)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # Mesures
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    _ = model(test_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
            
            return {
                'inference_time_ms': np.mean(times),
                'inference_std_ms': np.std(times),
                'memory_mb': Path(model_path).stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {'error': str(e)}

def main():
    """🚀 Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(description="Convertisseur de modèles PyTorch")
    
    # Arguments principaux
    parser.add_argument('--input', '-i', type=str, help='Chemin du modèle source')
    parser.add_argument('--output', '-o', type=str, help='Chemin de sortie')
    parser.add_argument('--format', '-f', type=str, choices=['onnx', 'torchscript', 'tensorrt', 'openvino', 'coreml', 'quantized'], help='Format cible')
    
    # Options de conversion
    parser.add_argument('--optimization-level', type=str, choices=['fast', 'balanced', 'quality'], default='balanced', help='Niveau d\'optimisation')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], help='Taille d\'entrée [H W]')
    parser.add_argument('--batch-size', type=int, default=1, help='Taille de batch')
    
    # Modes spéciaux
    parser.add_argument('--batch-convert', action='store_true', help='Conversion batch de tous les modèles')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark des formats')
    parser.add_argument('--all-formats', action='store_true', help='Utiliser tous les formats')
    
    # Répertoires
    parser.add_argument('--models-dir', type=str, default='storage/models', help='Répertoire des modèles')
    
    args = parser.parse_args()
    
    # Créer le convertisseur
    converter = ModelConverter()
    
    if args.batch_convert:
        # Conversion batch
        formats = ['onnx', 'torchscript', 'quantized'] if args.all_formats else [args.format]
        if not args.format and not args.all_formats:
            formats = ['onnx', 'torchscript']  # Par défaut
        
        results = converter.batch_convert(
            args.models_dir,
            formats,
            args.optimization_level
        )
        
        print("\n📊 Résultats de conversion batch:")
        for model_format, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {model_format}")
    
    elif args.benchmark:
        # Benchmark
        if not args.input:
            print("❌ --input requis pour benchmark")
            return
        
        formats = ['onnx', 'torchscript'] if args.all_formats else [args.format] if args.format else ['onnx']
        
        results = converter.benchmark_formats(args.input, formats)
        
        print("\n📊 Résultats de benchmark:")
        for format_name, metrics in results.items():
            if 'error' in metrics:
                print(f"❌ {format_name}: {metrics['error']}")
            else:
                print(f"✅ {format_name}:")
                print(f"   Temps moyen: {metrics.get('inference_time_ms', 0):.2f}ms")
                print(f"   Mémoire: {metrics.get('memory_mb', 0):.1f}MB")
    
    else:
        # Conversion simple
        if not all([args.input, args.output, args.format]):
            print("❌ --input, --output et --format requis")
            return
        
        success = converter.convert_model(
            args.input,
            args.output,
            args.format,
            args.optimization_level,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size
        )
        
        if success:
            print(f"✅ Conversion réussie: {args.output}")
        else:
            print(f"❌ Conversion échouée")

if __name__ == "__main__":
    main()