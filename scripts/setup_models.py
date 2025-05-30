#!/usr/bin/env python3
"""
🏗️ SETUP MODELS - SCRIPT D'INSTALLATION ET CONFIGURATION DES MODÈLES
=====================================================================
Script complet pour installer, configurer et valider les modèles de détection

Fonctionnalités:
- Téléchargement automatique des modèles depuis différentes sources
- Installation et configuration des dépendances
- Validation et tests des modèles
- Conversion automatique vers formats optimisés
- Configuration de l'environnement de production
- Sauvegarde et restauration des modèles
- Mise à jour automatique des modèles

Sources supportées:
- URLs directes (HTTP/HTTPS)
- Google Drive (via gdown)
- Hugging Face Hub
- Fichiers locaux
- Archives compressées (ZIP, TAR)

Utilisation:
    python setup_models.py --install-all               # Installation complète
    python setup_models.py --download-from-url <url>   # Téléchargement URL
    python setup_models.py --validate                  # Validation modèles
    python setup_models.py --convert-all               # Conversion optimisée
    python setup_models.py --production-setup          # Configuration production
"""

import argparse
import asyncio
import logging
import json
import hashlib
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
import sys
import os
import subprocess

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

import requests
from tqdm import tqdm
import torch

# Imports du projet
from storage.models.model_manager import ModelFileManager
from storage.models.config_epoch_30 import get_epoch30_model_config, MODEL_NAME as EPOCH30_NAME
from storage.models.config_extended import get_extended_model_config, MODEL_NAME as EXTENDED_NAME

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelInstaller:
    """🏗️ Installateur de modèles"""
    
    def __init__(self):
        self.models_dir = Path("storage/models")
        self.temp_dir = Path("storage/temp/model_install")
        self.config_dir = Path("storage/models/configs")
        
        # Créer répertoires
        for directory in [self.models_dir, self.temp_dir, self.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Gestionnaire de fichiers modèles
        self.file_manager = ModelFileManager()
        
        # Registre des modèles disponibles
        self.model_registry = self._load_model_registry()
        
        logger.info("🏗️ ModelInstaller initialisé")
    
    def _load_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """📋 Charge le registre des modèles disponibles"""
        
        return {
            "epoch_30": {
                "name": "Stable Model Epoch 30",
                "version": "1.0.0",
                "description": "Modèle champion F1=49.86%",
                "architecture": "mobilenet_ssd",
                "size_mb": 45.2,
                "classes": 28,
                "performance": {
                    "f1_score": 0.4986,
                    "precision": 0.6073,
                    "map": 0.4156
                },
                "urls": {
                    "primary": None,  # À configurer avec vraie URL
                    "mirror": None,
                    "gdrive": None
                },
                "checksum": None,  # Hash MD5 à configurer
                "required": True
            },
            "extended": {
                "name": "Best Extended Model",
                "version": "1.1.0", 
                "description": "Modèle étendu 28 classes haute précision",
                "architecture": "efficientdet",
                "size_mb": 67.8,
                "classes": 28,
                "performance": {
                    "f1_score": 0.5234,
                    "precision": 0.6445,
                    "map": 0.4789
                },
                "urls": {
                    "primary": None,
                    "mirror": None,
                    "gdrive": None
                },
                "checksum": None,
                "required": False
            },
            "fast": {
                "name": "Fast Streaming Model",
                "version": "1.0.0",
                "description": "Modèle optimisé streaming temps réel",
                "architecture": "yolo_v5",
                "size_mb": 28.1,
                "classes": 28,
                "performance": {
                    "f1_score": 0.4523,
                    "precision": 0.5678,
                    "map": 0.3845
                },
                "urls": {
                    "primary": None,
                    "mirror": None,
                    "gdrive": None
                },
                "checksum": None,
                "required": False
            }
        }
    
    def install_all_models(self, force_reinstall: bool = False) -> bool:
        """🏗️ Installation complète de tous les modèles"""
        
        logger.info("🏗️ Installation complète des modèles")
        
        success_count = 0
        total_count = len(self.model_registry)
        
        # Installation par ordre de priorité
        model_order = ["epoch_30", "extended", "fast"]  # Champion en premier
        
        for model_name in model_order:
            if model_name not in self.model_registry:
                continue
            
            model_info = self.model_registry[model_name]
            
            logger.info(f"📦 Installation {model_info['name']}...")
            
            try:
                success = self.install_model(model_name, force_reinstall)
                if success:
                    success_count += 1
                    logger.info(f"✅ {model_info['name']} installé")
                else:
                    logger.error(f"❌ Échec installation {model_info['name']}")
                    
                    # Si modèle requis échoue, créer version de test
                    if model_info.get("required", False):
                        logger.info("🔧 Création version de test...")
                        self._create_test_model(model_name)
                        success_count += 1
                        
            except Exception as e:
                logger.error(f"❌ Erreur installation {model_name}: {e}")
                
                # Modèle requis -> version de test
                if model_info.get("required", False):
                    self._create_test_model(model_name)
                    success_count += 1
        
        # Validation finale
        if success_count > 0:
            self.validate_installation()
        
        logger.info(f"✅ Installation terminée: {success_count}/{total_count} modèles")
        return success_count == total_count
    
    def install_model(self, model_name: str, force_reinstall: bool = False) -> bool:
        """📦 Installation d'un modèle spécifique"""
        
        if model_name not in self.model_registry:
            logger.error(f"❌ Modèle inconnu: {model_name}")
            return False
        
        model_info = self.model_registry[model_name]
        model_path = self.models_dir / f"{model_name}.pth"
        
        # Vérifier si déjà installé
        if model_path.exists() and not force_reinstall:
            if self._validate_model_file(model_path, model_info):
                logger.info(f"✅ Modèle déjà installé et valide: {model_name}")
                return True
            else:
                logger.warning(f"⚠️ Modèle existant invalide, réinstallation...")
        
        # Tentatives de téléchargement
        urls = model_info.get("urls", {})
        
        for source_name, url in urls.items():
            if not url:
                continue
                
            logger.info(f"🔗 Tentative téléchargement depuis {source_name}: {url}")
            
            try:
                if self._download_model(url, model_path, model_info):
                    logger.info(f"✅ Téléchargement réussi depuis {source_name}")
                    return True
                    
            except Exception as e:
                logger.warning(f"⚠️ Échec téléchargement {source_name}: {e}")
                continue
        
        # Si aucun téléchargement ne fonctionne
        logger.warning(f"⚠️ Aucune source disponible pour {model_name}")
        return False
    
    def _download_model(self, url: str, output_path: Path, model_info: Dict[str, Any]) -> bool:
        """⬇️ Télécharge un modèle depuis une URL"""
        
        try:
            # Analyser URL
            parsed = urlparse(url)
            
            if "drive.google.com" in parsed.netloc:
                return self._download_from_gdrive(url, output_path, model_info)
            elif "huggingface.co" in parsed.netloc:
                return self._download_from_huggingface(url, output_path, model_info)
            else:
                return self._download_from_http(url, output_path, model_info)
                
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement: {e}")
            return False
    
    def _download_from_http(self, url: str, output_path: Path, model_info: Dict[str, Any]) -> bool:
        """⬇️ Téléchargement HTTP standard"""
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Taille du fichier
            total_size = int(response.headers.get('content-length', 0))
            expected_size = model_info.get("size_mb", 0) * 1024 * 1024
            
            # Vérification approximative de la taille
            if total_size > 0 and expected_size > 0:
                size_diff = abs(total_size - expected_size) / expected_size
                if size_diff > 0.1:  # Plus de 10% de différence
                    logger.warning(f"⚠️ Taille inattendue: {total_size/1024/1024:.1f}MB (attendu: {expected_size/1024/1024:.1f}MB)")
            
            # Téléchargement avec barre de progression
            with open(output_path, 'wb') as f:
                with tqdm(
                    desc=f"Téléchargement {output_path.name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Validation
            if self._validate_model_file(output_path, model_info):
                logger.info(f"✅ Téléchargement HTTP réussi: {output_path}")
                return True
            else:
                output_path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement HTTP: {e}")
            output_path.unlink(missing_ok=True)
            return False
    
    def _download_from_gdrive(self, url: str, output_path: Path, model_info: Dict[str, Any]) -> bool:
        """⬇️ Téléchargement depuis Google Drive"""
        
        try:
            import gdown
            
            # Extraire ID du fichier depuis URL
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            elif "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
            else:
                logger.error("❌ Format URL Google Drive invalide")
                return False
            
            # Téléchargement
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(gdown_url, str(output_path), quiet=False)
            
            # Validation
            if self._validate_model_file(output_path, model_info):
                logger.info(f"✅ Téléchargement Google Drive réussi: {output_path}")
                return True
            else:
                output_path.unlink(missing_ok=True)
                return False
                
        except ImportError:
            logger.error("❌ gdown non installé. Installez avec: pip install gdown")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement Google Drive: {e}")
            output_path.unlink(missing_ok=True)
            return False
    
    def _download_from_huggingface(self, url: str, output_path: Path, model_info: Dict[str, Any]) -> bool:
        """⬇️ Téléchargement depuis Hugging Face Hub"""
        
        try:
            from huggingface_hub import hf_hub_download
            
            # Parser URL Hugging Face
            # Format: https://huggingface.co/user/repo/resolve/main/model.pth
            parts = url.split("/")
            if len(parts) < 6:
                logger.error("❌ Format URL Hugging Face invalide")
                return False
            
            repo_id = f"{parts[3]}/{parts[4]}"
            filename = parts[-1]
            
            # Téléchargement
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.temp_dir)
            )
            
            # Copier vers destination finale
            shutil.copy2(downloaded_path, output_path)
            
            # Validation
            if self._validate_model_file(output_path, model_info):
                logger.info(f"✅ Téléchargement Hugging Face réussi: {output_path}")
                return True
            else:
                output_path.unlink(missing_ok=True)
                return False
                
        except ImportError:
            logger.error("❌ huggingface_hub non installé. Installez avec: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement Hugging Face: {e}")
            output_path.unlink(missing_ok=True)
            return False
    
    def _validate_model_file(self, model_path: Path, model_info: Dict[str, Any]) -> bool:
        """✅ Valide un fichier de modèle"""
        
        try:
            # Vérifier existence
            if not model_path.exists():
                logger.error(f"❌ Fichier inexistant: {model_path}")
                return False
            
            # Vérifier taille
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            expected_size_mb = model_info.get("size_mb", 0)
            
            if expected_size_mb > 0:
                size_diff = abs(file_size_mb - expected_size_mb) / expected_size_mb
                if size_diff > 0.2:  # Plus de 20% de différence
                    logger.warning(f"⚠️ Taille inattendue: {file_size_mb:.1f}MB (attendu: {expected_size_mb:.1f}MB)")
            
            # Vérifier checksum si disponible
            expected_checksum = model_info.get("checksum")
            if expected_checksum:
                actual_checksum = self._calculate_file_checksum(model_path)
                if actual_checksum != expected_checksum:
                    logger.error(f"❌ Checksum invalide: {actual_checksum} != {expected_checksum}")
                    return False
            
            # Tentative de chargement PyTorch
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Vérifications de base
                required_keys = ['model_state_dict']
                for key in required_keys:
                    if key not in checkpoint:
                        logger.error(f"❌ Clé manquante dans checkpoint: {key}")
                        return False
                
                logger.info(f"✅ Modèle valide: {model_path} ({file_size_mb:.1f}MB)")
                return True
                
            except Exception as e:
                logger.error(f"❌ Erreur chargement PyTorch: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")
            return False
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """🔐 Calcule le checksum MD5 d'un fichier"""
        
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _create_test_model(self, model_name: str):
        """🔧 Crée une version de test d'un modèle"""
        
        logger.info(f"🔧 Création modèle de test: {model_name}")
        
        try:
            if model_name == "epoch_30":
                self.file_manager.create_stable_model_epoch_30()
            elif model_name == "extended":
                self.file_manager.create_best_extended_model()
            else:
                # Modèle générique
                self.file_manager.create_stable_model_epoch_30()
                
            logger.info(f"✅ Modèle de test créé: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Erreur création modèle test: {e}")
    
    def download_from_url(self, url: str, model_name: Optional[str] = None) -> bool:
        """⬇️ Télécharge un modèle depuis une URL directe"""
        
        logger.info(f"⬇️ Téléchargement depuis URL: {url}")
        
        # Déterminer nom du modèle
        if not model_name:
            parsed = urlparse(url)
            filename = Path(parsed.path).stem
            model_name = filename
        
        output_path = self.models_dir / f"{model_name}.pth"
        
        # Info modèle basique
        model_info = {
            "name": model_name,
            "size_mb": 0,  # Inconnu
            "checksum": None
        }
        
        try:
            success = self._download_from_http(url, output_path, model_info)
            
            if success:
                logger.info(f"✅ Téléchargement réussi: {output_path}")
                return True
            else:
                logger.error(f"❌ Échec téléchargement: {url}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement URL: {e}")
            return False
    
    def install_from_archive(self, archive_path: str) -> bool:
        """📦 Installation depuis archive (ZIP/TAR)"""
        
        logger.info(f"📦 Installation depuis archive: {archive_path}")
        
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            logger.error(f"❌ Archive non trouvée: {archive_path}")
            return False
        
        try:
            extract_dir = self.temp_dir / f"extract_{archive_path.stem}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Extraction selon le type
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                logger.error(f"❌ Format d'archive non supporté: {archive_path.suffix}")
                return False
            
            # Chercher fichiers .pth dans l'extraction
            model_files = list(extract_dir.rglob("*.pth"))
            
            if not model_files:
                logger.error("❌ Aucun fichier .pth trouvé dans l'archive")
                return False
            
            # Copier les modèles
            copied_count = 0
            for model_file in model_files:
                dest_path = self.models_dir / model_file.name
                shutil.copy2(model_file, dest_path)
                logger.info(f"📦 Modèle extrait: {dest_path}")
                copied_count += 1
            
            # Nettoyer
            shutil.rmtree(extract_dir, ignore_errors=True)
            
            logger.info(f"✅ Archive installée: {copied_count} modèles")
            return copied_count > 0
            
        except Exception as e:
            logger.error(f"❌ Erreur installation archive: {e}")
            return False
    
    def validate_installation(self) -> Dict[str, bool]:
        """✅ Valide l'installation complète"""
        
        logger.info("✅ Validation de l'installation")
        
        results = {}
        
        # Utiliser le gestionnaire de fichiers pour validation
        models_summary = self.file_manager.get_models_summary()
        
        for model_name, info in models_summary.items():
            is_valid = info.get("valid", False)
            results[model_name] = is_valid
            
            if is_valid:
                logger.info(f"✅ {model_name}: Valide ({info.get('size_mb', 0):.1f}MB)")
            else:
                logger.error(f"❌ {model_name}: {info.get('error', 'Erreur inconnue')}")
        
        # Résumé
        valid_count = sum(1 for valid in results.values() if valid)
        total_count = len(results)
        
        logger.info(f"📊 Validation: {valid_count}/{total_count} modèles valides")
        
        return results
    
    def convert_all_models(self) -> bool:
        """🔄 Conversion de tous les modèles vers formats optimisés"""
        
        logger.info("🔄 Conversion des modèles vers formats optimisés")
        
        try:
            # Import du convertisseur
            sys.path.append(str(Path(__file__).parent))
            from model_converter import ModelConverter
            
            converter = ModelConverter()
            
            # Conversion batch vers ONNX et TorchScript
            results = converter.batch_convert(
                str(self.models_dir),
                ["onnx", "torchscript"],
                "balanced"
            )
            
            # Résumé
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            logger.info(f"🔄 Conversion terminée: {successful}/{total} réussies")
            
            return successful > 0
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion: {e}")
            return False
    
    def setup_production_environment(self) -> bool:
        """🏭 Configuration pour environnement de production"""
        
        logger.info("🏭 Configuration environnement de production")
        
        try:
            # Vérifier modèles requis
            required_models = ["epoch_30"]  # Minimum pour production
            
            for model_name in required_models:
                model_path = self.models_dir / f"{model_name}.pth"
                
                if not model_path.exists():
                    logger.error(f"❌ Modèle requis manquant: {model_name}")
                    return False
            
            # Optimiser les modèles pour production
            if not self.convert_all_models():
                logger.warning("⚠️ Conversion des modèles échouée")
            
            # Créer fichier de configuration production
            prod_config = {
                "environment": "production",
                "models": {
                    model_name: {
                        "path": str(self.models_dir / f"{model_name}.pth"),
                        "enabled": True,
                        "priority": 1 if model_name == "epoch_30" else 2
                    }
                    for model_name in self.model_registry.keys()
                    if (self.models_dir / f"{model_name}.pth").exists()
                },
                "optimization": {
                    "use_tensorrt": False,  # À activer si TensorRT disponible
                    "use_onnx": True,
                    "half_precision": False,  # Activer si GPU supporté
                    "batch_size": 8
                },
                "monitoring": {
                    "enable_metrics": True,
                    "log_level": "INFO"
                }
            }
            
            config_file = self.models_dir / "production_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(prod_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Configuration production créée: {config_file}")
            
            # Permissions et sécurité
            self._secure_model_files()
            
            logger.info("✅ Environnement de production configuré")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur configuration production: {e}")
            return False
    
    def _secure_model_files(self):
        """🔒 Sécurise les fichiers de modèles"""
        
        try:
            # Lecture seule pour les modèles en production
            for model_file in self.models_dir.glob("*.pth"):
                os.chmod(model_file, 0o444)  # Read-only
            
            logger.info("🔒 Fichiers modèles sécurisés")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur sécurisation: {e}")
    
    def backup_models(self, backup_path: str) -> bool:
        """💾 Sauvegarde des modèles"""
        
        logger.info(f"💾 Sauvegarde des modèles vers: {backup_path}")
        
        try:
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copier tous les fichiers de modèles
            model_files = list(self.models_dir.glob("*.pth"))
            config_files = list(self.models_dir.glob("*.py"))
            
            copied_count = 0
            
            for file_path in model_files + config_files:
                dest_path = backup_path / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_count += 1
            
            # Créer manifeste de sauvegarde
            manifest = {
                "backup_date": time.time(),
                "files": [f.name for f in model_files + config_files],
                "total_size_mb": sum(f.stat().st_size for f in model_files + config_files) / (1024 * 1024)
            }
            
            with open(backup_path / "backup_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Sauvegarde terminée: {copied_count} fichiers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def restore_models(self, backup_path: str) -> bool:
        """♻️ Restauration des modèles"""
        
        logger.info(f"♻️ Restauration des modèles depuis: {backup_path}")
        
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                logger.error(f"❌ Répertoire de sauvegarde inexistant: {backup_path}")
                return False
            
            # Vérifier manifeste
            manifest_file = backup_path / "backup_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                logger.info(f"📋 Sauvegarde du {manifest.get('backup_date', 'inconnu')}")
            
            # Restaurer fichiers
            backup_files = list(backup_path.glob("*.pth")) + list(backup_path.glob("*.py"))
            
            restored_count = 0
            for file_path in backup_files:
                dest_path = self.models_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                restored_count += 1
            
            logger.info(f"✅ Restauration terminée: {restored_count} fichiers")
            
            # Validation
            self.validate_installation()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur restauration: {e}")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """🔍 Vérification des dépendances"""
        
        logger.info("🔍 Vérification des dépendances")
        
        dependencies = {
            "torch": ("PyTorch", "pip install torch"),
            "torchvision": ("TorchVision", "pip install torchvision"),
            "PIL": ("Pillow", "pip install Pillow"),
            "cv2": ("OpenCV", "pip install opencv-python"),
            "numpy": ("NumPy", "pip install numpy"),
            "requests": ("Requests", "pip install requests"),
            "tqdm": ("tqdm", "pip install tqdm")
        }
        
        optional_dependencies = {
            "onnx": ("ONNX", "pip install onnx"),
            "onnxruntime": ("ONNX Runtime", "pip install onnxruntime"),
            "gdown": ("gdown", "pip install gdown"),
            "huggingface_hub": ("Hugging Face Hub", "pip install huggingface_hub")
        }
        
        results = {}
        
        # Vérifier dépendances requises
        for module_name, (display_name, install_cmd) in dependencies.items():
            try:
                __import__(module_name)
                results[module_name] = True
                logger.info(f"✅ {display_name}: Disponible")
            except ImportError:
                results[module_name] = False
                logger.error(f"❌ {display_name}: Manquant ({install_cmd})")
        
        # Vérifier dépendances optionnelles
        for module_name, (display_name, install_cmd) in optional_dependencies.items():
            try:
                __import__(module_name)
                results[module_name] = True
                logger.info(f"✅ {display_name}: Disponible (optionnel)")
            except ImportError:
                results[module_name] = False
                logger.warning(f"⚠️ {display_name}: Manquant (optionnel - {install_cmd})")
        
        return results
    
    def install_dependencies(self) -> bool:
        """📦 Installation automatique des dépendances"""
        
        logger.info("📦 Installation des dépendances")
        
        try:
            # Dépendances essentielles
            essential_packages = [
                "torch", "torchvision", "Pillow", "opencv-python", 
                "numpy", "requests", "tqdm"
            ]
            
            for package in essential_packages:
                logger.info(f"📦 Installation {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ {package} installé")
                else:
                    logger.error(f"❌ Erreur installation {package}: {result.stderr}")
                    return False
            
            logger.info("✅ Dépendances essentielles installées")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur installation dépendances: {e}")
            return False

def main():
    """🚀 Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(description="Installation et configuration des modèles")
    
    # Actions principales
    parser.add_argument('--install-all', action='store_true', help='Installation complète')
    parser.add_argument('--validate', action='store_true', help='Validation des modèles')
    parser.add_argument('--convert-all', action='store_true', help='Conversion vers formats optimisés')
    parser.add_argument('--production-setup', action='store_true', help='Configuration production')
    
    # Téléchargements
    parser.add_argument('--download-from-url', type=str, help='Télécharger depuis URL')
    parser.add_argument('--install-from-archive', type=str, help='Installer depuis archive')
    parser.add_argument('--model-name', type=str, help='Nom du modèle pour téléchargement')
    
    # Sauvegarde/Restauration
    parser.add_argument('--backup', type=str, help='Sauvegarder vers répertoire')
    parser.add_argument('--restore', type=str, help='Restaurer depuis répertoire')
    
    # Dépendances
    parser.add_argument('--check-deps', action='store_true', help='Vérifier dépendances')
    parser.add_argument('--install-deps', action='store_true', help='Installer dépendances')
    
    # Options
    parser.add_argument('--force', action='store_true', help='Forcer réinstallation')
    
    args = parser.parse_args()
    
    # Créer installateur
    installer = ModelInstaller()
    
    try:
        if args.install_all:
            # Installation complète
            logger.info("🚀 Installation complète des modèles")
            success = installer.install_all_models(args.force)
            if success:
                print("✅ Installation complète réussie")
            else:
                print("⚠️ Installation partielle (voir logs pour détails)")
        
        elif args.validate:
            # Validation
            results = installer.validate_installation()
            valid_count = sum(1 for valid in results.values() if valid)
            total_count = len(results)
            print(f"📊 Validation: {valid_count}/{total_count} modèles valides")
            
            for model_name, is_valid in results.items():
                status = "✅" if is_valid else "❌"
                print(f"{status} {model_name}")
        
        elif args.convert_all:
            # Conversion
            success = installer.convert_all_models()
            if success:
                print("✅ Conversion des modèles réussie")
            else:
                print("❌ Erreur lors de la conversion")
        
        elif args.production_setup:
            # Configuration production
            success = installer.setup_production_environment()
            if success:
                print("✅ Environnement de production configuré")
            else:
                print("❌ Erreur configuration production")
        
        elif args.download_from_url:
            # Téléchargement URL
            success = installer.download_from_url(args.download_from_url, args.model_name)
            if success:
                print("✅ Téléchargement réussi")
            else:
                print("❌ Échec téléchargement")
        
        elif args.install_from_archive:
            # Installation archive
            success = installer.install_from_archive(args.install_from_archive)
            if success:
                print("✅ Installation depuis archive réussie")
            else:
                print("❌ Échec installation archive")
        
        elif args.backup:
            # Sauvegarde
            success = installer.backup_models(args.backup)
            if success:
                print(f"✅ Sauvegarde créée: {args.backup}")
            else:
                print("❌ Erreur sauvegarde")
        
        elif args.restore:
            # Restauration
            success = installer.restore_models(args.restore)
            if success:
                print(f"✅ Restauration réussie: {args.restore}")
            else:
                print("❌ Erreur restauration")
        
        elif args.check_deps:
            # Vérification dépendances
            results = installer.check_dependencies()
            missing = [dep for dep, available in results.items() if not available]
            
            if missing:
                print(f"⚠️ Dépendances manquantes: {', '.join(missing)}")
            else:
                print("✅ Toutes les dépendances sont disponibles")
        
        elif args.install_deps:
            # Installation dépendances
            success = installer.install_dependencies()
            if success:
                print("✅ Dépendances installées")
            else:
                print("❌ Erreur installation dépendances")
        
        else:
            # Afficher aide
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⏹️ Installation interrompue par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Erreur principale: {e}", exc_info=True)

if __name__ == "__main__":
    import time
    main()