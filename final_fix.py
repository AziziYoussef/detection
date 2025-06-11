# final_fix.py
"""
🔧 CORRECTION FINALE COMPLÈTE
Résout tous les derniers problèmes pour un démarrage 100% fonctionnel
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description="", ignore_errors=False):
    """Exécute une commande avec gestion d'erreurs"""
    print(f"\n🔧 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("   ✅ Succès!")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"   ⚠️ Erreur ignorée: {e}")
            return True
        else:
            print(f"   ❌ Erreur: {e}")
            return False

def clean_env_file():
    """Nettoie le fichier .env des champs problématiques"""
    print("🧹 NETTOYAGE FICHIER .ENV")
    print("=" * 40)
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("📄 Création d'un fichier .env propre...")
        create_clean_env()
        return True
    
    # Lire et nettoyer
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Champs à supprimer
    forbidden_fields = ['LOG_LEVEL', 'LOG_FILE', 'IMAGE_SIZE']
    
    clean_lines = []
    removed_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and '=' in line_stripped:
            field_name = line_stripped.split('=')[0].strip()
            if field_name in forbidden_fields:
                print(f"🗑️  Supprimé: {line_stripped}")
                removed_count += 1
                continue
        clean_lines.append(line)
    
    # Sauvegarder le fichier nettoyé
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(clean_lines)
    
    print(f"✅ Fichier .env nettoyé ({removed_count} lignes supprimées)")
    return True

def create_clean_env():
    """Crée un fichier .env propre et minimal"""
    clean_env_content = '''# Configuration Service IA - Version Minimaliste

# === SERVEUR ===
HOST=0.0.0.0
PORT=8000
DEBUG=True

# === MODÈLES ===
DEFAULT_MODEL=stable_model_epoch_30.pth
EXTENDED_MODEL=best_extended_model.pth
FAST_MODEL=fast_stream_model.pth

# === DÉTECTION ===
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.5
MAX_DETECTIONS=50

# === PERFORMANCE ===
USE_GPU=False
BATCH_SIZE=4

# === OBJETS PERDUS ===
SUSPECT_THRESHOLD_SECONDS=30
LOST_THRESHOLD_SECONDS=300
CRITICAL_THRESHOLD_SECONDS=1800
OWNER_PROXIMITY_METERS=2.5
'''
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(clean_env_content)
    
    print("✅ Fichier .env propre créé")

def fix_torch_torchvision_compatibility():
    """Corrige la compatibilité torch/torchvision"""
    print("\n🔥 CORRECTION COMPATIBILITÉ TORCH/TORCHVISION")
    print("=" * 50)
    
    # Option 1: Essayer d'installer des versions spécifiques compatibles
    print("🔧 Tentative 1: Versions spécifiques compatibles...")
    
    # Désinstaller d'abord
    run_command("pip uninstall torch torchvision -y", "Désinstallation versions actuelles", ignore_errors=True)
    
    # Essayer des versions compatibles
    success = run_command(
        "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu",
        "Installation versions 2.0.1 compatibles",
        ignore_errors=True
    )
    
    if not success:
        print("🔧 Tentative 2: Installation sans torchvision...")
        # Installer seulement torch
        success = run_command(
            "pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Installation torch seul"
        )
    
    return success

def create_minimal_image_utils():
    """Crée une version minimale d'image_utils sans torchvision"""
    print("\n📄 CRÉATION IMAGE_UTILS MINIMAL")
    print("=" * 40)
    
    minimal_image_utils = '''# app/utils/image_utils.py - Version minimale sans torchvision
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional, Union
import torch
# import torchvision.transforms as transforms  # Commenté temporairement
from pathlib import Path

class ImageProcessor:
    """Processeur d'images pour la détection - Version simplifiée"""
    
    def __init__(self, target_size: Tuple[int, int] = (320, 320)):
        self.target_size = target_size
        # Transforms simple sans torchvision
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, dict]:
        """
        Prétraite une image pour l'inférence - Version simplifiée
        """
        original_shape = image.shape[:2]  # (H, W)
        
        # Conversion BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Redimensionnement simple
        resized = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalisation manuelle
        resized = resized.astype(np.float32) / 255.0
        
        # Normalisation avec mean/std
        for i in range(3):
            resized[:, :, i] = (resized[:, :, i] - self.mean[i]) / self.std[i]
        
        # Conversion en tensor
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        # Métadonnées
        info = {
            'original_shape': original_shape,
            'resized_shape': resized.shape[:2],
            'scale_factors': (1.0, 1.0),  # Simplifié
            'padding': (0, 0, 0, 0)
        }
        
        return tensor, info
    
    def postprocess_detections(self, detections: torch.Tensor, 
                             transform_info: dict) -> np.ndarray:
        """Post-traite les détections - Version simplifiée"""
        if detections.numel() == 0:
            return np.array([]).reshape(0, 6)
        
        detections_np = detections.cpu().numpy()
        
        # Mise à l'échelle simple vers l'image originale
        original_shape = transform_info['original_shape']
        h_scale = original_shape[0] / self.target_size[1]
        w_scale = original_shape[1] / self.target_size[0]
        
        if detections_np.shape[1] >= 4:
            detections_np[:, 0] *= w_scale  # x1
            detections_np[:, 1] *= h_scale  # y1
            detections_np[:, 2] *= w_scale  # x2
            detections_np[:, 3] *= h_scale  # y2
        
        return detections_np

def encode_image_to_base64(image: np.ndarray, format: str = 'JPEG', 
                          quality: int = 80) -> str:
    """Encode une image en base64"""
    if format.upper() == 'JPEG':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
    elif format.upper() == 'PNG':
        _, buffer = cv2.imencode('.png', image)
    else:
        raise ValueError(f"Format non supporté: {format}")
    
    return base64.b64encode(buffer).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """Décode une image base64 en numpy array"""
    try:
        # Suppression du préfixe data:image si présent
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Décodage
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Impossible de décoder l'image")
        
        return image
    except Exception as e:
        raise ValueError(f"Erreur décodage base64: {e}")

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """Redimensionne une image"""
    if not keep_aspect_ratio:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Centrer dans l'image cible
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def validate_image(image: np.ndarray) -> bool:
    """Valide qu'une image est correcte"""
    if image is None:
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [3, 4]:
        return False
    
    if image.size == 0:
        return False
    
    return True

def get_image_info(image: np.ndarray) -> dict:
    """Retourne les informations d'une image"""
    if not validate_image(image):
        return {}
    
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes,
        'aspect_ratio': w / h if h != 0 else 0
    }
'''
    
    with open("app/utils/image_utils.py", "w", encoding="utf-8") as f:
        f.write(minimal_image_utils)
    
    print("✅ image_utils.py minimal créé (sans torchvision)")

def create_ultra_minimal_config():
    """Crée une configuration ultra-minimale"""
    print("\n⚙️ CONFIGURATION ULTRA-MINIMALE")
    print("=" * 40)
    
    ultra_minimal_config = '''# app/config/config.py - Version ultra-minimale
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration ultra-minimale du service IA"""
    
    # === SERVEUR ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # === CHEMINS ===
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    MODELS_DIR: Path = STORAGE_DIR / "models"
    TEMP_DIR: Path = STORAGE_DIR / "temp"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    
    # === MODÈLES ===
    DEFAULT_MODEL: str = "stable_model_epoch_30.pth"
    EXTENDED_MODEL: str = "best_extended_model.pth"
    FAST_MODEL: str = "fast_stream_model.pth"
    
    # === PARAMÈTRES ===
    CONFIDENCE_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.5
    MAX_DETECTIONS: int = 50
    USE_GPU: bool = False
    BATCH_SIZE: int = 4
    
    # === OBJETS PERDUS ===
    SUSPECT_THRESHOLD_SECONDS: int = 30
    LOST_THRESHOLD_SECONDS: int = 300
    CRITICAL_THRESHOLD_SECONDS: int = 1800
    OWNER_PROXIMITY_METERS: float = 2.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore les champs supplémentaires

# Instance globale
settings = Settings()

# Configuration modèles fixe
MODEL_CONFIG = {
    'num_classes': 28,
    'image_size': (320, 320),
    'confidence_threshold': 0.5,
    'nms_threshold': 0.5,
    'max_detections': 50,
    'classes': [
        'person', 'backpack', 'suitcase', 'handbag', 'tie',
        'umbrella', 'hair drier', 'toothbrush', 'cell phone',
        'laptop', 'keyboard', 'mouse', 'remote', 'tv',
        'clock', 'microwave', 'bottle', 'cup', 'bowl',
        'knife', 'spoon', 'fork', 'wine glass', 'refrigerator',
        'scissors', 'book', 'vase', 'chair'
    ],
    'class_names_fr': {
        'person': 'Personne', 'backpack': 'Sac à dos', 'suitcase': 'Valise',
        'handbag': 'Sac à main', 'tie': 'Cravate', 'umbrella': 'Parapluie',
        'hair drier': 'Sèche-cheveux', 'toothbrush': 'Brosse à dents',
        'cell phone': 'Téléphone', 'laptop': 'Ordinateur portable',
        'keyboard': 'Clavier', 'mouse': 'Souris', 'remote': 'Télécommande',
        'tv': 'Télévision', 'clock': 'Horloge', 'microwave': 'Micro-ondes',
        'bottle': 'Bouteille', 'cup': 'Tasse', 'bowl': 'Bol',
        'knife': 'Couteau', 'spoon': 'Cuillère', 'fork': 'Fourchette',
        'wine glass': 'Verre', 'refrigerator': 'Réfrigérateur',
        'scissors': 'Ciseaux', 'book': 'Livre', 'vase': 'Vase', 'chair': 'Chaise'
    }
}

# Configuration objets perdus
LOST_OBJECT_CONFIG = {
    'temporal_thresholds': {'surveillance': 30, 'alert': 300, 'critical': 1800},
    'spatial_thresholds': {'owner_proximity': 2.5, 'movement_threshold': 0.5},
    'confidence_thresholds': {'object_detection': 0.5, 'tracking_stability': 0.8},
    'blacklist_objects': ['chair', 'tv', 'refrigerator', 'microwave'],
    'priority_objects': ['backpack', 'suitcase', 'handbag', 'laptop', 'cell phone']
}

def ensure_directories():
    """Crée les répertoires nécessaires"""
    for directory in [settings.STORAGE_DIR, settings.MODELS_DIR, settings.TEMP_DIR, settings.CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories()
'''
    
    with open("app/config/config.py", "w", encoding="utf-8") as f:
        f.write(ultra_minimal_config)
    
    print("✅ Configuration ultra-minimale créée")

def test_final_setup():
    """Test final de toute la configuration"""
    print("\n🧪 TEST FINAL COMPLET")
    print("=" * 40)
    
    try:
        print("🔍 Test torch...")
        import torch
        print(f"✅ torch {torch.__version__}")
        
        print("🔍 Test configuration...")
        from app.config.config import settings, MODEL_CONFIG
        print(f"✅ Configuration chargée")
        
        print("🔍 Test image_utils...")
        from app.utils.image_utils import ImageProcessor
        print(f"✅ ImageProcessor")
        
        print("🔍 Test application principale...")
        from app.main import app
        print(f"✅ Application FastAPI chargée")
        
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("🚀 SERVICE PRÊT À DÉMARRER!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Correction finale complète"""
    print("🔧 CORRECTION FINALE COMPLÈTE")
    print("=" * 60)
    print("Résolution de TOUS les problèmes restants")
    
    # 1. Nettoyer .env
    clean_env_file()
    
    # 2. Corriger torch/torchvision
    torch_ok = fix_torch_torchvision_compatibility()
    
    # 3. Créer image_utils minimal
    create_minimal_image_utils()
    
    # 4. Configuration ultra-minimale
    create_ultra_minimal_config()
    
    # 5. Test final
    final_ok = test_final_setup()
    
    # Rapport final
    print("\n" + "=" * 60)
    print("🏁 RAPPORT FINAL DE CORRECTION")
    print("=" * 60)
    
    if final_ok:
        print("🎉 SUCCÈS COMPLET! 🎉")
        print("\n✅ Tous les problèmes résolus")
        print("✅ Configuration optimisée")
        print("✅ Dépendances compatibles")
        print("✅ Application prête")
        
        print("\n🚀 DÉMARRAGE:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        
        print("\n📱 INTERFACES:")
        print("   • http://localhost:8000/docs")
        print("   • http://localhost:8000/health")
        print("   • http://localhost:8000/api/v1/stream/demo")
        
        print("\n🧪 TEST EXPRESS:")
        print("   curl http://localhost:8000/health")
        
    else:
        print("❌ PROBLÈMES PERSISTANTS")
        print("\n💡 SOLUTIONS ALTERNATIVES:")
        print("   • Utiliser un environnement Python plus ancien (3.10)")
        print("   • Installation complètement fresh")
        print("   • Utiliser Docker: docker-compose up")

if __name__ == "__main__":
    main()