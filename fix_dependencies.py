# fix_dependencies.py
"""
🔧 Correction des Problèmes de Dépendances
Résout les problèmes de compatibilité Pydantic et Torch/Torchvision
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Exécute une commande avec gestion d'erreurs"""
    print(f"\n🔧 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("   ✅ Succès!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erreur: {e}")
        if e.stdout:
            print(f"   📤 Sortie: {e.stdout}")
        if e.stderr:
            print(f"   💥 Erreur: {e.stderr}")
        return False

def fix_pydantic_settings():
    """Installe pydantic-settings pour résoudre BaseSettings"""
    print("\n" + "="*50)
    print("📦 CORRECTION PYDANTIC SETTINGS")
    print("="*50)
    
    success = run_command(
        "pip install pydantic-settings",
        "Installation de pydantic-settings"
    )
    
    return success

def fix_torch_compatibility():
    """Corrige la compatibilité torch/torchvision"""
    print("\n" + "="*50)
    print("🔥 CORRECTION TORCH/TORCHVISION")
    print("="*50)
    
    print("ℹ️  Réinstallation de torch et torchvision compatibles...")
    
    # Désinstaller les versions problématiques
    run_command(
        "pip uninstall torch torchvision -y",
        "Désinstallation des versions problématiques"
    )
    
    # Installer des versions compatibles
    success = run_command(
        "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu",
        "Installation de versions compatibles torch/torchvision"
    )
    
    return success

def fix_config_file():
    """Corrige le fichier de configuration pour utiliser pydantic-settings"""
    print("\n" + "="*50)
    print("⚙️ CORRECTION DU FICHIER CONFIG")
    print("="*50)
    
    config_file = Path("app/config/config.py")
    
    if not config_file.exists():
        print("❌ Fichier config.py non trouvé")
        return False
    
    # Lire le fichier actuel
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remplacer l'import problématique
    old_import = "from pydantic import BaseSettings"
    new_import = "from pydantic_settings import BaseSettings"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        
        # Sauvegarder
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fichier config.py corrigé")
        return True
    else:
        print("ℹ️  Import déjà correct ou différent")
        return True

def create_minimal_config():
    """Crée une configuration minimale qui fonctionne"""
    print("\n🔧 Création d'une configuration minimale...")
    
    minimal_config = '''# app/config/config.py
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration du service IA"""
    
    # === CONFIGURATION SERVEUR ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    WORKERS: int = 1
    
    # === CHEMINS ===
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    MODELS_DIR: Path = STORAGE_DIR / "models"
    TEMP_DIR: Path = STORAGE_DIR / "temp"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    
    # === CONFIGURATION MODÈLES ===
    DEFAULT_MODEL: str = "stable_model_epoch_30.pth"
    EXTENDED_MODEL: str = "best_extended_model.pth"
    FAST_MODEL: str = "fast_stream_model.pth"
    
    # === PARAMÈTRES DÉTECTION ===
    CONFIDENCE_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.5
    MAX_DETECTIONS: int = 50
    IMAGE_SIZE: tuple = (320, 320)
    
    # === STREAMING ===
    MAX_CONNECTIONS: int = 10
    STREAM_FPS: int = 15
    BUFFER_SIZE: int = 30
    
    # === OBJETS PERDUS - LOGIQUE MÉTIER ===
    SUSPECT_THRESHOLD_SECONDS: int = 30
    LOST_THRESHOLD_SECONDS: int = 300  # 5 minutes
    CRITICAL_THRESHOLD_SECONDS: int = 1800  # 30 minutes
    OWNER_PROXIMITY_METERS: float = 2.5
    
    # === PERFORMANCE ===
    USE_GPU: bool = False  # Désactivé par défaut pour éviter les problèmes
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 0
    MAX_MEMORY_USAGE: float = 0.8
    
    # === CACHE ===
    CACHE_TTL: int = 3600
    MAX_CACHE_SIZE: int = 100
    
    class Config:
        env_file = ".env"

# Instance globale des paramètres
settings = Settings()

# Configuration des modèles simplifiée
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
        'person': 'Personne',
        'backpack': 'Sac à dos',
        'suitcase': 'Valise',
        'handbag': 'Sac à main',
        'tie': 'Cravate',
        'hair drier': 'Sèche-cheveux',
        'toothbrush': 'Brosse à dents',
        'cell phone': 'Téléphone',
        'laptop': 'Ordinateur portable',
        'keyboard': 'Clavier',
        'mouse': 'Souris',
        'remote': 'Télécommande',
        'tv': 'Télévision',
        'bottle': 'Bouteille',
        'cup': 'Tasse',
        'bowl': 'Bol',
        'knife': 'Couteau',
        'spoon': 'Cuillère',
        'fork': 'Fourchette',
        'wine glass': 'Verre',
        'scissors': 'Ciseaux',
        'book': 'Livre',
        'clock': 'Horloge',
        'umbrella': 'Parapluie',
        'vase': 'Vase',
        'chair': 'Chaise',
        'microwave': 'Micro-ondes',
        'refrigerator': 'Réfrigérateur'
    }
}

# Configuration simplifiée des objets perdus
LOST_OBJECT_CONFIG = {
    'temporal_thresholds': {
        'surveillance': 30,
        'alert': 300,
        'critical': 1800,
        'escalation': 3600
    },
    
    'spatial_thresholds': {
        'owner_proximity': 2.5,
        'movement_threshold': 0.5,
        'zone_boundary': 10.0
    },
    
    'confidence_thresholds': {
        'object_detection': 0.5,
        'tracking_stability': 0.8,
        'person_association': 0.6
    },
    
    'blacklist_objects': [
        'chair', 'tv', 'refrigerator', 'microwave'
    ],
    
    'priority_objects': [
        'backpack', 'suitcase', 'handbag', 'laptop', 'cell phone'
    ]
}

def ensure_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        settings.STORAGE_DIR,
        settings.MODELS_DIR,
        settings.TEMP_DIR,
        settings.CACHE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialisation des répertoires
ensure_directories()
'''
    
    with open("app/config/config.py", "w", encoding="utf-8") as f:
        f.write(minimal_config)
    
    print("✅ Configuration minimale créée")

def test_imports_after_fix():
    """Teste les imports après correction"""
    print("\n" + "="*50)
    print("🧪 TEST DES IMPORTS APRÈS CORRECTION")
    print("="*50)
    
    try:
        print("🔍 Test torch...")
        import torch
        print(f"✅ torch {torch.__version__}")
        
        print("🔍 Test torchvision...")
        import torchvision
        print(f"✅ torchvision {torchvision.__version__}")
        
        print("🔍 Test pydantic-settings...")
        from pydantic_settings import BaseSettings
        print("✅ pydantic-settings")
        
        print("🔍 Test config...")
        from app.config.config import settings
        print("✅ app.config")
        
        print("🔍 Test main...")
        from app.main import app
        print("✅ app.main")
        
        print("\n🎉 TOUS LES IMPORTS RÉUSSIS!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de correction"""
    print("🔧 CORRECTION DES DÉPENDANCES")
    print("=" * 50)
    print("Résolution des problèmes Pydantic et Torch/Torchvision\n")
    
    # 1. Installer pydantic-settings
    pydantic_ok = fix_pydantic_settings()
    
    # 2. Corriger torch/torchvision
    torch_ok = fix_torch_compatibility()
    
    # 3. Créer config minimal
    create_minimal_config()
    
    # 4. Tester les imports
    imports_ok = test_imports_after_fix()
    
    # Rapport final
    print("\n" + "=" * 50)
    print("📊 RAPPORT DE CORRECTION")
    print("=" * 50)
    
    print(f"✅ Pydantic Settings: {'OK' if pydantic_ok else 'ÉCHEC'}")
    print(f"✅ Torch/Torchvision: {'OK' if torch_ok else 'ÉCHEC'}")
    print(f"✅ Imports: {'OK' if imports_ok else 'ÉCHEC'}")
    
    if imports_ok:
        print(f"\n🎉 CORRECTION RÉUSSIE!")
        print(f"\n🚀 PROCHAINES ÉTAPES:")
        print(f"   1. Redémarrez le service:")
        print(f"      python scripts/start_service.py")
        print(f"   2. Ou directement:")
        print(f"      uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print(f"   3. Testez: http://localhost:8000/docs")
    else:
        print(f"\n❌ PROBLÈMES PERSISTANTS")
        print(f"💡 Essayez une installation propre:")
        print(f"   conda create -n ai_service_clean python=3.10")
        print(f"   conda activate ai_service_clean")
        print(f"   pip install -r requirements.txt")

if __name__ == "__main__":
    main()