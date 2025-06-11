# quick_fix.py
"""
🔧 Correction Rapide des Problèmes d'Import
Script pour résoudre rapidement les erreurs de démarrage
"""

import os
from pathlib import Path

def fix_app_init():
    """Corrige le fichier app/__init__.py"""
    print("🔧 Correction de app/__init__.py...")
    
    corrected_content = '''# app/__init__.py
"""
🔍 Service IA - Détection d'Objets Perdus
Service intelligent de détection et surveillance d'objets perdus en temps réel
"""

__version__ = "1.0.0"
__author__ = "Équipe IA"
__description__ = "Service de détection d'objets perdus basé sur l'IA"
__email__ = "contact@yourcompany.com"
__license__ = "MIT"

# NOTE: Les imports spécifiques sont gérés par chaque sous-module
# Éviter les imports circulaires en ne définissant que les métadonnées ici
'''
    
    with open("app/__init__.py", "w", encoding="utf-8") as f:
        f.write(corrected_content)
    
    print("✅ app/__init__.py corrigé")

def fix_utils_init():
    """Corrige le fichier app/utils/__init__.py pour éviter les imports circulaires"""
    print("🔧 Correction de app/utils/__init__.py...")
    
    utils_init_content = '''# app/utils/__init__.py
"""
🔧 Utilities
Fonctions utilitaires et helpers
"""

# Imports disponibles mais pas automatiques pour éviter les circularités
# Utilisez: from app.utils.image_utils import ImageProcessor
# Utilisez: from app.utils.box_utils import box_iou

__all__ = ["image_utils", "box_utils"]
'''
    
    with open("app/utils/__init__.py", "w", encoding="utf-8") as f:
        f.write(utils_init_content)
    
    print("✅ app/utils/__init__.py corrigé")

def fix_all_inits():
    """Corrige tous les fichiers __init__.py pour éviter les imports problématiques"""
    
    print("🔧 Correction de tous les fichiers __init__.py...")
    
    # Dictionnaire des corrections pour chaque __init__.py
    init_fixes = {
        "app/api/__init__.py": '''# app/api/__init__.py
"""
🌐 API Routes
Endpoints FastAPI pour le service de détection
"""

# Imports gérés par le routeur principal
__all__ = ["routes"]
''',
        
        "app/api/endpoints/__init__.py": '''# app/api/endpoints/__init__.py
"""
🎯 API Endpoints
Points d'entrée spécialisés pour différents types de détection
"""

__all__ = [
    "image_detection",
    "video_detection", 
    "stream_detection",
    "models"
]
''',

        "app/core/__init__.py": '''# app/core/__init__.py
"""
🧠 Core Logic
Logique métier principale du service
"""

# Imports disponibles mais pas automatiques
__all__ = ["detector", "model_manager"]
''',

        "app/schemas/__init__.py": '''# app/schemas/__init__.py
"""
📋 Schemas
Schémas Pydantic pour validation des données
"""

# Imports disponibles mais pas automatiques
__all__ = ["detection"]
''',

        "app/services/__init__.py": '''# app/services/__init__.py
"""
🎯 Services
Services métier spécialisés
"""

__all__ = ["stream_service"]
''',

        "app/config/__init__.py": '''# app/config/__init__.py
"""
⚙️ Configuration
Paramètres et configuration du service
"""

__all__ = ["config"]
'''
    }
    
    for file_path, content in init_fixes.items():
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ {file_path} corrigé")
        except Exception as e:
            print(f"⚠️ Erreur correction {file_path}: {e}")

def test_imports():
    """Teste que les imports fonctionnent maintenant"""
    print("\n🧪 Test des imports...")
    
    try:
        # Test import app
        from app import __version__
        print(f"✅ app importé (version: {__version__})")
        
        # Test import main
        from app.main import app as fastapi_app
        print("✅ app.main importé")
        
        # Test import config
        from app.config.config import settings
        print("✅ app.config importé")
        
        print("\n🎉 Tous les imports critiques fonctionnent !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        return False

def main():
    """Fonction principale de correction"""
    print("🔧 CORRECTION RAPIDE DES IMPORTS")
    print("=" * 40)
    
    # 1. Corriger app/__init__.py
    fix_app_init()
    
    # 2. Corriger utils/__init__.py
    fix_utils_init()
    
    # 3. Corriger tous les autres __init__.py
    fix_all_inits()
    
    # 4. Tester les imports
    success = test_imports()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ CORRECTION RÉUSSIE !")
        print("\n🚀 Vous pouvez maintenant redémarrer le service:")
        print("   python scripts/start_service.py")
        print("   OU")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("❌ Problèmes persistants")
        print("💡 Essayez: python -c 'from app.main import app; print(\"OK\")'")

if __name__ == "__main__":
    main()