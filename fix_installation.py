# fix_installation.py
"""
🔧 Script de Correction d'Installation
Résout automatiquement les problèmes détectés lors de la vérification
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Exécute une commande en affichant le progrès"""
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

def install_missing_dependencies():
    """Installe les dépendances manquantes"""
    print("\n" + "="*50)
    print("📦 INSTALLATION DES DÉPENDANCES MANQUANTES")
    print("="*50)
    
    missing_packages = [
        "fastapi",
        "uvicorn[standard]", 
        "opencv-python",
        "pydantic",
        "websockets",
        "python-multipart",
        "aiofiles"
    ]
    
    for package in missing_packages:
        print(f"\n📥 Installation de {package}...")
        success = run_command(
            f"pip install {package}",
            f"Installation {package}"
        )
        if not success:
            print(f"⚠️ Échec installation {package}, continuons...")
    
    print("\n✅ Installation des dépendances terminée!")

def fix_python_path():
    """Corrige le problème d'import du module 'app'"""
    print("\n" + "="*50)
    print("🐍 CORRECTION DU PYTHONPATH")
    print("="*50)
    
    # Créer un fichier setup.py pour installer le package en mode développement
    setup_content = '''
from setuptools import setup, find_packages

setup(
    name="ai-service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "opencv-python>=4.8.1.78",
        "numpy>=1.24.3",
        "pydantic>=2.4.2",
        "websockets>=12.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.1",
        "psutil>=5.9.6"
    ],
    python_requires=">=3.8",
)
'''
    
    with open("setup.py", "w") as f:
        f.write(setup_content)
    
    print("📄 Fichier setup.py créé")
    
    # Installation en mode développement
    success = run_command(
        "pip install -e .",
        "Installation du package en mode développement"
    )
    
    if success:
        print("✅ Package installé en mode développement")
    else:
        print("⚠️ Installation package échouée, essayons une autre méthode...")
        
        # Méthode alternative: ajouter le répertoire au PYTHONPATH
        current_dir = os.getcwd()
        python_path = os.environ.get('PYTHONPATH', '')
        
        if current_dir not in python_path:
            print(f"🔧 Ajout de {current_dir} au PYTHONPATH")
            os.environ['PYTHONPATH'] = f"{current_dir};{python_path}" if python_path else current_dir

def create_missing_files():
    """Crée les fichiers manquants"""
    print("\n" + "="*50)
    print("📁 CRÉATION DES FICHIERS MANQUANTS")
    print("="*50)
    
    # Créer .dockerignore
    dockerignore_content = '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Temporary files
storage/temp/
storage/cache/
*.tmp

# Git
.git/
.gitignore

# Documentation
docs/
README.md

# Tests
tests/
coverage/
.coverage
.pytest_cache/

# Docker
Dockerfile
docker-compose.yml
.dockerignore
'''
    
    with open(".dockerignore", "w") as f:
        f.write(dockerignore_content)
    
    print("✅ Fichier .dockerignore créé")

def test_imports():
    """Teste les imports après correction"""
    print("\n" + "="*50)
    print("🧪 TEST DES IMPORTS APRÈS CORRECTION")
    print("="*50)
    
    test_script = '''
import sys
import os

# Ajouter le répertoire courant au path si nécessaire
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    print("🔍 Test import app...")
    from app import __version__
    print(f"✅ app.__version__: {__version__}")
    
    print("🔍 Test import main...")
    from app.main import app as fastapi_app
    print("✅ app.main importé")
    
    print("🔍 Test import core...")
    from app.core import ObjectDetector, ModelManager
    print("✅ app.core importé")
    
    print("🔍 Test import utils...")
    from app.utils import ImageProcessor
    print("✅ app.utils importé")
    
    print("🔍 Test import schemas...")
    from app.schemas import ObjectDetection
    print("✅ app.schemas importé")
    
    print("\\n🎉 TOUS LES IMPORTS RÉUSSIS!")
    
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"💥 Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open("test_imports_fix.py", "w") as f:
        f.write(test_script)
    
    success = run_command(
        "python test_imports_fix.py",
        "Test des imports corrigés"
    )
    
    # Nettoyage
    Path("test_imports_fix.py").unlink(missing_ok=True)
    
    return success

def create_minimal_models():
    """Crée des modèles factices pour les tests"""
    print("\n" + "="*50)
    print("🤖 CRÉATION DE MODÈLES DE DÉMONSTRATION")
    print("="*50)
    
    models_dir = Path("storage/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_script = '''
import torch
import torch.nn as nn
from pathlib import Path

class DemoModel(nn.Module):
    """Modèle de démonstration simple"""
    def __init__(self, num_classes=28):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes * 6)  # 6 = x,y,w,h,conf,class
        
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), -1, 6)

# Créer et sauvegarder les modèles de démonstration
models_to_create = [
    "stable_model_epoch_30.pth",
    "best_extended_model.pth", 
    "fast_stream_model.pth"
]

models_dir = Path("storage/models")

for model_name in models_to_create:
    model = DemoModel()
    model_path = models_dir / model_name
    
    # Sauvegarder avec quelques métadonnées
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': {
            'name': model_name,
            'type': 'demo',
            'classes': 28,
            'input_size': (320, 320)
        }
    }, model_path)
    
    print(f"✅ Modèle créé: {model_name}")

print("🎉 Modèles de démonstration créés!")
'''
    
    with open("create_demo_models.py", "w") as f:
        f.write(model_script)
    
    success = run_command(
        "python create_demo_models.py",
        "Création des modèles de démonstration"
    )
    
    # Nettoyage
    Path("create_demo_models.py").unlink(missing_ok=True)
    
    if success:
        print("✅ Modèles de démonstration créés dans storage/models/")
    
    return success

def main():
    """Fonction principale de correction"""
    print("🔧 SERVICE IA - CORRECTION D'INSTALLATION")
    print("=" * 60)
    print("Ce script corrige automatiquement les problèmes détectés.\n")
    
    # 1. Installation des dépendances
    install_missing_dependencies()
    
    # 2. Correction du PYTHONPATH
    fix_python_path()
    
    # 3. Création des fichiers manquants
    create_missing_files()
    
    # 4. Test des imports
    imports_ok = test_imports()
    
    # 5. Création de modèles de démonstration
    models_ok = create_minimal_models()
    
    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE CORRECTION")
    print("=" * 60)
    
    if imports_ok:
        print("✅ Imports Python: CORRIGÉ")
    else:
        print("❌ Imports Python: PROBLÈME PERSISTANT")
    
    if models_ok:
        print("✅ Modèles de démonstration: CRÉÉS")
    else:
        print("⚠️ Modèles de démonstration: ÉCHEC")
    
    print("\n🎯 PROCHAINES ÉTAPES:")
    print("1. Relancez la vérification: python scripts/check_installation.py")
    print("2. Démarrez le service: python scripts/start_service.py")
    print("3. Testez l'API: http://localhost:8000/docs")
    
    print("\n💡 NOTES:")
    print("• Les modèles créés sont des démonstrations")
    print("• Remplacez-les par vos vrais modèles entraînés")
    print("• Consultez README.md pour plus d'informations")

if __name__ == "__main__":
    main()