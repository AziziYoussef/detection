# scripts/start_service.py
#!/usr/bin/env python3
"""
🚀 Script de démarrage du Service IA
Automatise l'initialisation et le démarrage du service
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

def check_python_version():
    """Vérifie la version Python"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")

def check_requirements():
    """Vérifie que les requirements sont installés"""
    try:
        import torch
        import fastapi
        import cv2
        print("✅ Dépendances principales installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        return False

def install_requirements():
    """Installe les requirements"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dépendances installées")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erreur installation dépendances")
        return False

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        "storage",
        "storage/models", 
        "storage/temp",
        "storage/cache",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Répertoire créé: {directory}")

def check_models():
    """Vérifie la présence des modèles"""
    models_dir = Path("storage/models")
    model_files = [
        "stable_model_epoch_30.pth",
        "best_extended_model.pth",
        "fast_stream_model.pth"
    ]
    
    missing_models = []
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"✅ Modèle trouvé: {model_file}")
        else:
            print(f"⚠️  Modèle manquant: {model_file}")
            missing_models.append(model_file)
    
    if missing_models:
        print("\n📥 Pour ajouter vos modèles:")
        print("   1. Placez vos fichiers .pth dans storage/models/")
        print("   2. Ou le service utilisera des modèles par défaut")
        print("   3. Consultez la documentation pour l'entraînement")
    
    return len(missing_models) == 0

def check_env_file():
    """Vérifie/crée le fichier .env"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  Fichier .env manquant, création avec valeurs par défaut...")
        
        default_env = """# Configuration générée automatiquement
HOST=0.0.0.0
PORT=8000
DEBUG=True
USE_GPU=False
CONFIDENCE_THRESHOLD=0.5
"""
        with open(env_file, 'w') as f:
            f.write(default_env)
        print("✅ Fichier .env créé")
    else:
        print("✅ Fichier .env trouvé")

def start_service(port=8000, host="0.0.0.0", reload=True, workers=1):
    """Démarre le service"""
    print(f"\n🚀 Démarrage du service sur {host}:{port}")
    
    cmd = [
        "uvicorn", "app.main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Arrêt du service")
    except FileNotFoundError:
        print("❌ uvicorn non trouvé. Installez avec: pip install uvicorn")

def main():
    parser = argparse.ArgumentParser(description="🔍 Service IA - Détection d'Objets Perdus")
    parser.add_argument("--port", type=int, default=8000, help="Port du serveur")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute")
    parser.add_argument("--no-reload", action="store_true", help="Désactiver reload automatique")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers")
    parser.add_argument("--skip-checks", action="store_true", help="Ignorer les vérifications")
    parser.add_argument("--install", action="store_true", help="Installer les dépendances")
    
    args = parser.parse_args()
    
    print("🔍 SERVICE IA - DÉTECTION D'OBJETS PERDUS")
    print("="*50)
    
    # Installation si demandée
    if args.install:
        if not install_requirements():
            sys.exit(1)
    
    # Vérifications
    if not args.skip_checks:
        print("\n🔍 Vérifications système...")
        check_python_version()
        
        if not check_requirements():
            print("\n📦 Voulez-vous installer les dépendances ? (y/N)")
            if input().lower() == 'y':
                if not install_requirements():
                    sys.exit(1)
            else:
                print("⚠️  Démarrage sans vérification des dépendances")
        
        create_directories()
        check_env_file()
        check_models()
    
    # Démarrage
    print("\n" + "="*50)
    start_service(
        port=args.port,
        host=args.host, 
        reload=not args.no_reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()

