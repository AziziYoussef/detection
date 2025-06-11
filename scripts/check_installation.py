# scripts/check_installation.py
#!/usr/bin/env python3
"""
🔍 Script de Vérification d'Installation
Vérifie que tous les composants du Service IA sont correctement installés et configurés
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
import requests
import time
import json
from typing import List, Tuple, Dict

class InstallationChecker:
    """Vérificateur d'installation complet"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_total = 0
        self.warnings = []
        self.errors = []
        
    def check(self, description: str, condition: bool, fix_suggestion: str = None):
        """Exécute une vérification"""
        self.checks_total += 1
        
        if condition:
            print(f"✅ {description}")
            self.checks_passed += 1
        else:
            print(f"❌ {description}")
            self.errors.append(description)
            if fix_suggestion:
                print(f"   💡 Solution: {fix_suggestion}")
    
    def warning(self, description: str, suggestion: str = None):
        """Ajoute un avertissement"""
        print(f"⚠️  {description}")
        self.warnings.append(description)
        if suggestion:
            print(f"   💡 Suggestion: {suggestion}")
    
    def info(self, message: str):
        """Message informatif"""
        print(f"ℹ️  {message}")

def check_python_version():
    """Vérifie la version Python"""
    checker = InstallationChecker()
    
    print("\n🐍 VÉRIFICATION PYTHON")
    print("=" * 30)
    
    version = sys.version_info
    checker.check(
        f"Python {version.major}.{version.minor}.{version.micro}",
        version >= (3, 8),
        "Installez Python 3.8 ou plus récent"
    )
    
    return checker

def check_dependencies():
    """Vérifie les dépendances Python"""
    checker = InstallationChecker()
    
    print("\n📦 VÉRIFICATION DÉPENDANCES")
    print("=" * 30)
    
    required_packages = [
        ("torch", "PyTorch pour l'IA"),
        ("torchvision", "Vision par ordinateur"),
        ("fastapi", "Framework web"),
        ("uvicorn", "Serveur ASGI"),
        ("opencv-python", "Traitement d'images"),
        ("numpy", "Calculs numériques"),
        ("pydantic", "Validation données"),
        ("websockets", "Communication temps réel"),
        ("psutil", "Monitoring système")
    ]
    
    for package, description in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            checker.check(f"{package} - {description}", True)
        except ImportError:
            checker.check(
                f"{package} - {description}", 
                False,
                f"pip install {package}"
            )
    
    return checker

def check_pytorch_setup():
    """Vérifie la configuration PyTorch"""
    checker = InstallationChecker()
    
    print("\n🤖 VÉRIFICATION PYTORCH")
    print("=" * 30)
    
    try:
        import torch
        
        # Version PyTorch
        version = torch.__version__
        checker.check(f"PyTorch version {version}", True)
        
        # Support CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            checker.check(f"CUDA disponible: {device_name}", True)
            checker.info(f"Périphériques GPU: {device_count}")
        else:
            checker.warning(
                "CUDA non disponible - utilisation CPU",
                "Installez CUDA et PyTorch GPU pour de meilleures performances"
            )
        
        # Test simple
        try:
            x = torch.randn(2, 3)
            y = x.sum()
            checker.check("Test calcul PyTorch", True)
        except Exception as e:
            checker.check("Test calcul PyTorch", False, str(e))
    
    except ImportError:
        checker.check("PyTorch importé", False, "pip install torch torchvision")
    
    return checker

def check_file_structure():
    """Vérifie la structure des fichiers"""
    checker = InstallationChecker()
    
    print("\n📁 VÉRIFICATION STRUCTURE")
    print("=" * 30)
    
    required_files = [
        ("app/main.py", "Point d'entrée principal"),
        ("app/config/config.py", "Configuration"),
        ("app/core/detector.py", "Détecteur principal"),
        ("app/core/model_manager.py", "Gestionnaire modèles"),
        ("app/api/routes.py", "Routes API"),
        ("requirements.txt", "Dépendances"),
        (".env", "Configuration environnement")
    ]
    
    required_dirs = [
        ("storage", "Stockage général"),
        ("storage/models", "Modèles AI"),
        ("storage/temp", "Fichiers temporaires"),
        ("storage/cache", "Cache"),
        ("logs", "Logs du système")
    ]
    
    # Vérification fichiers
    for file_path, description in required_files:
        path = Path(file_path)
        checker.check(
            f"{file_path} - {description}",
            path.exists(),
            f"Créez le fichier {file_path}"
        )
    
    # Vérification répertoires
    for dir_path, description in required_dirs:
        path = Path(dir_path)
        if path.exists():
            checker.check(f"{dir_path}/ - {description}", True)
        else:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            checker.check(f"{dir_path}/ - {description} (créé)", True)
    
    return checker

def check_models():
    """Vérifie la présence des modèles"""
    checker = InstallationChecker()
    
    print("\n🏆 VÉRIFICATION MODÈLES")
    print("=" * 30)
    
    models_dir = Path("storage/models")
    expected_models = [
        "stable_model_epoch_30.pth",
        "best_extended_model.pth", 
        "fast_stream_model.pth"
    ]
    
    found_models = 0
    for model_file in expected_models:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            checker.check(f"{model_file} ({size_mb:.1f} MB)", True)
            found_models += 1
        else:
            checker.warning(
                f"Modèle manquant: {model_file}",
                "Placez vos modèles .pth dans storage/models/"
            )
    
    if found_models == 0:
        checker.warning(
            "Aucun modèle trouvé",
            "Le service utilisera des modèles par défaut (démonstration uniquement)"
        )
    
    return checker

def check_configuration():
    """Vérifie la configuration"""
    checker = InstallationChecker()
    
    print("\n⚙️  VÉRIFICATION CONFIGURATION")
    print("=" * 30)
    
    # Fichier .env
    env_file = Path(".env")
    if env_file.exists():
        checker.check("Fichier .env présent", True)
        
        # Lecture et vérification des variables importantes
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
                
            important_vars = [
                "HOST", "PORT", "DEBUG", "USE_GPU",
                "CONFIDENCE_THRESHOLD", "SUSPECT_THRESHOLD_SECONDS"
            ]
            
            for var in important_vars:
                if var in env_content:
                    checker.check(f"Variable {var} configurée", True)
                else:
                    checker.warning(f"Variable {var} manquante")
        
        except Exception as e:
            checker.warning(f"Erreur lecture .env: {e}")
    
    else:
        checker.warning(
            "Fichier .env manquant",
            "Copiez .env.example vers .env et configurez"
        )
    
    return checker

def check_service_startup():
    """Teste le démarrage du service"""
    checker = InstallationChecker()
    
    print("\n🚀 TEST DÉMARRAGE SERVICE")
    print("=" * 30)
    
    try:
        # Test d'import des modules principaux
        from app.main import app
        checker.check("Import de l'application", True)
        
        from app.config.config import settings
        checker.check("Configuration chargée", True)
        
        from app.core.model_manager import ModelManager
        checker.check("ModelManager importé", True)
        
        # Test création ModelManager
        try:
            model_manager = ModelManager()
            checker.check("ModelManager instancié", True)
        except Exception as e:
            checker.check("ModelManager instancié", False, str(e))
    
    except ImportError as e:
        checker.check("Import modules", False, f"Erreur: {e}")
    
    return checker

def check_running_service():
    """Vérifie si le service est en cours d'exécution"""
    checker = InstallationChecker()
    
    print("\n🌐 TEST SERVICE EN COURS")
    print("=" * 30)
    
    ports_to_check = [8000, 8001, 8080]
    service_found = False
    
    for port in ports_to_check:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                checker.check(f"Service actif sur port {port}", True)
                checker.info(f"Status: {data.get('status', 'unknown')}")
                service_found = True
                
                # Tests endpoints supplémentaires
                try:
                    models_response = requests.get(f"http://localhost:{port}/api/v1/models/", timeout=5)
                    checker.check("Endpoint modèles accessible", models_response.status_code == 200)
                except:
                    checker.warning("Endpoint modèles inaccessible")
                
                break
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            checker.warning(f"Erreur test port {port}: {e}")
    
    if not service_found:
        checker.warning(
            "Service non démarré",
            "Lancez: python scripts/start_service.py"
        )
    
    return checker

def check_docker_setup():
    """Vérifie la configuration Docker"""
    checker = InstallationChecker()
    
    print("\n🐳 VÉRIFICATION DOCKER")
    print("=" * 30)
    
    # Docker installé
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            checker.check(f"Docker installé: {version}", True)
        else:
            checker.warning("Docker non fonctionnel")
    except FileNotFoundError:
        checker.warning(
            "Docker non installé",
            "Installez Docker pour utiliser le déploiement containerisé"
        )
    
    # Docker Compose
    try:
        result = subprocess.run(["docker-compose", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            checker.check("Docker Compose disponible", True)
        else:
            checker.warning("Docker Compose non fonctionnel")
    except FileNotFoundError:
        checker.warning("Docker Compose non installé")
    
    # Fichiers Docker
    docker_files = [
        ("Dockerfile", "Image Docker"),
        ("docker-compose.yml", "Orchestration Docker"),
        (".dockerignore", "Exclusions Docker")
    ]
    
    for file_path, description in docker_files:
        path = Path(file_path)
        if path.exists():
            checker.check(f"{file_path} - {description}", True)
        else:
            checker.warning(f"{file_path} manquant")
    
    return checker

def generate_report(checkers: List[InstallationChecker]):
    """Génère un rapport final"""
    print("\n" + "=" * 60)
    print("📊 RAPPORT D'INSTALLATION")
    print("=" * 60)
    
    total_checks = sum(c.checks_total for c in checkers)
    total_passed = sum(c.checks_passed for c in checkers)
    total_warnings = sum(len(c.warnings) for c in checkers)
    total_errors = sum(len(c.errors) for c in checkers)
    
    success_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0
    
    print(f"✅ Tests réussis: {total_passed}/{total_checks} ({success_rate:.1f}%)")
    print(f"⚠️  Avertissements: {total_warnings}")
    print(f"❌ Erreurs: {total_errors}")
    
    # Évaluation globale
    if success_rate >= 90 and total_errors == 0:
        status = "🟢 EXCELLENT"
        message = "Installation parfaite ! Vous pouvez utiliser tous les services."
    elif success_rate >= 75 and total_errors <= 2:
        status = "🟡 BON"
        message = "Installation fonctionnelle avec quelques améliorations possibles."
    elif success_rate >= 50:
        status = "🟠 MOYEN"
        message = "Installation partielle. Certaines fonctionnalités peuvent ne pas marcher."
    else:
        status = "🔴 PROBLÉMATIQUE"
        message = "Installation incomplète. Résolvez les erreurs avant de continuer."
    
    print(f"\n{status}")
    print(f"📝 {message}")
    
    # Prochaines étapes
    print(f"\n🎯 PROCHAINES ÉTAPES")
    print("-" * 30)
    
    if total_errors > 0:
        print("1. 🔧 Résolvez les erreurs marquées ❌")
        print("2. ♻️  Relancez cette vérification")
    else:
        print("1. 🚀 Démarrez le service: python scripts/start_service.py")
        print("2. 🌐 Ouvrez http://localhost:8000/docs")
        print("3. 🎮 Testez avec http://localhost:8000/api/v1/stream/demo")
        
        if total_warnings > 0:
            print("4. ⚠️  Examinez les avertissements pour optimiser")

def main():
    """Fonction principale"""
    print("🔍 SERVICE IA - VÉRIFICATION D'INSTALLATION")
    print("=" * 60)
    print("Ce script vérifie que votre installation est complète et fonctionnelle.\n")
    
    # Exécution des vérifications
    checkers = [
        check_python_version(),
        check_dependencies(),
        check_pytorch_setup(),
        check_file_structure(),
        check_models(),
        check_configuration(),
        check_service_startup(),
        check_running_service(),
        check_docker_setup()
    ]
    
    # Rapport final
    generate_report(checkers)
    
    # Sauvegarde du rapport
    report_data = {
        "timestamp": time.time(),
        "total_checks": sum(c.checks_total for c in checkers),
        "passed_checks": sum(c.checks_passed for c in checkers),
        "warnings": sum(len(c.warnings) for c in checkers),
        "errors": sum(len(c.errors) for c in checkers),
        "success_rate": sum(c.checks_passed for c in checkers) / sum(c.checks_total for c in checkers) * 100
    }
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    with open(logs_dir / "installation_check.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Rapport sauvegardé: logs/installation_check.json")

if __name__ == "__main__":
    main()