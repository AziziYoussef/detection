# testing_guide.py
"""
🧪 Guide de Test Complet - Service IA
Guide étape par étape pour tester votre service de détection d'objets perdus
"""

import subprocess
import sys
import requests
import json
import time
from pathlib import Path

def print_step(step_num, title, description=""):
    """Affiche une étape de test"""
    print(f"\n{'='*60}")
    print(f"📋 ÉTAPE {step_num}: {title}")
    print(f"{'='*60}")
    if description:
        print(f"💡 {description}")

def run_command(command, description=""):
    """Exécute une commande et affiche le résultat"""
    print(f"\n🔧 Exécution: {command}")
    if description:
        print(f"📝 {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Succès !")
            if result.stdout:
                print(f"📤 Sortie:\n{result.stdout}")
            return True
        else:
            print("❌ Erreur !")
            if result.stderr:
                print(f"💥 Erreur:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False

def test_api_endpoint(url, description="", expected_status=200):
    """Teste un endpoint API"""
    print(f"\n🌐 Test endpoint: {url}")
    print(f"📝 {description}")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == expected_status:
            print(f"✅ Status: {response.status_code}")
            return response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        else:
            print(f"❌ Status incorrect: {response.status_code} (attendu: {expected_status})")
            return None
    except Exception as e:
        print(f"💥 Erreur: {e}")
        return None

def main():
    """Guide de test principal"""
    
    print("🔍 SERVICE IA - GUIDE DE TEST COMPLET")
    print("🎯 Ce guide vous accompagne pour tester votre installation")
    
    # ===============================================
    # ÉTAPE 1: VÉRIFICATION INSTALLATION
    # ===============================================
    
    print_step(1, "VÉRIFICATION DE L'INSTALLATION", 
               "Vérifie que tous les composants sont correctement installés")
    
    print("📋 Lancement du vérificateur d'installation...")
    if run_command("python scripts/check_installation.py"):
        print("✅ Installation vérifiée avec succès !")
    else:
        print("⚠️ Problèmes détectés. Résolvez-les avant de continuer.")
        print("💡 Consultez le rapport d'installation pour les corrections à apporter.")
    
    input("\n⏸️ Appuyez sur Entrée pour continuer...")
    
    # ===============================================
    # ÉTAPE 2: TESTS D'IMPORTATION
    # ===============================================
    
    print_step(2, "TESTS D'IMPORTATION", 
               "Vérifie que tous les modules Python peuvent être importés")
    
    test_imports = """
# Test des imports principaux
try:
    from app import __version__
    print(f"✅ Version: {__version__}")
    
    from app.main import app
    print("✅ Application FastAPI")
    
    from app.core import ObjectDetector, ModelManager
    print("✅ Core components")
    
    from app.utils import ImageProcessor
    print("✅ Utils")
    
    from app.schemas import ObjectDetection
    print("✅ Schemas")
    
    print("🎉 Tous les imports réussis !")
    
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    exit(1)
"""
    
    with open("test_imports.py", "w") as f:
        f.write(test_imports)
    
    if run_command("python test_imports.py", "Test des importations Python"):
        print("✅ Tous les modules s'importent correctement !")
    else:
        print("❌ Problème d'importation. Vérifiez la structure des fichiers.")
        return
    
    # Nettoyage
    Path("test_imports.py").unlink(missing_ok=True)
    
    # ===============================================
    # ÉTAPE 3: DÉMARRAGE DU SERVICE
    # ===============================================
    
    print_step(3, "DÉMARRAGE DU SERVICE",
               "Lance le service IA et vérifie qu'il démarre correctement")
    
    print("🚀 Démarrage du service en arrière-plan...")
    print("⚠️ Si le service ne démarre pas, vérifiez les logs d'erreur")
    
    # On utilise le script de démarrage
    print("💡 Lancez dans un autre terminal :")
    print("   python scripts/start_service.py")
    print("\n   OU avec uvicorn directement :")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    
    print("\n⏳ Attendez que le service démarre, puis appuyez sur Entrée...")
    input()
    
    # ===============================================
    # ÉTAPE 4: TESTS DES ENDPOINTS DE BASE
    # ===============================================
    
    print_step(4, "TESTS DES ENDPOINTS DE BASE",
               "Vérifie que les endpoints principaux répondent")
    
    base_url = "http://localhost:8000"
    
    # Test endpoint racine
    root_data = test_api_endpoint(f"{base_url}/", "Endpoint racine")
    if root_data:
        print(f"📄 Service: {root_data.get('service', 'N/A')}")
    
    # Test endpoint santé
    health_data = test_api_endpoint(f"{base_url}/health", "Vérification santé")
    if health_data:
        print(f"💚 Status: {health_data.get('status', 'N/A')}")
    
    # Test liste des modèles
    models_data = test_api_endpoint(f"{base_url}/api/v1/models/", "Liste des modèles")
    if models_data:
        print(f"🤖 Modèles disponibles: {len(models_data)}")
    
    # Test documentation
    docs_response = test_api_endpoint(f"{base_url}/docs", "Documentation API", expected_status=200)
    
    # ===============================================
    # ÉTAPE 5: TEST DÉTECTION D'IMAGES
    # ===============================================
    
    print_step(5, "TEST DÉTECTION D'IMAGES",
               "Teste la détection d'objets sur une image de test")
    
    # Création d'une image de test
    print("🖼️ Création d'une image de test...")
    
    test_image_script = """
import cv2
import numpy as np

# Création d'une image de test 320x320
image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

# Ajout de formes pour simuler des objets
cv2.rectangle(image, (50, 50), (150, 120), (0, 255, 0), -1)  # Rectangle vert
cv2.circle(image, (250, 100), 40, (255, 0, 0), -1)  # Cercle bleu
cv2.rectangle(image, (100, 200), (250, 280), (0, 0, 255), -1)  # Rectangle rouge

# Sauvegarde
cv2.imwrite('test_image.jpg', image)
print("✅ Image de test créée: test_image.jpg")
"""
    
    with open("create_test_image.py", "w") as f:
        f.write(test_image_script)
    
    if run_command("python create_test_image.py"):
        # Test de détection via API
        print("\n🔍 Test de détection d'image...")
        
        try:
            with open('test_image.jpg', 'rb') as f:
                files = {'file': f}
                data = {
                    'model_name': 'stable_epoch_30',
                    'confidence_threshold': 0.3,
                    'enable_lost_detection': True
                }
                
                response = requests.post(f"{base_url}/api/v1/detect/image", 
                                       files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Détection réussie !")
                print(f"📊 Objets détectés: {result.get('total_objects', 0)}")
                print(f"⚠️ Objets perdus: {result.get('lost_objects', 0)}")
                print(f"⏱️ Temps traitement: {result.get('processing_time', 0):.1f}ms")
            else:
                print(f"❌ Erreur détection: {response.status_code}")
                print(f"💥 Réponse: {response.text}")
        
        except Exception as e:
            print(f"💥 Erreur test détection: {e}")
    
    # Nettoyage
    Path("create_test_image.py").unlink(missing_ok=True)
    Path("test_image.jpg").unlink(missing_ok=True)
    
    # ===============================================
    # ÉTAPE 6: TEST STREAMING (OPTIONNEL)
    # ===============================================
    
    print_step(6, "TEST STREAMING (OPTIONNEL)",
               "Teste l'interface de streaming temps réel")
    
    print("🎮 Interface de démonstration du streaming:")
    print(f"   👉 {base_url}/api/v1/stream/demo")
    print("\n📱 Ouvrez cette URL dans votre navigateur pour tester le streaming")
    print("💡 Vous pourrez tester avec votre webcam ou des images uploadées")
    
    # Test statut streaming
    stream_status = test_api_endpoint(f"{base_url}/api/v1/stream/status", "Status streaming")
    if stream_status:
        print(f"🔗 Connexions actives: {stream_status.get('total_connections', 0)}")
    
    # ===============================================
    # ÉTAPE 7: TESTS UNITAIRES (OPTIONNEL)
    # ===============================================
    
    print_step(7, "TESTS UNITAIRES (OPTIONNEL)",
               "Exécute la suite de tests automatisés")
    
    print("🧪 Vous pouvez exécuter les tests unitaires avec:")
    print("   python -m pytest tests/ -v")
    print("\n💡 Ou pour lancer maintenant:")
    
    run_tests = input("Voulez-vous lancer les tests maintenant ? (y/N): ")
    if run_tests.lower() == 'y':
        run_command("python -m pytest tests/ -v", "Tests unitaires")
    
    # ===============================================
    # ÉTAPE 8: RÉSUMÉ ET PROCHAINES ÉTAPES
    # ===============================================
    
    print_step(8, "RÉSUMÉ ET PROCHAINES ÉTAPES",
               "Récapitulatif et guide pour la suite")
    
    print("🎉 FÉLICITATIONS ! Votre Service IA est fonctionnel !")
    
    print("\n📚 RESSOURCES DISPONIBLES:")
    print(f"   📖 Documentation API: {base_url}/docs")
    print(f"   🎮 Interface streaming: {base_url}/api/v1/stream/demo") 
    print(f"   💚 Santé du service: {base_url}/health")
    print(f"   🤖 Gestion modèles: {base_url}/api/v1/models/")
    
    print("\n🚀 PROCHAINES ÉTAPES:")
    print("   1. 📁 Ajoutez vos propres modèles dans storage/models/")
    print("   2. ⚙️ Configurez .env selon vos besoins")
    print("   3. 🎬 Testez avec vos propres vidéos/images")
    print("   4. 🔧 Personnalisez la logique métier dans core/detector.py")
    print("   5. 🐳 Déployez avec Docker en production")
    
    print("\n🛠️ DÉVELOPPEMENT:")
    print("   • Code source dans app/")
    print("   • Tests dans tests/")
    print("   • Configuration dans app/config/")
    print("   • Logs dans logs/")
    
    print("\n📞 SUPPORT:")
    print("   • Consultez README.md pour la documentation complète")
    print("   • Vérifiez logs/ en cas de problème")
    print("   • Utilisez scripts/check_installation.py pour diagnostic")

if __name__ == "__main__":
    main()