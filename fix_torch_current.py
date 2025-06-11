# fix_torch_current.py
"""
🔥 Correction Torch avec Versions Actuelles
Installe les versions actuellement disponibles de PyTorch
"""

import subprocess
import sys

def run_command(command, description=""):
    """Exécute une commande avec gestion d'erreurs"""
    print(f"\n🔧 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("   ✅ Succès!")
        if result.stdout:
            print(f"   📤 Sortie: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erreur: {e}")
        if e.stdout:
            print(f"   📤 Sortie: {e.stdout}")
        if e.stderr:
            print(f"   💥 Erreur: {e.stderr}")
        return False

def install_current_torch():
    """Installe les versions actuelles de torch/torchvision"""
    print("🔥 INSTALLATION TORCH VERSIONS ACTUELLES")
    print("=" * 50)
    
    # Installer la version CPU actuelle disponible
    success = run_command(
        "pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu",
        "Installation torch/torchvision CPU versions actuelles"
    )
    
    if not success:
        print("\n⚠️ Tentative avec versions par défaut...")
        success = run_command(
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", 
            "Installation versions par défaut CPU"
        )
    
    if not success:
        print("\n⚠️ Tentative installation simple...")
        success = run_command(
            "pip install torch torchvision",
            "Installation simple torch/torchvision"
        )
    
    return success

def test_torch_install():
    """Teste l'installation de torch"""
    print("\n🧪 TEST INSTALLATION TORCH")
    print("=" * 30)
    
    try:
        import torch
        print(f"✅ Torch installé: {torch.__version__}")
        
        import torchvision
        print(f"✅ Torchvision installé: {torchvision.__version__}")
        
        # Test calcul simple
        x = torch.randn(2, 3)
        y = x.sum()
        print(f"✅ Calcul test: {y.item():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_full_imports():
    """Teste tous les imports du service"""
    print("\n🧪 TEST IMPORTS COMPLETS")
    print("=" * 30)
    
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
        
        print("🔍 Test app.config...")
        from app.config.config import settings
        print("✅ app.config")
        
        print("🔍 Test app.main...")
        from app.main import app
        print("✅ app.main - APPLICATION PRÊTE!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    print("🔥 CORRECTION TORCH - VERSIONS ACTUELLES")
    print("=" * 60)
    
    # 1. Installer torch actuel
    torch_ok = install_current_torch()
    
    # 2. Tester torch
    if torch_ok:
        torch_test_ok = test_torch_install()
    else:
        torch_test_ok = False
    
    # 3. Tester tous les imports
    if torch_test_ok:
        all_imports_ok = test_full_imports()
    else:
        all_imports_ok = False
    
    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT FINAL")
    print("=" * 60)
    
    if all_imports_ok:
        print("🎉 SUCCÈS COMPLET!")
        print("\n✅ Toutes les dépendances sont installées")
        print("✅ Tous les imports fonctionnent")
        print("✅ L'application est prête")
        
        print("\n🚀 DÉMARRAGE:")
        print("   python scripts/start_service.py")
        print("   OU")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        
        print("\n📱 INTERFACES:")
        print("   • Documentation: http://localhost:8000/docs")
        print("   • Health Check: http://localhost:8000/health")
        print("   • Streaming Demo: http://localhost:8000/api/v1/stream/demo")
        
    else:
        print("❌ PROBLÈMES PERSISTANTS")
        
        if not torch_ok:
            print("\n💡 SOLUTION ALTERNATIVE - INSTALLATION CONDA:")
            print("   conda install pytorch torchvision cpuonly -c pytorch")
        
        print("\n💡 OU SOLUTION MANUELLE:")
        print("   1. pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("   2. pip install torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("   3. python -c 'import torch; print(torch.__version__)'")

if __name__ == "__main__":
    main()