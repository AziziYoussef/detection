# === SCRIPT BASH ÉQUIVALENT ===
# scripts/start_service.sh
#!/bin/bash
# 🚀 Script de démarrage bash (alternative)

set -e

echo "🔍 SERVICE IA - DÉTECTION D'OBJETS PERDUS"
echo "============================================"

# Vérification Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé"
    exit 1
fi

echo "✅ Python trouvé: $(python3 --version)"

# Création des répertoires
mkdir -p storage/{models,temp,cache} logs
echo "📁 Répertoires créés"

# Vérification .env
if [ ! -f ".env" ]; then
    echo "⚠️  Création fichier .env par défaut..."
    cat > .env << 'EOF'
HOST=0.0.0.0
PORT=8000
DEBUG=True
USE_GPU=False
EOF
    echo "✅ Fichier .env créé"
fi

# Installation dépendances si nécessaire
if [ "$1" = "--install" ]; then
    echo "📦 Installation des dépendances..."
    pip3 install -r requirements.txt
    echo "✅ Dépendances installées"
fi

# Démarrage
echo "🚀 Démarrage du service..."
echo "📖 Documentation: http://localhost:8000/docs"
echo "🎮 Interface streaming: http://localhost:8000/api/v1/stream/demo"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload