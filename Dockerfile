# ==============================================================================
# 🐳 DOCKERFILE - SERVICE IA DE DÉTECTION D'OBJETS PERDUS
# ==============================================================================
# Image Docker optimisée pour FastAPI + PyTorch
# Support CPU et GPU (NVIDIA CUDA)
# Multi-stage build pour optimiser la taille
# ==============================================================================

# 🏗️ ÉTAPE 1: IMAGE DE BASE
# ==============================================================================
# Choisir l'image selon l'environnement cible
# Pour CPU: python:3.11-slim
# Pour GPU: nvidia/cuda:11.8-devel-ubuntu22.04
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} as base

# 📋 Métadonnées de l'image
LABEL maintainer="Équipe L3 Informatique"
LABEL version="1.0.0"
LABEL description="Service IA de détection d'objets perdus"
LABEL org.opencontainers.image.source="https://github.com/votre-repo/ai-detection-service"

# 🌍 Variables d'environnement de base
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# 🔧 Arguments de build
ARG ENVIRONMENT=production
ARG INSTALL_DEV=false
ARG ENABLE_GPU=false

# 📦 ÉTAPE 2: INSTALLATION DES DÉPENDANCES SYSTÈME
# ==============================================================================
FROM base as system-deps

# 🔄 Mise à jour système et installation outils essentiels
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Outils de base
    curl \
    wget \
    git \
    unzip \
    ca-certificates \
    # Compilation
    build-essential \
    gcc \
    g++ \
    # Développement Python
    python3-dev \
    python3-pip \
    # OpenCV et traitement d'images
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # FFmpeg pour vidéos
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Fonts pour génération d'images
    fonts-dejavu-core \
    fontconfig \
    # Nettoyage du cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 🎯 Installation spécifique GPU (si activé)
RUN if [ "$ENABLE_GPU" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    # CUDA toolkit components
    cuda-toolkit-11-8 \
    # cuDNN
    libcudnn8 \
    libcudnn8-dev \
    # TensorRT (optionnel)
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*; \
fi

# ⬆️ ÉTAPE 3: MISE À JOUR PIP ET OUTILS PYTHON
# ==============================================================================
FROM system-deps as python-deps

# 🔧 Mise à jour pip et outils
RUN pip install --no-cache-dir --upgrade \
    pip==23.3.1 \
    setuptools==69.0.2 \
    wheel==0.42.0

# 📦 ÉTAPE 4: INSTALLATION DES DÉPENDANCES PYTHON
# ==============================================================================
FROM python-deps as python-packages

# 📋 Copier les fichiers de requirements
COPY requirements.txt /tmp/requirements.txt

# 🎯 Installation PyTorch selon l'environnement
RUN if [ "$ENABLE_GPU" = "true" ]; then \
    # Version GPU avec CUDA 11.8
    pip install --no-cache-dir torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118; \
else \
    # Version CPU
    pip install --no-cache-dir torch==2.1.1+cpu torchvision==0.16.1+cpu torchaudio==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu; \
fi

# 📦 Installation des autres dépendances
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 🧪 Dépendances de développement (si activées)
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN if [ "$INSTALL_DEV" = "true" ]; then \
    pip install --no-cache-dir -r /tmp/requirements-dev.txt; \
fi

# 🏗️ ÉTAPE 5: PRÉPARATION DE L'APPLICATION
# ==============================================================================
FROM python-packages as app-base

# 👤 Créer utilisateur non-root pour sécurité
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

# 📁 Créer structure de répertoires
RUN mkdir -p /app \
    /app/storage/models \
    /app/storage/temp \
    /app/storage/cache \
    /app/storage/logs \
    /app/storage/uploads \
    /app/storage/results \
    && chown -R aiuser:aiuser /app

# 🏠 Définir répertoire de travail
WORKDIR /app

# 📋 ÉTAPE 6: COPIE DU CODE APPLICATION
# ==============================================================================
FROM app-base as app-code

# 📂 Copier le code source
COPY --chown=aiuser:aiuser . /app/

# 🔧 Copier fichiers de configuration
COPY --chown=aiuser:aiuser .env.docker /app/.env

# 📦 Installation des modèles (optionnel lors du build)
ARG DOWNLOAD_MODELS=false
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
    python scripts/setup_models.py --install-all --check-deps; \
fi

# 🔐 Permissions et sécurité
RUN chmod +x /app/scripts/*.py \
    && chmod 755 /app/main.py

# 🏭 ÉTAPE 7: IMAGE DE PRODUCTION
# ==============================================================================
FROM app-code as production

# 👤 Basculer vers utilisateur non-root
USER aiuser

# 🌐 Exposer les ports
EXPOSE 8001
EXPOSE 8002

# 🏥 Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 📊 Variables d'environnement de production
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV WORKERS=4
ENV HOST=0.0.0.0
ENV PORT=8001

# 🎯 Point d'entrée et commande par défaut
ENTRYPOINT ["python", "-m"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]

# 🧪 ÉTAPE 8: IMAGE DE DÉVELOPPEMENT
# ==============================================================================
FROM app-code as development

# 📦 Installation outils de développement additionnels
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    ipython==8.17.2 \
    debugpy==1.8.0

# 👤 Rester en root pour développement (flexibilité)
USER root

# 🌐 Exposer ports additionnels pour dev
EXPOSE 8001 8002 8888 5678

# 📊 Variables d'environnement de développement
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG
ENV DEBUG=true
ENV RELOAD=true

# 🎯 Commande de développement avec hot reload
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload", "--log-level", "debug"]

# 🐳 ÉTAPE 9: CONFIGURATION MULTI-STAGE
# ==============================================================================
# Par défaut, utiliser l'image de production
FROM production as final

# ==============================================================================
# 📋 INSTRUCTIONS D'UTILISATION
# ==============================================================================
# 
# 🏗️ BUILD:
# 
# # Build basique (CPU)
# docker build -t ai-detection-service .
# 
# # Build avec GPU
# docker build --build-arg ENABLE_GPU=true -t ai-detection-service:gpu .
# 
# # Build développement
# docker build --target development -t ai-detection-service:dev .
# 
# # Build avec téléchargement des modèles
# docker build --build-arg DOWNLOAD_MODELS=true -t ai-detection-service:full .
# 
# 🚀 RUN:
# 
# # Exécution basique
# docker run -p 8001:8001 ai-detection-service
# 
# # Avec GPU
# docker run --gpus all -p 8001:8001 ai-detection-service:gpu
# 
# # Développement avec volume
# docker run -p 8001:8001 -v $(pwd):/app ai-detection-service:dev
# 
# # Avec variables d'environnement
# docker run -p 8001:8001 -e WORKERS=8 -e LOG_LEVEL=DEBUG ai-detection-service
# 
# 🔧 CONFIGURATION:
# 
# Variables d'environnement importantes:
# - ENVIRONMENT: production/development/testing
# - WORKERS: Nombre de workers uvicorn
# - LOG_LEVEL: DEBUG/INFO/WARNING/ERROR
# - ENABLE_GPU: true/false
# - DATABASE_URL: URL de base de données
# - REDIS_URL: URL Redis pour cache
# 
# ==============================================================================

# 📦 VARIANTES D'IMAGES SPÉCIALISÉES
# ==============================================================================

# 🎯 Image GPU NVIDIA
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu-runtime
COPY --from=production /app /app
COPY --from=production /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=production /usr/local/bin /usr/local/bin
WORKDIR /app
USER aiuser
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]

# 📱 Image ultra-légère (Alpine)
FROM python:3.11-alpine as alpine
RUN apk add --no-cache \
    gcc musl-dev linux-headers \
    jpeg-dev zlib-dev freetype-dev lcms2-dev openjpeg-dev tiff-dev tk-dev tcl-dev
COPY --from=python-packages /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=app-code /app /app
WORKDIR /app
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# ==============================================================================
# 📏 OPTIMISATIONS ET BONNES PRATIQUES
# ==============================================================================
# 
# ✅ Multi-stage build pour réduire la taille finale
# ✅ Cache des layers Docker optimisé
# ✅ Utilisateur non-root pour la sécurité
# ✅ Health checks intégrés
# ✅ Support CPU et GPU
# ✅ Variables d'environnement configurables
# ✅ Nettoyage automatique des caches
# ✅ Images spécialisées pour différents cas d'usage
# 
# 🎯 Taille approximative des images:
# - Base (CPU): ~1.2GB
# - GPU: ~2.8GB
# - Development: ~1.5GB
# - Alpine: ~800MB
# 
# ==============================================================================