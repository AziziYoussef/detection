# 🔍 Lost Objects Detection Service

**AI-powered real-time detection and tracking system for lost objects in public spaces**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 **Overview**

The Lost Objects Detection Service is a comprehensive AI-powered system designed to automatically detect, track, and alert about potentially lost or abandoned objects in public spaces such as airports, train stations, and offices.

### ✨ **Key Features**

- 🤖 **AI-Powered Detection**: Advanced object detection using PyTorch and computer vision
- 📡 **Real-time Streaming**: WebSocket-based live video processing
- 🎬 **Video Analysis**: Batch processing of video files with timeline analysis
- 📦 **Batch Processing**: Efficient bulk image processing
- 🔄 **Temporal Tracking**: Intelligent object state management (Normal → Suspect → Lost)
- 🌐 **RESTful API**: Comprehensive API with automatic documentation
- 🐳 **Docker Ready**: Full containerization with GPU support
- 📊 **Monitoring**: Built-in metrics and health checks
- ⚡ **High Performance**: Optimized for production workloads

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────┐
│            FastAPI Application           │
├─────────────────────────────────────────┤
│  🌐 API Layer                           │
│  ├── Image Detection                    │
│  ├── Video Processing                   │
│  ├── Batch Processing                   │
│  └── WebSocket Streaming                │
├─────────────────────────────────────────┤
│  🧠 Business Logic                      │
│  ├── Object Detector                    │
│  ├── Model Manager                      │
│  ├── Temporal Tracking                  │
│  └── Lost Object Logic                  │
├─────────────────────────────────────────┤
│  🤖 AI Models                           │
│  ├── MobileNet/EfficientNet Backbone    │
│  ├── Feature Pyramid Network            │
│  ├── Detection Heads                    │
│  └── Multiple Model Support             │
├─────────────────────────────────────────┤
│  💾 Storage & Cache                     │
│  ├── Model Storage                      │
│  ├── Result Cache                       │
│  └── Temporary Files                    │
└─────────────────────────────────────────┘
```

## 📋 **Quick Start**

### **Prerequisites**

- Python 3.8+
- Docker & Docker Compose (recommended)
- CUDA-compatible GPU (optional, for acceleration)

### **🐳 Docker Deployment (Recommended)**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lost-objects-detection
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start services**
   ```bash
   # CPU-only deployment
   docker-compose up -d
   
   # GPU-enabled deployment
   docker-compose --profile gpu up -d
   
   # With monitoring
   docker-compose --profile monitoring up -d
   ```

4. **Verify deployment**
   ```bash
   curl http://localhost:8000/health
   ```

### **🐍 Local Development**

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download models** (place in `storage/models/`)
   ```bash
   python scripts/cache_manager.py download stable_model_epoch_30 <MODEL_URL>
   ```

3. **Start development server**
   ```bash
   uvicorn app.main:app --reload
   ```

## 📚 **API Usage**

### **🖼️ Image Detection**

```bash
curl -X POST "http://localhost:8000/api/v1/detect/image" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### **🎬 Video Processing**

```bash
# Start video processing job
curl -X POST "http://localhost:8000/api/v1/detect/video/upload" \
     -F "file=@video.mp4"

# Check job status
curl "http://localhost:8000/api/v1/detect/video/job/{job_id}/status"
```

### **📦 Batch Processing**

```bash
# Upload multiple images
curl -X POST "http://localhost:8000/api/v1/detect/batch/upload" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

### **📡 WebSocket Streaming**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream/client_123');

// Send image frame
ws.send(imageBlob);

// Receive detection results
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log('Detection result:', result);
};
```

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Service Configuration
API_PORT=8000
LOG_LEVEL=info
ENVIRONMENT=production

# Model Configuration
MODELS_DIR=./storage/models
DEFAULT_MODEL=stable_model_epoch_30
CONFIDENCE_THRESHOLD=0.3

# Database Configuration
POSTGRES_PASSWORD=your_secure_password
REDIS_PORT=6379

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### **Model Configuration**

Models are configured in `app/config/model_config.py`:

```python
# Available model variants
MODEL_VARIANTS = {
    'production': 'Balanced model for production use',
    'lightweight': 'Fast model for edge deployment',
    'high_accuracy': 'High precision for critical applications',
    'streaming': 'Optimized for real-time processing'
}
```

## 🤖 **Models**

### **Supported Models**

| Model | Description | Performance | Use Case |
|-------|-------------|-------------|----------|
| **Epoch 30** | Production model | F1=49.86%, ~100ms | General detection |
| **Extended** | 28-class model | High precision | Detailed classification |
| **Fast Stream** | Real-time optimized | ~30ms | Live streaming |
| **Mobile** | Edge deployment | Lightweight | Mobile/IoT devices |

### **Model Management**

```bash
# List available models
python scripts/cache_manager.py list

# Download new model
python scripts/cache_manager.py download model_name URL

# Verify model integrity
python scripts/cache_manager.py verify

# Backup models
python scripts/cache_manager.py backup models_backup.zip
```

## 📊 **Monitoring & Analytics**

### **Health Checks**

- **Service Health**: `GET /health`
- **Model Status**: `GET /api/v1/models`
- **System Stats**: `GET /stats`

### **Metrics Dashboard**

Access Grafana dashboard at `http://localhost:3000` (when monitoring profile is enabled)

Default credentials: `admin/admin`

### **Logs**

```bash
# View service logs
docker-compose logs -f lost-objects-api

# View specific service logs
docker-compose logs -f nginx redis postgres
```

## 🧪 **Testing**

### **Run Test Suite**

```bash
# Basic functionality tests
python scripts/test_service.py --quick

# Comprehensive test suite
python scripts/test_service.py

# Unit tests
pytest tests/
```

### **Performance Benchmarking**

```bash
# Benchmark detection speed
python scripts/test_service.py --url http://localhost:8000
```

## 🚀 **Deployment**

### **Production Deployment**

```bash
# Deploy with all optimizations
python scripts/deploy.py deploy --config config/production.json

# Monitor deployment
python scripts/deploy.py status

# Scale services
docker-compose up -d --scale lost-objects-api=3
```

### **Environment-Specific Configurations**

- **Airport**: Extended monitoring, 10-minute thresholds
- **Train Station**: High-traffic optimizations
- **Office**: Longer thresholds, work-hours only
- **Public Space**: Weather-dependent adjustments

## 🛠️ **Development**

### **Project Structure**

```
lost-objects-detection/
├── app/                     # Main application
│   ├── api/                 # API endpoints
│   ├── core/                # Business logic
│   ├── models/              # AI model definitions
│   ├── services/            # Service layer
│   ├── utils/               # Utilities
│   ├── config/              # Configuration
│   └── schemas/             # Pydantic schemas
├── storage/                 # Data storage
│   ├── models/              # Trained models
│   ├── cache/               # Cache files
│   └── temp/                # Temporary files
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── config/                  # Configuration files
├── docs/                    # Documentation
└── docker-compose.yml       # Container orchestration
```

### **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### **Code Style**

```bash
# Format code
black app/ tests/ scripts/

# Lint code
flake8 app/ tests/ scripts/

# Type checking
mypy app/
```

## 📈 **Performance Optimization**

### **GPU Acceleration**

- CUDA 11.8+ support
- Multi-GPU deployment
- Mixed precision training
- TensorRT optimization (optional)

### **Scaling Options**

- Horizontal scaling with load balancer
- Model caching and optimization
- Async processing pipelines
- Resource-based auto-scaling

## 🔒 **Security**

- Non-root container execution
- Environment-based secrets
- API rate limiting
- Input validation and sanitization
- SSL/TLS support

## 📞 **Support**

- **Documentation**: [docs/](docs/)
- **API Reference**: `http://localhost:8000/docs`
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- PyTorch team for the deep learning framework
- FastAPI team for the modern web framework
- OpenCV community for computer vision tools
- Contributors and beta testers

---

**🚀 Ready to detect lost objects? Get started with the [Quick Start](#-quick-start) guide!**