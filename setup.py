"""
Setup script for Lost Objects Detection Service
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Package metadata
setup(
    name="lost-objects-detection",
    version="1.0.0",
    author="Your Organization",
    author_email="contact@yourorganization.com",
    description="AI-powered real-time detection and tracking system for lost objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/lost-objects-detection",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Security",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
        ],
        "gpu": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "google-cloud-storage>=2.8.0",
            "azure-storage-blob>=12.14.0",
        ],
        "production": [
            "gunicorn>=21.2.0",
            "uvicorn[standard]>=0.24.0",
            "redis>=5.0.1",
            "psycopg2-binary>=2.9.5",
        ],
        "all": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
            "boto3>=1.26.0",
            "google-cloud-storage>=2.8.0",
            "azure-storage-blob>=12.14.0",
            "gunicorn>=21.2.0",
            "uvicorn[standard]>=0.24.0",
            "redis>=5.0.1",
            "psycopg2-binary>=2.9.5",
        ]
    },
    entry_points={
        "console_scripts": [
            "lost-objects-server=app.main:main",
            "lost-objects-deploy=scripts.deploy:main",
            "lost-objects-test=scripts.test_service:main",
            "lost-objects-cache=scripts.cache_manager:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": [
            "config/*.json",
            "config/*.yaml",
            "config/*.yml",
        ],
        "": [
            "*.md",
            "*.txt",
            "*.yml",
            "*.yaml",
            "*.json",
        ]
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "computer vision", 
        "object detection",
        "lost objects",
        "surveillance",
        "security",
        "real-time",
        "pytorch",
        "fastapi",
        "machine learning",
        "deep learning",
        "video analysis",
        "streaming",
        "REST API",
        "WebSocket"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/lost-objects-detection/issues",
        "Source": "https://github.com/your-org/lost-objects-detection",
        "Documentation": "https://your-org.github.io/lost-objects-detection/",
        "Changelog": "https://github.com/your-org/lost-objects-detection/blob/main/CHANGELOG.md",
    },
    license="MIT",
    platforms=["any"],
    
    # Additional metadata
    maintainer="Your Organization",
    maintainer_email="maintain@yourorganization.com",
    
    # Package discovery
    packages_dir={"": "."},
    
    # Data files
    data_files=[
        ("config", [
            "config/deployment.json",
            "docker-compose.yml",
        ]),
        ("docs", [
            "README.md",
        ]),
    ],
)

# Development installation helper
if __name__ == "__main__":
    import sys
    import subprocess
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        print("Installing in development mode...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".[dev,all]"
        ])
    elif len(sys.argv) > 1 and sys.argv[1] == "prod":
        print("Installing for production...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", ".[production]"
        ])
    else:
        # Standard setup
        from setuptools import setup
        setup()