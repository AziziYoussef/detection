
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
