"""
Cache and Model Management Utility
Handles model downloads, cache cleanup, and optimization
"""
import os
import shutil
import hashlib
import json
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import zipfile
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheManager:
    """Manages model cache and storage"""
    
    def __init__(self, 
                 models_dir: str = "storage/models",
                 cache_dir: str = "storage/cache", 
                 temp_dir: str = "storage/temp"):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories
        for dir_path in [self.models_dir, self.cache_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry_file = self.models_dir / "model_registry.json"
        self.model_registry = self.load_model_registry()
    
    def load_model_registry(self) -> Dict:
        """Load model registry from file"""
        if self.model_registry_file.exists():
            try:
                with open(self.model_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
        
        # Default registry
        return {
            "models": {},
            "last_updated": None,
            "version": "1.0"
        }
    
    def save_model_registry(self):
        """Save model registry to file"""
        try:
            with open(self.model_registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash"""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def scan_models(self) -> Dict:
        """Scan models directory and update registry"""
        logger.info("Scanning models directory...")
        
        model_files = list(self.models_dir.glob("*.pth"))
        scanned_models = {}
        
        for model_file in model_files:
            model_name = model_file.stem
            file_size = model_file.stat().st_size
            file_hash = self.calculate_file_hash(model_file)
            
            model_info = {
                "name": model_name,
                "file_path": str(model_file),
                "file_size": file_size,
                "file_hash": file_hash,
                "file_size_human": self.format_size(file_size),
                "last_modified": model_file.stat().st_mtime,
                "verified": True
            }
            
            scanned_models[model_name] = model_info
            logger.info(f"Found model: {model_name} ({self.format_size(file_size)})")
        
        # Update registry
        self.model_registry["models"] = scanned_models
        self.model_registry["last_updated"] = time.time()
        self.save_model_registry()
        
        logger.info(f"Scanned {len(scanned_models)} models")
        return scanned_models
    
    def verify_model(self, model_name: str) -> bool:
        """Verify model file integrity"""
        if model_name not in self.model_registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        model_info = self.model_registry["models"][model_name]
        model_path = Path(model_info["file_path"])
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Verify file hash
        current_hash = self.calculate_file_hash(model_path)
        expected_hash = model_info["file_hash"]
        
        if current_hash != expected_hash:
            logger.error(f"Hash mismatch for {model_name}")
            logger.error(f"Expected: {expected_hash}")
            logger.error(f"Current:  {current_hash}")
            return False
        
        logger.info(f"Model {model_name} verified successfully")
        return True
    
    def download_model(self, 
                      model_name: str, 
                      download_url: str, 
                      expected_hash: Optional[str] = None) -> bool:
        """Download model from URL"""
        logger.info(f"Downloading model: {model_name}")
        
        try:
            # Download to temporary file first
            temp_file = self.temp_dir / f"{model_name}_download.tmp"
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify hash if provided
            if expected_hash:
                file_hash = self.calculate_file_hash(temp_file)
                if file_hash != expected_hash:
                    logger.error(f"Downloaded file hash mismatch")
                    temp_file.unlink()
                    return False
            
            # Move to models directory
            final_path = self.models_dir / f"{model_name}.pth"
            shutil.move(temp_file, final_path)
            
            # Update registry
            file_size = final_path.stat().st_size
            file_hash = self.calculate_file_hash(final_path)
            
            self.model_registry["models"][model_name] = {
                "name": model_name,
                "file_path": str(final_path),
                "file_size": file_size,
                "file_hash": file_hash,
                "file_size_human": self.format_size(file_size),
                "last_modified": final_path.stat().st_mtime,
                "download_url": download_url,
                "verified": True
            }
            
            self.save_model_registry()
            
            logger.info(f"Model {model_name} downloaded successfully ({self.format_size(file_size)})")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
    
    def clean_cache(self, max_age_days: int = 7) -> int:
        """Clean old cache files"""
        import time
        
        logger.info(f"Cleaning cache files older than {max_age_days} days...")
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned_files = 0
        cleaned_size = 0
        
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file():
                if cache_file.stat().st_mtime < cutoff_time:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    cleaned_files += 1
                    cleaned_size += file_size
        
        logger.info(f"Cleaned {cleaned_files} files ({self.format_size(cleaned_size)})")
        return cleaned_files
    
    def clean_temp_files(self) -> int:
        """Clean temporary files"""
        logger.info("Cleaning temporary files...")
        
        cleaned_files = 0
        cleaned_size = 0
        
        for temp_file in self.temp_dir.rglob('*'):
            if temp_file.is_file():
                file_size = temp_file.stat().st_size
                temp_file.unlink()
                cleaned_files += 1
                cleaned_size += file_size
        
        logger.info(f"Cleaned {cleaned_files} temp files ({self.format_size(cleaned_size)})")
        return cleaned_files
    
    def backup_models(self, backup_path: str) -> bool:
        """Create backup of all models"""
        logger.info(f"Creating models backup: {backup_path}")
        
        try:
            backup_file = Path(backup_path)
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add model files
                for model_file in self.models_dir.glob("*.pth"):
                    zf.write(model_file, model_file.name)
                
                # Add registry
                if self.model_registry_file.exists():
                    zf.write(self.model_registry_file, self.model_registry_file.name)
            
            backup_size = backup_file.stat().st_size
            logger.info(f"Backup created: {self.format_size(backup_size)}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def restore_models(self, backup_path: str) -> bool:
        """Restore models from backup"""
        logger.info(f"Restoring models from: {backup_path}")
        
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                logger.error("Backup file not found")
                return False
            
            with zipfile.ZipFile(backup_file, 'r') as zf:
                zf.extractall(self.models_dir)
            
            # Rescan models
            self.scan_models()
            
            logger.info("Models restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def get_storage_info(self) -> Dict:
        """Get storage usage information"""
        models_size = self.get_directory_size(self.models_dir)
        cache_size = self.get_directory_size(self.cache_dir)
        temp_size = self.get_directory_size(self.temp_dir)
        total_size = models_size + cache_size + temp_size
        
        return {
            "models": {
                "size_bytes": models_size,
                "size_human": self.format_size(models_size),
                "file_count": len(list(self.models_dir.glob("*.pth")))
            },
            "cache": {
                "size_bytes": cache_size,
                "size_human": self.format_size(cache_size),
                "file_count": len(list(self.cache_dir.rglob('*')))
            },
            "temp": {
                "size_bytes": temp_size,
                "size_human": self.format_size(temp_size),
                "file_count": len(list(self.temp_dir.rglob('*')))
            },
            "total": {
                "size_bytes": total_size,
                "size_human": self.format_size(total_size)
            }
        }
    
    def optimize_models(self) -> bool:
        """Optimize model files (placeholder for future optimization)"""
        logger.info("Model optimization not yet implemented")
        # TODO: Implement model quantization, pruning, etc.
        return True
    
    def list_models(self) -> List[Dict]:
        """List all available models"""
        models = []
        
        for model_name, model_info in self.model_registry["models"].items():
            models.append({
                "name": model_name,
                "size": model_info["file_size_human"],
                "verified": model_info.get("verified", False),
                "path": model_info["file_path"]
            })
        
        return models

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Cache and Model Management Utility")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan models directory')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify model integrity')
    verify_parser.add_argument('--model', help='Specific model to verify')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download model')
    download_parser.add_argument('name', help='Model name')
    download_parser.add_argument('url', help='Download URL')
    download_parser.add_argument('--hash', help='Expected file hash')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cache and temp files')
    clean_parser.add_argument('--max-age', type=int, default=7, help='Max age in days for cache files')
    clean_parser.add_argument('--temp-only', action='store_true', help='Clean only temp files')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup models')
    backup_parser.add_argument('path', help='Backup file path')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore models from backup')
    restore_parser.add_argument('path', help='Backup file path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show storage information')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    try:
        if args.command == 'scan':
            cache_manager.scan_models()
        
        elif args.command == 'verify':
            if args.model:
                cache_manager.verify_model(args.model)
            else:
                # Verify all models
                for model_name in cache_manager.model_registry["models"]:
                    cache_manager.verify_model(model_name)
        
        elif args.command == 'download':
            cache_manager.download_model(args.name, args.url, args.hash)
        
        elif args.command == 'clean':
            if args.temp_only:
                cache_manager.clean_temp_files()
            else:
                cache_manager.clean_cache(args.max_age)
                cache_manager.clean_temp_files()
        
        elif args.command == 'backup':
            cache_manager.backup_models(args.path)
        
        elif args.command == 'restore':
            cache_manager.restore_models(args.path)
        
        elif args.command == 'info':
            info = cache_manager.get_storage_info()
            print(json.dumps(info, indent=2))
        
        elif args.command == 'list':
            models = cache_manager.list_models()
            print(f"\n📦 Available Models ({len(models)}):")
            print("-" * 50)
            for model in models:
                status = "✅" if model["verified"] else "❌"
                print(f"{status} {model['name']} ({model['size']})")
            print()
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import time
    exit(main())