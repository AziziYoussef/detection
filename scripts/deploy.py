"""
Deployment Script for Lost Objects Detection Service
Handles service deployment, configuration, and health checks
"""
import os
import sys
import subprocess
import argparse
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployment of the Lost Objects Detection Service"""
    
    def __init__(self, config_path: str = "config/deployment.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.service_url = f"http://{self.config['host']}:{self.config['port']}"
        
    def load_config(self) -> Dict:
        """Load deployment configuration"""
        default_config = {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "max_requests": 1000,
            "max_requests_jitter": 50,
            "timeout_keep_alive": 5,
            "log_level": "info",
            "reload": False,
            "ssl_keyfile": None,
            "ssl_certfile": None,
            "environment": "production",
            "models_dir": "storage/models",
            "temp_dir": "storage/temp",
            "cache_dir": "storage/cache",
            "log_dir": "logs",
            "health_check_interval": 30,
            "metrics_enabled": True
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        logger.info("Validating deployment environment...")
        
        # Check Python version
        if sys.version_info < (3.8, 0):
            logger.error("Python 3.8 or higher is required")
            return False
        
        # Check required directories
        required_dirs = [
            self.config['models_dir'],
            self.config['temp_dir'],
            self.config['cache_dir'],
            self.config['log_dir']
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {dir_path}")
        
        # Check required models
        models_dir = Path(self.config['models_dir'])
        model_files = list(models_dir.glob("*.pth"))
        
        if not model_files:
            logger.warning("No model files found in models directory")
            logger.warning("Service will start but detection may not work")
        else:
            logger.info(f"Found {len(model_files)} model files")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU available: {gpu_count} device(s)")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    logger.info(f"GPU {i}: {gpu_name}")
            else:
                logger.info("GPU not available, using CPU")
        except ImportError:
            logger.warning("PyTorch not installed, cannot check GPU")
        
        return True
    
    def install_dependencies(self, requirements_file: str = "requirements.txt"):
        """Install Python dependencies"""
        logger.info("Installing dependencies...")
        
        if not os.path.exists(requirements_file):
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], check=True)
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def generate_startup_script(self) -> str:
        """Generate startup script for the service"""
        uvicorn_cmd = [
            "uvicorn",
            "app.main:app",
            f"--host={self.config['host']}",
            f"--port={self.config['port']}",
            f"--workers={self.config['workers']}",
            f"--log-level={self.config['log_level']}",
            f"--timeout-keep-alive={self.config['timeout_keep_alive']}"
        ]
        
        if self.config['max_requests']:
            uvicorn_cmd.append(f"--limit-max-requests={self.config['max_requests']}")
        
        if self.config['max_requests_jitter']:
            uvicorn_cmd.append(f"--limit-max-requests-jitter={self.config['max_requests_jitter']}")
        
        if self.config['reload']:
            uvicorn_cmd.append("--reload")
        
        if self.config['ssl_keyfile'] and self.config['ssl_certfile']:
            uvicorn_cmd.extend([
                f"--ssl-keyfile={self.config['ssl_keyfile']}",
                f"--ssl-certfile={self.config['ssl_certfile']}"
            ])
        
        return " ".join(uvicorn_cmd)
    
    def start_service(self, background: bool = False) -> bool:
        """Start the service"""
        logger.info("Starting Lost Objects Detection Service...")
        
        startup_cmd = self.generate_startup_script()
        logger.info(f"Startup command: {startup_cmd}")
        
        try:
            if background:
                # Start in background
                log_file = os.path.join(self.config['log_dir'], 'service.log')
                with open(log_file, 'a') as f:
                    process = subprocess.Popen(
                        startup_cmd.split(),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True
                    )
                
                # Save PID for later management
                pid_file = os.path.join(self.config['log_dir'], 'service.pid')
                with open(pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                logger.info(f"Service started in background with PID: {process.pid}")
                logger.info(f"Logs: {log_file}")
                
                # Wait a moment and check if still running
                time.sleep(3)
                if process.poll() is None:
                    logger.info("Service appears to be running")
                    return True
                else:
                    logger.error("Service failed to start")
                    return False
            else:
                # Start in foreground
                subprocess.run(startup_cmd.split())
                return True
                
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Stop the service"""
        logger.info("Stopping service...")
        
        pid_file = os.path.join(self.config['log_dir'], 'service.pid')
        
        if not os.path.exists(pid_file):
            logger.warning("PID file not found, service may not be running")
            return False
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Try to terminate gracefully
            os.kill(pid, 15)  # SIGTERM
            time.sleep(5)
            
            # Check if still running
            try:
                os.kill(pid, 0)  # Check if process exists
                logger.warning("Process still running, forcing termination")
                os.kill(pid, 9)  # SIGKILL
            except OSError:
                pass  # Process already terminated
            
            # Remove PID file
            os.remove(pid_file)
            logger.info("Service stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            return False
    
    def health_check(self, timeout: int = 10) -> bool:
        """Perform health check on the service"""
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check passed: {data.get('status', 'unknown')}")
                return True
            else:
                logger.error(f"Health check failed: HTTP {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def wait_for_service(self, max_wait: int = 60) -> bool:
        """Wait for service to become available"""
        logger.info(f"Waiting for service to become available (max {max_wait}s)...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.health_check(timeout=5):
                elapsed = time.time() - start_time
                logger.info(f"Service is available after {elapsed:.1f}s")
                return True
            
            time.sleep(2)
        
        logger.error(f"Service did not become available within {max_wait}s")
        return False
    
    def get_service_status(self) -> Dict:
        """Get detailed service status"""
        try:
            # Check if process is running
            pid_file = os.path.join(self.config['log_dir'], 'service.pid')
            process_running = False
            pid = None
            
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    os.kill(pid, 0)  # Check if process exists
                    process_running = True
                except (OSError, ValueError):
                    pass
            
            # Check service health
            service_healthy = self.health_check(timeout=5)
            
            # Get service stats if healthy
            stats = {}
            if service_healthy:
                try:
                    response = requests.get(f"{self.service_url}/stats", timeout=5)
                    if response.status_code == 200:
                        stats = response.json()
                except requests.RequestException:
                    pass
            
            return {
                'process_running': process_running,
                'pid': pid,
                'service_healthy': service_healthy,
                'service_url': self.service_url,
                'config': self.config,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {'error': str(e)}
    
    def deploy(self, install_deps: bool = True, background: bool = True) -> bool:
        """Full deployment process"""
        logger.info("Starting deployment process...")
        
        # Validate environment
        if not self.validate_environment():
            logger.error("Environment validation failed")
            return False
        
        # Install dependencies
        if install_deps:
            if not self.install_dependencies():
                logger.error("Dependency installation failed")
                return False
        
        # Start service
        if not self.start_service(background=background):
            logger.error("Service startup failed")
            return False
        
        # Wait for service and verify
        if background:
            if not self.wait_for_service():
                logger.error("Service health check failed")
                return False
        
        logger.info("Deployment completed successfully!")
        return True

def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(
        description="Lost Objects Detection Service Deployment"
    )
    
    parser.add_argument(
        'action',
        choices=['deploy', 'start', 'stop', 'restart', 'status', 'health'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--config',
        default='config/deployment.json',
        help='Path to deployment configuration file'
    )
    
    parser.add_argument(
        '--no-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--foreground',
        action='store_true',
        help='Run service in foreground'
    )
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployer = DeploymentManager(args.config)
    
    try:
        if args.action == 'deploy':
            success = deployer.deploy(
                install_deps=not args.no_deps,
                background=not args.foreground
            )
            sys.exit(0 if success else 1)
        
        elif args.action == 'start':
            success = deployer.start_service(background=not args.foreground)
            if success and not args.foreground:
                deployer.wait_for_service()
            sys.exit(0 if success else 1)
        
        elif args.action == 'stop':
            success = deployer.stop_service()
            sys.exit(0 if success else 1)
        
        elif args.action == 'restart':
            logger.info("Restarting service...")
            deployer.stop_service()
            time.sleep(2)
            success = deployer.start_service(background=not args.foreground)
            if success and not args.foreground:
                deployer.wait_for_service()
            sys.exit(0 if success else 1)
        
        elif args.action == 'status':
            status = deployer.get_service_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.action == 'health':
            healthy = deployer.health_check()
            print(f"Service health: {'OK' if healthy else 'FAILED'}")
            sys.exit(0 if healthy else 1)
    
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()