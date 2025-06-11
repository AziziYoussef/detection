"""
Comprehensive Testing Script for Lost Objects Detection Service
Tests all API endpoints and functionality
"""
import asyncio
import aiohttp
import websockets
import json
import time
import numpy as np
import cv2
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceTester:
    """Comprehensive service testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace('http', 'ws')
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        self.results['total_tests'] += 1
        
        if success:
            self.results['passed_tests'] += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.results['failed_tests'] += 1
            logger.error(f"❌ {test_name}: FAILED - {details}")
        
        self.results['test_details'].append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
    
    def create_test_image(self, width: int = 640, height: int = 480) -> bytes:
        """Create a test image"""
        # Create a simple test image with some shapes
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some colored rectangles (simulating objects)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
        cv2.rectangle(image, (200, 100), (300, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(image, (400, 50), (500, 150), (0, 0, 255), -1)  # Red rectangle
        
        # Add some text
        cv2.putText(image, "Test Image", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    async def test_health_endpoint(self) -> bool:
        """Test health check endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.log_test_result("Health Check", True)
                        return True
                    else:
                        self.log_test_result("Health Check", False, f"HTTP {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Health Check", False, str(e))
            return False
    
    async def test_models_endpoint(self) -> bool:
        """Test models listing endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('available_models', {})
                        self.log_test_result("Models Endpoint", True, f"Found {len(models)} models")
                        return True
                    else:
                        self.log_test_result("Models Endpoint", False, f"HTTP {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Models Endpoint", False, str(e))
            return False
    
    async def test_image_detection(self) -> bool:
        """Test image detection endpoint"""
        try:
            test_image = self.create_test_image()
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('file', io.BytesIO(test_image), filename='test.jpg', content_type='image/jpeg')
                
                async with session.post(f"{self.base_url}/api/v1/detect/image", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        detections = result.get('detections', [])
                        self.log_test_result("Image Detection", True, f"Found {len(detections)} detections")
                        return True
                    else:
                        self.log_test_result("Image Detection", False, f"HTTP {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Image Detection", False, str(e))
            return False
    
    async def test_batch_detection(self) -> bool:
        """Test batch detection endpoint"""
        try:
            # Create multiple test images
            test_images = [self.create_test_image() for _ in range(3)]
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                
                for i, image_data in enumerate(test_images):
                    data.add_field('files', io.BytesIO(image_data), 
                                 filename=f'test_{i}.jpg', content_type='image/jpeg')
                
                # Start batch job
                async with session.post(f"{self.base_url}/api/v1/detect/batch/upload", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        job_id = result.get('job_id')
                        
                        if job_id:
                            # Wait for job completion
                            for _ in range(30):  # Wait up to 30 seconds
                                await asyncio.sleep(1)
                                
                                async with session.get(f"{self.base_url}/api/v1/detect/batch/job/{job_id}/status") as status_response:
                                    if status_response.status == 200:
                                        status_data = await status_response.json()
                                        if status_data.get('status') == 'completed':
                                            self.log_test_result("Batch Detection", True, f"Job {job_id} completed")
                                            return True
                                        elif status_data.get('status') == 'failed':
                                            self.log_test_result("Batch Detection", False, f"Job {job_id} failed")
                                            return False
                            
                            self.log_test_result("Batch Detection", False, "Job timeout")
                            return False
                        else:
                            self.log_test_result("Batch Detection", False, "No job ID returned")
                            return False
                    else:
                        self.log_test_result("Batch Detection", False, f"HTTP {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Batch Detection", False, str(e))
            return False
    
    async def test_websocket_streaming(self) -> bool:
        """Test WebSocket streaming endpoint"""
        try:
            client_id = f"test_client_{int(time.time())}"
            ws_uri = f"{self.ws_url}/ws/stream/{client_id}"
            
            async with websockets.connect(ws_uri) as websocket:
                # Wait for welcome message
                welcome_msg = await websocket.recv()
                welcome_data = json.loads(welcome_msg)
                
                if welcome_data.get('type') == 'welcome':
                    # Send a ping
                    ping_msg = {
                        'type': 'ping',
                        'timestamp': time.time()
                    }
                    await websocket.send(json.dumps(ping_msg))
                    
                    # Wait for pong
                    pong_msg = await websocket.recv()
                    pong_data = json.loads(pong_msg)
                    
                    if pong_data.get('type') == 'pong':
                        # Send test frame
                        test_image = self.create_test_image()
                        await websocket.send(test_image)
                        
                        # Wait for frame acknowledgment
                        response = await websocket.recv()
                        response_data = json.loads(response)
                        
                        if response_data.get('type') == 'frame_received':
                            self.log_test_result("WebSocket Streaming", True, "Frame processed successfully")
                            return True
                        else:
                            self.log_test_result("WebSocket Streaming", False, "No frame acknowledgment")
                            return False
                    else:
                        self.log_test_result("WebSocket Streaming", False, "No pong response")
                        return False
                else:
                    self.log_test_result("WebSocket Streaming", False, "No welcome message")
                    return False
                    
        except Exception as e:
            self.log_test_result("WebSocket Streaming", False, str(e))
            return False
    
    async def test_stats_endpoint(self) -> bool:
        """Test statistics endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/stats") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.log_test_result("Stats Endpoint", True)
                        return True
                    else:
                        self.log_test_result("Stats Endpoint", False, f"HTTP {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Stats Endpoint", False, str(e))
            return False
    
    async def test_streaming_endpoints(self) -> bool:
        """Test streaming-related REST endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test streaming stats
                async with session.get(f"{self.base_url}/api/v1/stream/stats") as response:
                    if response.status == 200:
                        self.log_test_result("Streaming Stats", True)
                        return True
                    else:
                        self.log_test_result("Streaming Stats", False, f"HTTP {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Streaming Stats", False, str(e))
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test with invalid image data
                data = aiohttp.FormData()
                data.add_field('file', io.BytesIO(b'invalid image data'), 
                             filename='invalid.jpg', content_type='image/jpeg')
                
                async with session.post(f"{self.base_url}/api/v1/detect/image", data=data) as response:
                    # Should return an error but not crash
                    if response.status in [400, 422, 500]:  # Expected error codes
                        self.log_test_result("Error Handling", True, "Service handled invalid input gracefully")
                        return True
                    else:
                        self.log_test_result("Error Handling", False, f"Unexpected status: {response.status}")
                        return False
        except Exception as e:
            self.log_test_result("Error Handling", False, str(e))
            return False
    
    async def test_concurrent_requests(self, num_requests: int = 5) -> bool:
        """Test concurrent request handling"""
        try:
            test_image = self.create_test_image()
            
            async def make_request(session, request_id):
                data = aiohttp.FormData()
                data.add_field('file', io.BytesIO(test_image), 
                             filename=f'test_{request_id}.jpg', content_type='image/jpeg')
                
                async with session.post(f"{self.base_url}/api/v1/detect/image", data=data) as response:
                    return response.status == 200
            
            async with aiohttp.ClientSession() as session:
                tasks = [make_request(session, i) for i in range(num_requests)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful_requests = sum(1 for r in results if r is True)
                
                if successful_requests >= num_requests * 0.8:  # Allow 20% failure rate
                    self.log_test_result("Concurrent Requests", True, 
                                       f"{successful_requests}/{num_requests} successful")
                    return True
                else:
                    self.log_test_result("Concurrent Requests", False, 
                                       f"Only {successful_requests}/{num_requests} successful")
                    return False
        except Exception as e:
            self.log_test_result("Concurrent Requests", False, str(e))
            return False
    
    async def test_performance_benchmark(self) -> bool:
        """Benchmark detection performance"""
        try:
            test_image = self.create_test_image()
            num_requests = 10
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                for _ in range(num_requests):
                    data = aiohttp.FormData()
                    data.add_field('file', io.BytesIO(test_image), 
                                 filename='benchmark.jpg', content_type='image/jpeg')
                    
                    async with session.post(f"{self.base_url}/api/v1/detect/image", data=data) as response:
                        if response.status != 200:
                            self.log_test_result("Performance Benchmark", False, "Request failed")
                            return False
            
            total_time = time.time() - start_time
            avg_time = total_time / num_requests
            rps = num_requests / total_time
            
            self.log_test_result("Performance Benchmark", True, 
                               f"Avg: {avg_time:.3f}s/request, RPS: {rps:.1f}")
            return True
            
        except Exception as e:
            self.log_test_result("Performance Benchmark", False, str(e))
            return False
    
    async def run_all_tests(self) -> Dict:
        """Run all tests"""
        logger.info("🚀 Starting comprehensive service tests...")
        
        # Basic functionality tests
        await self.test_health_endpoint()
        await self.test_models_endpoint()
        await self.test_stats_endpoint()
        
        # Core detection tests
        await self.test_image_detection()
        await self.test_batch_detection()
        
        # Streaming tests
        await self.test_websocket_streaming()
        await self.test_streaming_endpoints()
        
        # Reliability tests
        await self.test_error_handling()
        await self.test_concurrent_requests()
        
        # Performance tests
        await self.test_performance_benchmark()
        
        # Calculate final results
        success_rate = (self.results['passed_tests'] / self.results['total_tests']) * 100
        
        logger.info(f"\n📊 Test Results Summary:")
        logger.info(f"Total Tests: {self.results['total_tests']}")
        logger.info(f"Passed: {self.results['passed_tests']}")
        logger.info(f"Failed: {self.results['failed_tests']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("🎉 Service is working well!")
        else:
            logger.warning("⚠️ Service has significant issues")
        
        return self.results
    
    def generate_report(self, output_file: str = "test_report.json"):
        """Generate detailed test report"""
        report = {
            'test_summary': {
                'total_tests': self.results['total_tests'],
                'passed_tests': self.results['passed_tests'],
                'failed_tests': self.results['failed_tests'],
                'success_rate': (self.results['passed_tests'] / self.results['total_tests']) * 100,
                'test_timestamp': time.time()
            },
            'test_details': self.results['test_details'],
            'service_info': {
                'base_url': self.base_url,
                'websocket_url': self.ws_url
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {output_file}")

async def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Lost Objects Detection Service")
    
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='Base URL of the service'
    )
    
    parser.add_argument(
        '--output',
        default='test_report.json',
        help='Output file for test report'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only basic tests (skip performance and stress tests)'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ServiceTester(args.url)
    
    try:
        if args.quick:
            # Quick tests only
            await tester.test_health_endpoint()
            await tester.test_models_endpoint()
            await tester.test_image_detection()
        else:
            # Full test suite
            await tester.run_all_tests()
        
        # Generate report
        tester.generate_report(args.output)
        
        # Exit with appropriate code
        success_rate = (tester.results['passed_tests'] / tester.results['total_tests']) * 100
        exit_code = 0 if success_rate >= 80 else 1
        exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        exit(1)

if __name__ == '__main__':
    asyncio.run(main())