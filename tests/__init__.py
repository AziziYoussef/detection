"""
Tests Package for Lost Objects Detection Service
Contains all test modules and test utilities
"""

import sys
from pathlib import Path

# Add project root to Python path for testing
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    "test_data_dir": "tests/fixtures",
    "test_models_dir": "tests/fixtures/models",
    "test_images_dir": "tests/fixtures/images",
    "test_videos_dir": "tests/fixtures/videos",
    "temp_test_dir": "tests/temp",
    "mock_external_services": True
}

# Test markers
TEST_MARKERS = {
    "integration": "Integration tests requiring full service setup",
    "performance": "Performance and load tests",
    "gpu": "Tests requiring GPU acceleration",
    "slow": "Slow running tests (> 10 seconds)",
    "external": "Tests requiring external services"
}

# Common test utilities
__all__ = [
    "TEST_CONFIG",
    "TEST_MARKERS"
]