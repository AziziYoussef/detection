"""
API Response Schemas for Lost Objects Detection Service
Defines Pydantic models for structured API responses
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

class ObjectStatus(str, Enum):
    """Object status enumeration"""
    NORMAL = "normal"
    SUSPECT = "suspect"
    LOST = "lost"
    RESOLVED = "resolved"

class AlertLevel(str, Enum):
    """Alert level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @validator('x2')
    def x2_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def y2_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v

class Detection(BaseModel):
    """Single object detection"""
    class_name: str = Field(..., alias="class", description="Object class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    class_id: Optional[int] = Field(None, description="Class ID number")
    
    class Config:
        allow_population_by_field_name = True

class LostObject(BaseModel):
    """Lost object with temporal information"""
    object_id: str = Field(..., description="Unique object identifier")
    class_name: str = Field(..., alias="class", description="Object class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    status: ObjectStatus = Field(..., description="Object status")
    first_seen: float = Field(..., description="Timestamp when first detected")
    last_seen: float = Field(..., description="Timestamp when last seen")
    duration_stationary: float = Field(..., description="Time stationary in seconds")
    nearest_person_distance: float = Field(..., description="Distance to nearest person")
    location: str = Field("", description="Location identifier")
    alert_level: AlertLevel = Field(AlertLevel.LOW, description="Alert severity level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        allow_population_by_field_name = True

class ProcessingStats(BaseModel):
    """Processing performance statistics"""
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: Optional[str] = Field(None, description="Model used for detection")
    image_size: Optional[tuple] = Field(None, description="Input image dimensions")
    total_detections: int = Field(..., description="Total objects detected")
    device: Optional[str] = Field(None, description="Processing device (cpu/gpu)")

# === IMAGE DETECTION RESPONSES ===

class ImageDetectionResponse(BaseModel):
    """Response for single image detection"""
    success: bool = Field(..., description="Whether detection was successful")
    timestamp: float = Field(..., description="Response timestamp")
    detections: List[Detection] = Field(..., description="Detected objects")
    stats: ProcessingStats = Field(..., description="Processing statistics")
    error: Optional[str] = Field(None, description="Error message if failed")

class ImageLostObjectsResponse(BaseModel):
    """Response for image lost objects detection"""
    success: bool = Field(..., description="Whether detection was successful")
    timestamp: float = Field(..., description="Response timestamp")
    location: str = Field("", description="Location identifier")
    all_detections: List[Detection] = Field(..., description="All detected objects")
    lost_objects: List[LostObject] = Field(..., description="Lost objects")
    suspect_objects: List[LostObject] = Field(..., description="Suspect objects")
    tracked_objects_count: int = Field(..., description="Total tracked objects")
    stats: ProcessingStats = Field(..., description="Processing statistics")
    error: Optional[str] = Field(None, description="Error message if failed")

# === VIDEO DETECTION RESPONSES ===

class VideoJobInfo(BaseModel):
    """Video processing job information"""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Video filename")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Completion progress")
    current_frame: int = Field(..., description="Current frame being processed")
    total_frames: int = Field(..., description="Total frames in video")
    start_time: float = Field(..., description="Job start timestamp")
    processing_duration: Optional[float] = Field(None, description="Processing duration")
    error: Optional[str] = Field(None, description="Error message if failed")

class VideoDetectionResponse(BaseModel):
    """Response for video detection job creation"""
    success: bool = Field(..., description="Whether job was created successfully")
    job_id: str = Field(..., description="Job identifier for tracking")
    message: str = Field(..., description="Response message")
    estimated_processing_time: float = Field(..., description="Estimated processing time")
    timestamp: float = Field(..., description="Response timestamp")

class VideoStatusResponse(BaseModel):
    """Response for video job status"""
    job_info: VideoJobInfo = Field(..., description="Job information")
    results_available: bool = Field(..., description="Whether results are ready")
    estimated_remaining_time: Optional[float] = Field(None, description="Estimated remaining time")

class FrameDetection(BaseModel):
    """Detection results for a video frame"""
    frame_number: int = Field(..., description="Frame number")
    timestamp: float = Field(..., description="Frame timestamp in video")
    detections: List[Detection] = Field(..., description="Detected objects")
    lost_objects: List[LostObject] = Field(..., description="Lost objects in frame")
    suspect_objects: List[LostObject] = Field(..., description="Suspect objects in frame")
    processing_time: float = Field(..., description="Frame processing time")

class VideoResultsSummary(BaseModel):
    """Summary of video processing results"""
    total_frames_processed: int = Field(..., description="Total frames processed")
    total_objects_detected: int = Field(..., description="Total objects detected")
    total_lost_objects: int = Field(..., description="Total lost objects found")
    total_suspect_objects: int = Field(..., description="Total suspect objects found")
    frames_with_lost_objects: List[int] = Field(..., description="Frame numbers with lost objects")
    avg_processing_time_per_frame: float = Field(..., description="Average processing time per frame")
    class_distribution: Dict[str, int] = Field(..., description="Distribution of detected classes")

class VideoResultsResponse(BaseModel):
    """Response for video processing results"""
    job_id: str = Field(..., description="Job identifier")
    video_path: str = Field(..., description="Original video path")
    processing_timestamp: float = Field(..., description="When processing completed")
    frame_detections: List[FrameDetection] = Field(..., description="Per-frame detection results")
    summary: VideoResultsSummary = Field(..., description="Results summary")
    settings_used: Dict[str, Any] = Field(..., description="Processing settings used")

# === BATCH DETECTION RESPONSES ===

class BatchJobInfo(BaseModel):
    """Batch processing job information"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Completion progress")
    processed_images: int = Field(..., description="Number of processed images")
    total_images: int = Field(..., description="Total images to process")
    elapsed_time: float = Field(..., description="Elapsed processing time")
    estimated_remaining_time: Optional[float] = Field(None, description="Estimated remaining time")
    error: Optional[str] = Field(None, description="Error message if failed")

class BatchDetectionResponse(BaseModel):
    """Response for batch detection job creation"""
    success: bool = Field(..., description="Whether job was created successfully")
    job_id: str = Field(..., description="Job identifier for tracking")
    message: str = Field(..., description="Response message")
    estimated_processing_time: float = Field(..., description="Estimated processing time")
    timestamp: float = Field(..., description="Response timestamp")

class BatchImageResult(BaseModel):
    """Detection result for a single image in batch"""
    image_index: int = Field(..., description="Image index in batch")
    filename: str = Field(..., description="Image filename")
    success: bool = Field(..., description="Whether detection succeeded")
    detections: List[Detection] = Field(..., description="Detected objects")
    processing_time: float = Field(..., description="Processing time for this image")
    error: Optional[str] = Field(None, description="Error message if failed")

class BatchResultsResponse(BaseModel):
    """Response for batch processing results"""
    job_id: str = Field(..., description="Job identifier")
    total_files: int = Field(..., description="Total files processed")
    successful_detections: int = Field(..., description="Number of successful detections")
    failed_detections: int = Field(..., description="Number of failed detections")
    results: List[BatchImageResult] = Field(..., description="Per-image results")
    processing_time: float = Field(..., description="Total processing time")
    settings_used: Dict[str, Any] = Field(..., description="Processing settings used")

# === STREAMING RESPONSES ===

class StreamClientInfo(BaseModel):
    """Streaming client information"""
    client_id: str = Field(..., description="Client identifier")
    connected_at: float = Field(..., description="Connection timestamp")
    last_ping: float = Field(..., description="Last ping timestamp")
    connection_duration: float = Field(..., description="Connection duration in seconds")
    active: bool = Field(..., description="Whether client is active")
    settings: Dict[str, Any] = Field(..., description="Client settings")
    stats: Dict[str, Any] = Field(..., description="Client statistics")

class StreamDetectionResult(BaseModel):
    """Real-time stream detection result"""
    frame_id: str = Field(..., description="Frame identifier")
    timestamp: float = Field(..., description="Processing timestamp")
    processing_time: float = Field(..., description="Processing time")
    detections: Dict[str, int] = Field(..., description="Detection counts")
    detailed_results: Optional[Dict[str, List[Union[Detection, LostObject]]]] = Field(
        None, description="Detailed detection results"
    )
    annotations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Annotation data for visualization"
    )

class StreamStatsResponse(BaseModel):
    """Streaming service statistics"""
    service_stats: Dict[str, Any] = Field(..., description="Service-level statistics")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics")
    clients: Dict[str, Dict[str, Any]] = Field(..., description="Per-client statistics")
    timestamp: float = Field(..., description="Statistics timestamp")

# === MODEL MANAGEMENT RESPONSES ===

class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    description: str = Field(..., description="Model description")
    version: str = Field(..., description="Model version")
    loaded: bool = Field(..., description="Whether model is currently loaded")
    memory_usage_mb: int = Field(..., description="Memory usage in MB")
    load_time: float = Field(..., description="Load time in seconds")
    last_used: float = Field(..., description="Last used timestamp")

class ModelsResponse(BaseModel):
    """Response for available models"""
    available_models: Dict[str, ModelInfo] = Field(..., description="Available models")
    timestamp: float = Field(..., description="Response timestamp")

class ModelPerformanceResponse(BaseModel):
    """Response for model performance statistics"""
    model_name: str = Field(..., description="Model name")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    performance: Dict[str, Any] = Field(..., description="Performance statistics")
    load_time: float = Field(..., description="Model load time")
    memory_usage_mb: int = Field(..., description="Memory usage in MB")
    last_used: float = Field(..., description="Last used timestamp")

# === SERVICE STATISTICS RESPONSES ===

class SystemInfo(BaseModel):
    """System information"""
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    available_memory_gb: float = Field(..., description="Available memory in GB")
    gpu_available: Optional[bool] = Field(None, description="Whether GPU is available")
    gpu_memory_allocated: Optional[float] = Field(None, description="GPU memory allocated in MB")
    gpu_memory_reserved: Optional[float] = Field(None, description="GPU memory reserved in MB")

class ServiceStatsResponse(BaseModel):
    """Service statistics response"""
    service_info: Dict[str, Any] = Field(..., description="Service information")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    cache: Dict[str, Any] = Field(..., description="Cache statistics")
    system: SystemInfo = Field(..., description="System information")
    models: Dict[str, int] = Field(..., description="Model statistics")
    timestamp: float = Field(..., description="Statistics timestamp")

# === HEALTH CHECK RESPONSES ===

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    message: str = Field(..., description="Health status message")
    timestamp: float = Field(..., description="Health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")

# === ERROR RESPONSES ===

class ErrorDetail(BaseModel):
    """Error detail information"""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(False, description="Success flag (always False for errors)")
    error: ErrorDetail = Field(..., description="Error information")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracing")

# === PAGINATION RESPONSES ===

class PaginationInfo(BaseModel):
    """Pagination information"""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")

class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    items: List[Any] = Field(..., description="Items for current page")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    timestamp: float = Field(..., description="Response timestamp")

# === UTILITY FUNCTIONS ===

def create_success_response(
    data: Any, 
    message: str = "Operation completed successfully",
    timestamp: Optional[float] = None
) -> Dict[str, Any]:
    """Create a standardized success response"""
    import time
    
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": timestamp or time.time()
    }

def create_error_response(
    error_type: str,
    message: str,
    code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None
) -> ErrorResponse:
    """Create a standardized error response"""
    import time
    
    return ErrorResponse(
        error=ErrorDetail(
            type=error_type,
            message=message,
            code=code,
            details=details
        ),
        timestamp=timestamp or time.time()
    )

def paginate_items(
    items: List[Any],
    page: int,
    page_size: int
) -> PaginatedResponse:
    """Create paginated response from items list"""
    import time
    import math
    
    total_items = len(items)
    total_pages = math.ceil(total_items / page_size) if total_items > 0 else 0
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = items[start_idx:end_idx]
    
    pagination = PaginationInfo(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )
    
    return PaginatedResponse(
        items=page_items,
        pagination=pagination,
        timestamp=time.time()
    )

# === VALIDATION UTILITIES ===

def validate_bbox_coordinates(bbox: BoundingBox, image_width: int, image_height: int) -> bool:
    """Validate bounding box coordinates against image dimensions"""
    return (
        0 <= bbox.x1 < bbox.x2 <= image_width and
        0 <= bbox.y1 < bbox.y2 <= image_height
    )

def normalize_detection_confidence(detection: Detection) -> Detection:
    """Ensure detection confidence is within valid range"""
    detection.confidence = max(0.0, min(1.0, detection.confidence))
    return detection

# Export commonly used response types
__all__ = [
    'ObjectStatus', 'AlertLevel', 'BoundingBox', 'Detection', 'LostObject',
    'ImageDetectionResponse', 'ImageLostObjectsResponse',
    'VideoDetectionResponse', 'VideoStatusResponse', 'VideoResultsResponse',
    'BatchDetectionResponse', 'BatchResultsResponse',
    'StreamDetectionResult', 'StreamStatsResponse',
    'ModelsResponse', 'ModelPerformanceResponse',
    'ServiceStatsResponse', 'HealthCheckResponse',
    'ErrorResponse', 'PaginatedResponse',
    'create_success_response', 'create_error_response', 'paginate_items'
]