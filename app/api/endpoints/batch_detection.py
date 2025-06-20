"""
Batch Processing Detection API Endpoint
Handles bulk image processing with optimized GPU utilization
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import time
import json
import zipfile
import tempfile
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

from ...services.model_service import ModelService
from ...schemas.detection import BatchDetectionRequest, BatchDetectionResponse
from ...utils.image_utils import preprocess_image, validate_image
from ...core.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch-processing"])

# Job storage (in production, use Redis or database)
batch_jobs: Dict[str, Dict] = {}

# Dependency to get model service
async def get_model_service() -> ModelService:
    """Get model service instance"""
    # This would be injected from the main app state
    from fastapi import Request
    request = Request.scope
    return request.app.state.model_service

class BatchProcessor:
    """Handles batch processing operations"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.max_batch_size = 32
        self.max_concurrent_jobs = 5
        self.active_jobs = 0
        self.temp_dir = Path(tempfile.gettempdir()) / "batch_detection"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def process_image_batch(
        self,
        images: List[np.ndarray],
        model_name: Optional[str] = None,
        batch_size: int = 8,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """Process a batch of images efficiently"""
        results = []
        total_images = len(images)
        
        # Get model
        detector = await self.model_service.get_model(model_name)
        
        # Process in smaller batches for GPU memory management
        for i in range(0, total_images, batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = []
            
            # Process each image in the batch
            for j, image in enumerate(batch_images):
                try:
                    result = detector.detect_objects(image)
                    batch_results.append({
                        'image_index': i + j,
                        'success': True,
                        'detections': result.get('detections', []),
                        'processing_time': result.get('processing_time', 0),
                        'error': None
                    })
                except Exception as e:
                    batch_results.append({
                        'image_index': i + j,
                        'success': False,
                        'detections': [],
                        'processing_time': 0,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
            
            # Update progress
            if progress_callback:
                progress = (i + len(batch_images)) / total_images
                await progress_callback(progress)
            
            # Allow other tasks to run
            await asyncio.sleep(0.01)
        
        return results
    
    async def process_files_batch(
        self,
        files: List[UploadFile],
        job_id: str,
        settings: Dict
    ) -> Dict:
        """Process uploaded files in batch"""
        try:
            # Load images
            images = []
            file_info = []
            
            for i, file in enumerate(files):
                try:
                    # Read file
                    content = await file.read()
                    
                    # Validate and load image
                    image = self._load_image_from_bytes(content)
                    
                    images.append(image)
                    file_info.append({
                        'filename': file.filename,
                        'size': len(content),
                        'index': i
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to load image {file.filename}: {e}")
                    file_info.append({
                        'filename': file.filename,
                        'size': 0,
                        'index': i,
                        'error': str(e)
                    })
            
            # Update job status
            batch_jobs[job_id]['status'] = 'processing'
            batch_jobs[job_id]['total_images'] = len(images)
            batch_jobs[job_id]['processed_images'] = 0
            
            # Progress callback
            async def update_progress(progress: float):
                batch_jobs[job_id]['progress'] = progress
                batch_jobs[job_id]['processed_images'] = int(progress * len(images))
            
            # Process images
            results = await self.process_image_batch(
                images=images,
                model_name=settings.get('model_name'),
                batch_size=settings.get('batch_size', 8),
                progress_callback=update_progress
            )
            
            # Compile final results
            final_results = {
                'job_id': job_id,
                'total_files': len(files),
                'successful_detections': sum(1 for r in results if r['success']),
                'failed_detections': sum(1 for r in results if not r['success']),
                'file_info': file_info,
                'detections': results,
                'processing_time': time.time() - batch_jobs[job_id]['start_time'],
                'settings_used': settings
            }
            
            # Save results
            results_path = await self._save_results(job_id, final_results)
            
            # Update job
            batch_jobs[job_id].update({
                'status': 'completed',
                'progress': 1.0,
                'results': final_results,
                'results_path': results_path,
                'end_time': time.time()
            })
            
            return final_results
            
        except Exception as e:
            # Update job with error
            batch_jobs[job_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': time.time()
            })
            raise
    
    def _load_image_from_bytes(self, content: bytes) -> np.ndarray:
        """Load image from bytes"""
        # Try with OpenCV first
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            # Try with PIL
            pil_image = Image.open(io.BytesIO(content))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        if image is None:
            raise ValueError("Unable to decode image")
        
        return image
    
    async def _save_results(self, job_id: str, results: Dict) -> str:
        """Save results to file"""
        results_dir = self.temp_dir / job_id
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        json_path = results_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create CSV summary
        csv_path = results_dir / "summary.csv"
        self._create_csv_summary(results, csv_path)
        
        # Create ZIP with all results
        zip_path = results_dir / "batch_results.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(json_path, "results.json")
            zf.write(csv_path, "summary.csv")
        
        return str(zip_path)
    
    def _create_csv_summary(self, results: Dict, csv_path: Path):
        """Create CSV summary of results"""
        import pandas as pd
        
        rows = []
        for detection in results['detections']:
            image_index = detection['image_index']
            file_info = results['file_info'][image_index]
            
            if detection['success']:
                for obj in detection['detections']:
                    rows.append({
                        'image_index': image_index,
                        'filename': file_info['filename'],
                        'object_class': obj['class'],
                        'confidence': obj['confidence'],
                        'bbox_x1': obj['bbox'][0],
                        'bbox_y1': obj['bbox'][1],
                        'bbox_x2': obj['bbox'][2],
                        'bbox_y2': obj['bbox'][3]
                    })
            else:
                rows.append({
                    'image_index': image_index,
                    'filename': file_info['filename'],
                    'object_class': 'ERROR',
                    'confidence': 0,
                    'bbox_x1': 0,
                    'bbox_y1': 0,
                    'bbox_x2': 0,
                    'bbox_y2': 0
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

# Initialize processor
processor = None

@router.on_event("startup")
async def startup_batch_processor():
    global processor
    # processor = BatchProcessor(get_model_service())
    pass

@router.post("/upload")
async def upload_batch_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Query(None, description="Model to use for detection"),
    batch_size: int = Query(8, description="Batch size for processing"),
    confidence_threshold: float = Query(0.3, description="Confidence threshold"),
    max_detections: int = Query(50, description="Maximum detections per image"),
    model_service: ModelService = Depends(get_model_service)
) -> JSONResponse:
    """
    Upload and process multiple images in batch
    
    Args:
        files: List of image files to process
        model_name: Model to use for detection
        batch_size: Processing batch size
        confidence_threshold: Detection confidence threshold
        max_detections: Maximum detections per image
        
    Returns:
        Job information for tracking progress
    """
    # Validate input
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if len(files) > 1000:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max 1000)")
    
    # Validate file types
    allowed_types = {'image/jpeg', 'image/png', 'image/webp', 'image/bmp'}
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
    
    # Create job
    job_id = str(uuid.uuid4())
    
    settings = {
        'model_name': model_name,
        'batch_size': batch_size,
        'confidence_threshold': confidence_threshold,
        'max_detections': max_detections
    }
    
    job_info = {
        'job_id': job_id,
        'status': 'queued',
        'progress': 0.0,
        'total_images': len(files),
        'processed_images': 0,
        'start_time': time.time(),
        'settings': settings,
        'file_count': len(files)
    }
    
    batch_jobs[job_id] = job_info
    
    # Initialize processor if needed
    global processor
    if processor is None:
        processor = BatchProcessor(model_service)
    
    # Start background processing
    background_tasks.add_task(
        processor.process_files_batch,
        files, job_id, settings
    )
    
    return JSONResponse({
        'job_id': job_id,
        'status': 'queued',
        'message': f'Batch processing started for {len(files)} images',
        'estimated_processing_time': len(files) * 0.5,  # Rough estimate
        'timestamp': time.time()
    })

@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str) -> JSONResponse:
    """Get batch processing job status"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    # Calculate additional metrics
    elapsed_time = time.time() - job['start_time']
    
    if job['progress'] > 0 and job['status'] == 'processing':
        estimated_total_time = elapsed_time / job['progress']
        remaining_time = estimated_total_time - elapsed_time
    else:
        remaining_time = None
    
    return JSONResponse({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'processed_images': job.get('processed_images', 0),
        'total_images': job['total_images'],
        'elapsed_time': elapsed_time,
        'estimated_remaining_time': remaining_time,
        'error': job.get('error'),
        'results_available': 'results_path' in job,
        'timestamp': time.time()
    })

@router.get("/job/{job_id}/results")
async def get_job_results(job_id: str) -> JSONResponse:
    """Get batch processing results"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if 'results' not in job:
        raise HTTPException(status_code=500, detail="Results not found")
    
    return JSONResponse({
        'job_id': job_id,
        'results': job['results'],
        'timestamp': time.time()
    })

@router.get("/job/{job_id}/download")
async def download_job_results(job_id: str) -> FileResponse:
    """Download batch processing results as ZIP file"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if 'results_path' not in job:
        raise HTTPException(status_code=500, detail="Results file not found")
    
    results_path = job['results_path']
    if not Path(results_path).exists():
        raise HTTPException(status_code=500, detail="Results file missing")
    
    return FileResponse(
        results_path,
        filename=f"batch_results_{job_id}.zip",
        media_type="application/zip"
    )

@router.delete("/job/{job_id}")
async def cancel_job(job_id: str) -> JSONResponse:
    """Cancel or delete batch processing job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    # Cancel if still running
    if job['status'] in ['queued', 'processing']:
        job['status'] = 'cancelled'
        job['end_time'] = time.time()
    
    # Clean up files
    try:
        if 'results_path' in job:
            results_dir = Path(job['results_path']).parent
            if results_dir.exists():
                import shutil
                shutil.rmtree(results_dir)
    except Exception as e:
        logger.error(f"Error cleaning up job files: {e}")
    
    # Remove from memory
    del batch_jobs[job_id]
    
    return JSONResponse({
        'message': f'Job {job_id} cancelled and cleaned up',
        'timestamp': time.time()
    })

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of jobs to return")
) -> JSONResponse:
    """List batch processing jobs"""
    jobs_list = []
    
    for job_id, job in batch_jobs.items():
        if status and job['status'] != status:
            continue
        
        jobs_list.append({
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'total_images': job['total_images'],
            'start_time': job['start_time'],
            'elapsed_time': time.time() - job['start_time']
        })
    
    # Sort by start time (newest first)
    jobs_list.sort(key=lambda x: x['start_time'], reverse=True)
    
    # Apply limit
    jobs_list = jobs_list[:limit]
    
    return JSONResponse({
        'jobs': jobs_list,
        'total_jobs': len(batch_jobs),
        'timestamp': time.time()
    })

@router.post("/cleanup")
async def cleanup_old_jobs(
    older_than_hours: int = Query(24, description="Clean up jobs older than X hours")
) -> JSONResponse:
    """Clean up old completed jobs"""
    cutoff_time = time.time() - (older_than_hours * 3600)
    cleaned_jobs = []
    
    for job_id, job in list(batch_jobs.items()):
        if (job['status'] in ['completed', 'failed', 'cancelled'] and 
            job['start_time'] < cutoff_time):
            
            # Clean up files
            try:
                if 'results_path' in job:
                    results_dir = Path(job['results_path']).parent
                    if results_dir.exists():
                        import shutil
                        shutil.rmtree(results_dir)
            except Exception as e:
                logger.error(f"Error cleaning up job {job_id}: {e}")
            
            del batch_jobs[job_id]
            cleaned_jobs.append(job_id)
    
    return JSONResponse({
        'cleaned_jobs': len(cleaned_jobs),
        'job_ids': cleaned_jobs,
        'cutoff_hours': older_than_hours,
        'timestamp': time.time()
    })

@router.get("/stats")
async def get_batch_stats() -> JSONResponse:
    """Get batch processing statistics"""
    status_counts = {}
    total_images = 0
    total_processing_time = 0
    
    for job in batch_jobs.values():
        status = job['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        total_images += job['total_images']
        
        if 'end_time' in job:
            total_processing_time += job['end_time'] - job['start_time']
    
    return JSONResponse({
        'total_jobs': len(batch_jobs),
        'status_distribution': status_counts,
        'total_images_processed': total_images,
        'total_processing_time_hours': total_processing_time / 3600,
        'average_images_per_job': total_images / len(batch_jobs) if batch_jobs else 0,
        'timestamp': time.time()
    })