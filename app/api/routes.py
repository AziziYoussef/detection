# app/api/routes.py
from fastapi import APIRouter
from app.api.endpoints import image_detection, video_detection, stream_detection, models

router = APIRouter()

# Inclusion des endpoints
router.include_router(
    image_detection.router,
    prefix="/detect",
    tags=["Image Detection"]
)

router.include_router(
    video_detection.router,
    prefix="/detect",
    tags=["Video Detection"]
)

router.include_router(
    stream_detection.router,
    prefix="/stream",
    tags=["Stream Detection"]
)

router.include_router(
    models.router,
    prefix="/models",
    tags=["Model Management"]
)