"""
🔧 UTILS PACKAGE - UTILITAIRES ET FONCTIONS HELPERS
==================================================
Package contenant tous les utilitaires pour la détection d'objets perdus

Ce package fournit:
- Fonctions de manipulation de boîtes englobantes
- Fonctions de perte pour l'entraînement
- Utilitaires pour datasets et data loading
- Traitement d'images et vidéos
- Métriques et évaluation
- Visualisation et debugging

Modules:
- box_utils: Manipulation boîtes (IoU, NMS, encodage/décodage)
- losses: Fonctions de perte (Focal Loss, Smooth L1, etc.)
- dataset: Utilitaires datasets et augmentations
- image_utils: Traitement d'images (resize, crop, normalize)
- video_utils: Traitement vidéos (extraction frames, encodage)

Utilisation:
```python
from app.utils import box_utils, losses
from app.utils.image_utils import resize_with_aspect_ratio
from app.utils.losses import FocalLoss, SmoothL1Loss
```

Ces utilitaires sont utilisés partout dans le projet:
- Modèles pour calcul des pertes
- Services pour post-traitement
- API endpoints pour validation
- Scripts d'entraînement et évaluation
"""

from .box_utils import (
    calculate_iou,
    apply_nms,
    encode_boxes,
    decode_boxes,
    convert_box_format,
    filter_boxes_by_score,
    box_area,
    clip_boxes_to_image
)

from .losses import (
    FocalLoss,
    SmoothL1Loss,
    DetectionLoss,
    IoULoss,
    GIoULoss,
    calculate_classification_loss,
    calculate_regression_loss
)

from .dataset import (
    DatasetUtils,
    ImageAugmentation,
    create_data_loader,
    collate_fn,
    parse_annotation_format
)

from .image_utils import (
    resize_with_aspect_ratio,
    normalize_image,
    denormalize_image,
    pad_to_square,
    crop_image,
    apply_image_augmentation,
    draw_bounding_boxes,
    save_annotated_image
)

from .video_utils import (
    extract_video_frames,
    encode_video_with_annotations,
    get_video_info,
    create_video_from_frames,
    optimize_video_for_web,
    extract_keyframes
)

__all__ = [
    # Box utilities
    "calculate_iou",
    "apply_nms", 
    "encode_boxes",
    "decode_boxes",
    "convert_box_format",
    "filter_boxes_by_score",
    "box_area",
    "clip_boxes_to_image",
    
    # Loss functions
    "FocalLoss",
    "SmoothL1Loss",
    "DetectionLoss", 
    "IoULoss",
    "GIoULoss",
    "calculate_classification_loss",
    "calculate_regression_loss",
    
    # Dataset utilities
    "DatasetUtils",
    "ImageAugmentation",
    "create_data_loader",
    "collate_fn",
    "parse_annotation_format",
    
    # Image utilities
    "resize_with_aspect_ratio",
    "normalize_image",
    "denormalize_image", 
    "pad_to_square",
    "crop_image",
    "apply_image_augmentation",
    "draw_bounding_boxes",
    "save_annotated_image",
    
    # Video utilities
    "extract_video_frames",
    "encode_video_with_annotations",
    "get_video_info",
    "create_video_from_frames", 
    "optimize_video_for_web",
    "extract_keyframes"
]