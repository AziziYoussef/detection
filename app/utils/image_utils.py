# app/utils/image_utils.py - Version minimale sans torchvision
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional, Union
import torch
# import torchvision.transforms as transforms  # Commenté temporairement
from pathlib import Path

class ImageProcessor:
    """Processeur d'images pour la détection - Version simplifiée"""
    
    def __init__(self, target_size: Tuple[int, int] = (320, 320)):
        self.target_size = target_size
        # Transforms simple sans torchvision
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, dict]:
        """
        Prétraite une image pour l'inférence - Version simplifiée
        """
        original_shape = image.shape[:2]  # (H, W)
        
        # Conversion BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Redimensionnement simple
        resized = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalisation manuelle
        resized = resized.astype(np.float32) / 255.0
        
        # Normalisation avec mean/std
        for i in range(3):
            resized[:, :, i] = (resized[:, :, i] - self.mean[i]) / self.std[i]
        
        # Conversion en tensor
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        # Métadonnées
        info = {
            'original_shape': original_shape,
            'resized_shape': resized.shape[:2],
            'scale_factors': (1.0, 1.0),  # Simplifié
            'padding': (0, 0, 0, 0)
        }
        
        return tensor, info
    
    def postprocess_detections(self, detections: torch.Tensor, 
                             transform_info: dict) -> np.ndarray:
        """Post-traite les détections - Version simplifiée"""
        if detections.numel() == 0:
            return np.array([]).reshape(0, 6)
        
        detections_np = detections.cpu().numpy()
        
        # Mise à l'échelle simple vers l'image originale
        original_shape = transform_info['original_shape']
        h_scale = original_shape[0] / self.target_size[1]
        w_scale = original_shape[1] / self.target_size[0]
        
        if detections_np.shape[1] >= 4:
            detections_np[:, 0] *= w_scale  # x1
            detections_np[:, 1] *= h_scale  # y1
            detections_np[:, 2] *= w_scale  # x2
            detections_np[:, 3] *= h_scale  # y2
        
        return detections_np

def encode_image_to_base64(image: np.ndarray, format: str = 'JPEG', 
                          quality: int = 80) -> str:
    """Encode une image en base64"""
    if format.upper() == 'JPEG':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
    elif format.upper() == 'PNG':
        _, buffer = cv2.imencode('.png', image)
    else:
        raise ValueError(f"Format non supporté: {format}")
    
    return base64.b64encode(buffer).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """Décode une image base64 en numpy array"""
    try:
        # Suppression du préfixe data:image si présent
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Décodage
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Impossible de décoder l'image")
        
        return image
    except Exception as e:
        raise ValueError(f"Erreur décodage base64: {e}")

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """Redimensionne une image"""
    if not keep_aspect_ratio:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Centrer dans l'image cible
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def validate_image(image: np.ndarray) -> bool:
    """Valide qu'une image est correcte"""
    if image is None:
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [3, 4]:
        return False
    
    if image.size == 0:
        return False
    
    return True

def get_image_info(image: np.ndarray) -> dict:
    """Retourne les informations d'une image"""
    if not validate_image(image):
        return {}
    
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes,
        'aspect_ratio': w / h if h != 0 else 0
    }
