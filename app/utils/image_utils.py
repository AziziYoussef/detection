"""
🖼️ IMAGE_UTILS - UTILITAIRES POUR TRAITEMENT D'IMAGES
=====================================================
Fonctions utilitaires pour le traitement et la manipulation d'images

Fonctionnalités:
- Redimensionnement intelligent avec conservation du ratio
- Normalisation/dénormalisation pour modèles PyTorch
- Padding et cropping adaptatifs
- Augmentations d'images spécialisées
- Visualisation avec boîtes englobantes
- Sauvegarde d'images annotées
- Optimisations pour différents formats

Formats supportés:
- JPEG, PNG, WEBP, BMP, TIFF
- Conversion automatique entre formats
- Optimisation pour web et mobile
- Support transparence (PNG, WEBP)

Performance:
- Vectorisation avec NumPy/OpenCV
- Support GPU via PyTorch
- Cache intelligent pour opérations répétées
- Optimisations mémoire pour grandes images
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, List, Union, Optional, Dict, Any
from pathlib import Path
import logging
import math
import io
import base64

logger = logging.getLogger(__name__)

# 📐 REDIMENSIONNEMENT ET GÉOMÉTRIE
def resize_with_aspect_ratio(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    target_size: Tuple[int, int],
    method: str = "bilinear",
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[Union[np.ndarray, torch.Tensor, Image.Image], Dict[str, Any]]:
    """
    📐 Redimensionne une image en conservant le ratio d'aspect
    
    Args:
        image: Image d'entrée
        target_size: Taille cible (width, height)
        method: Méthode d'interpolation
        fill_color: Couleur de remplissage pour le padding
        
    Returns:
        Tuple (image_redimensionnée, metadata)
    """
    
    target_width, target_height = target_size
    
    # Détection du type d'image
    if isinstance(image, torch.Tensor):
        return _resize_tensor_with_aspect_ratio(image, target_size, method, fill_color)
    elif isinstance(image, np.ndarray):
        return _resize_numpy_with_aspect_ratio(image, target_size, method, fill_color)
    elif isinstance(image, Image.Image):
        return _resize_pil_with_aspect_ratio(image, target_size, method, fill_color)
    else:
        raise TypeError(f"Type d'image non supporté: {type(image)}")

def _resize_pil_with_aspect_ratio(
    image: Image.Image,
    target_size: Tuple[int, int],
    method: str,
    fill_color: Tuple[int, int, int]
) -> Tuple[Image.Image, Dict[str, Any]]:
    """📐 Redimensionnement PIL avec aspect ratio"""
    
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calcul du ratio optimal
    ratio = min(target_width / original_width, target_height / original_height)
    
    # Nouvelles dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Méthode de redimensionnement
    resample_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS
    }
    resample = resample_map.get(method, Image.BILINEAR)
    
    # Redimensionnement
    resized = image.resize((new_width, new_height), resample)
    
    # Padding si nécessaire
    if new_width != target_width or new_height != target_height:
        # Créer canvas avec couleur de fond
        padded = Image.new("RGB", target_size, fill_color)
        
        # Centrer l'image redimensionnée
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        padded.paste(resized, (x_offset, y_offset))
        final_image = padded
    else:
        final_image = resized
    
    # Métadonnées
    metadata = {
        "original_size": (original_width, original_height),
        "target_size": target_size,
        "new_size": (new_width, new_height),
        "scale_ratio": ratio,
        "padding": (
            (target_width - new_width) // 2,
            (target_height - new_height) // 2
        )
    }
    
    return final_image, metadata

def _resize_numpy_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    method: str,
    fill_color: Tuple[int, int, int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """📐 Redimensionnement NumPy avec aspect ratio"""
    
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]
    
    # Ratio optimal
    ratio = min(target_width / original_width, target_height / original_height)
    
    # Nouvelles dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Méthode OpenCV
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    interpolation = interpolation_map.get(method, cv2.INTER_LINEAR)
    
    # Redimensionnement
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    # Padding
    if new_width != target_width or new_height != target_height:
        # Canvas avec couleur de fond
        if len(image.shape) == 3:
            padded = np.full((target_height, target_width, image.shape[2]), fill_color, dtype=image.dtype)
        else:
            padded = np.full((target_height, target_width), fill_color[0], dtype=image.dtype)
        
        # Centrage
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        final_image = padded
    else:
        final_image = resized
    
    metadata = {
        "original_size": (original_width, original_height),
        "target_size": target_size,
        "new_size": (new_width, new_height),
        "scale_ratio": ratio,
        "padding": (x_offset, y_offset)
    }
    
    return final_image, metadata

def _resize_tensor_with_aspect_ratio(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    method: str,
    fill_color: Tuple[int, int, int]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """📐 Redimensionnement Tensor avec aspect ratio"""
    
    # Convertir en NumPy, traiter, reconvertir
    if image.dim() == 4:  # Batch
        batch_size = image.size(0)
        results = []
        metadatas = []
        
        for i in range(batch_size):
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            if img_np.dtype == np.float32 and img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            resized_np, metadata = _resize_numpy_with_aspect_ratio(
                img_np, target_size, method, fill_color
            )
            
            # Reconversion tensor
            resized_tensor = torch.from_numpy(resized_np).permute(2, 0, 1)
            if image.dtype == torch.float32:
                resized_tensor = resized_tensor.float() / 255.0
            
            results.append(resized_tensor)
            metadatas.append(metadata)
        
        return torch.stack(results), metadatas[0]  # Premier metadata comme représentatif
    
    else:  # Image unique
        if image.dim() == 3:  # CHW
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:  # HW
            img_np = image.cpu().numpy()
        
        if img_np.dtype == np.float32 and img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        resized_np, metadata = _resize_numpy_with_aspect_ratio(
            img_np, target_size, method, fill_color
        )
        
        # Reconversion
        if len(resized_np.shape) == 3:
            resized_tensor = torch.from_numpy(resized_np).permute(2, 0, 1)
        else:
            resized_tensor = torch.from_numpy(resized_np)
        
        if image.dtype == torch.float32:
            resized_tensor = resized_tensor.float() / 255.0
        
        return resized_tensor.to(image.device), metadata

# 📏 NORMALISATION
def normalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    input_range: str = "0-255"
) -> Union[np.ndarray, torch.Tensor]:
    """
    📏 Normalise une image pour les modèles PyTorch
    
    Args:
        image: Image à normaliser
        mean: Moyennes par canal
        std: Écarts-types par canal
        input_range: "0-255" ou "0-1"
        
    Returns:
        Image normalisée
    """
    
    if isinstance(image, torch.Tensor):
        return _normalize_tensor(image, mean, std, input_range)
    else:
        return _normalize_numpy(image, mean, std, input_range)

def _normalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    input_range: str
) -> torch.Tensor:
    """📏 Normalisation tensor"""
    
    # Conversion 0-1 si nécessaire
    if input_range == "0-255":
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif tensor.max() > 1.0:
            tensor = tensor / 255.0
    
    # Application de la normalisation
    mean_tensor = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std_tensor = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    if tensor.dim() == 4:  # Batch [N, C, H, W]
        mean_tensor = mean_tensor.view(1, -1, 1, 1)
        std_tensor = std_tensor.view(1, -1, 1, 1)
    elif tensor.dim() == 3:  # Single image [C, H, W]
        mean_tensor = mean_tensor.view(-1, 1, 1)
        std_tensor = std_tensor.view(-1, 1, 1)
    
    normalized = (tensor - mean_tensor) / std_tensor
    return normalized

def _normalize_numpy(
    array: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    input_range: str
) -> np.ndarray:
    """📏 Normalisation NumPy"""
    
    # Conversion float
    if array.dtype == np.uint8:
        array = array.astype(np.float32)
        if input_range == "0-255":
            array /= 255.0
    elif input_range == "0-255" and array.max() > 1.0:
        array = array / 255.0
    
    # Normalisation
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    
    if array.ndim == 3:  # [H, W, C]
        normalized = (array - mean_array) / std_array
    elif array.ndim == 4:  # [N, H, W, C]
        mean_array = mean_array.reshape(1, 1, 1, -1)
        std_array = std_array.reshape(1, 1, 1, -1)
        normalized = (array - mean_array) / std_array
    else:
        raise ValueError(f"Dimensions non supportées: {array.ndim}")
    
    return normalized

def denormalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    output_range: str = "0-255"
) -> Union[np.ndarray, torch.Tensor]:
    """📏 Dénormalise une image"""
    
    if isinstance(image, torch.Tensor):
        return _denormalize_tensor(image, mean, std, output_range)
    else:
        return _denormalize_numpy(image, mean, std, output_range)

def _denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    output_range: str
) -> torch.Tensor:
    """📏 Dénormalisation tensor"""
    
    mean_tensor = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std_tensor = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    if tensor.dim() == 4:  # Batch
        mean_tensor = mean_tensor.view(1, -1, 1, 1)
        std_tensor = std_tensor.view(1, -1, 1, 1)
    elif tensor.dim() == 3:  # Single image
        mean_tensor = mean_tensor.view(-1, 1, 1)
        std_tensor = std_tensor.view(-1, 1, 1)
    
    # Dénormalisation
    denormalized = tensor * std_tensor + mean_tensor
    
    # Conversion range si demandé
    if output_range == "0-255":
        denormalized = torch.clamp(denormalized * 255, 0, 255)
        if tensor.dtype == torch.uint8:
            denormalized = denormalized.byte()
    else:
        denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized

def _denormalize_numpy(
    array: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    output_range: str
) -> np.ndarray:
    """📏 Dénormalisation NumPy"""
    
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    
    if array.ndim == 3:  # [H, W, C]
        denormalized = array * std_array + mean_array
    elif array.ndim == 4:  # [N, H, W, C]
        mean_array = mean_array.reshape(1, 1, 1, -1)
        std_array = std_array.reshape(1, 1, 1, -1)
        denormalized = array * std_array + mean_array
    else:
        raise ValueError(f"Dimensions non supportées: {array.ndim}")
    
    # Conversion range
    if output_range == "0-255":
        denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
    else:
        denormalized = np.clip(denormalized, 0, 1).astype(np.float32)
    
    return denormalized

# ⬜ PADDING ET CROPPING
def pad_to_square(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Union[np.ndarray, torch.Tensor, Image.Image]:
    """⬜ Ajoute du padding pour rendre l'image carrée"""
    
    if isinstance(image, Image.Image):
        width, height = image.size
        size = max(width, height)
        
        padded = Image.new("RGB", (size, size), fill_color)
        x_offset = (size - width) // 2
        y_offset = (size - height) // 2
        padded.paste(image, (x_offset, y_offset))
        
        return padded
    
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        size = max(height, width)
        
        if len(image.shape) == 3:
            padded = np.full((size, size, image.shape[2]), fill_color, dtype=image.dtype)
        else:
            padded = np.full((size, size), fill_color[0], dtype=image.dtype)
        
        y_offset = (size - height) // 2
        x_offset = (size - width) // 2
        
        padded[y_offset:y_offset + height, x_offset:x_offset + width] = image
        return padded
    
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:  # CHW
            _, height, width = image.shape
            size = max(height, width)
            
            padded = torch.full((image.size(0), size, size), fill_color[0], 
                              dtype=image.dtype, device=image.device)
            
            y_offset = (size - height) // 2
            x_offset = (size - width) // 2
            
            padded[:, y_offset:y_offset + height, x_offset:x_offset + width] = image
            return padded
        
        else:
            raise ValueError("Tensor doit être au format CHW")
    
    else:
        raise TypeError(f"Type non supporté: {type(image)}")

def crop_image(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    bbox: Tuple[int, int, int, int],  # x1, y1, x2, y2
    padding: int = 0
) -> Union[np.ndarray, torch.Tensor, Image.Image]:
    """✂️ Découpe une région de l'image"""
    
    x1, y1, x2, y2 = bbox
    
    # Ajout padding
    if padding > 0:
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding
    
    if isinstance(image, Image.Image):
        width, height = image.size
        x2 = min(width, x2)
        y2 = min(height, y2)
        return image.crop((x1, y1, x2, y2))
    
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        x2 = min(width, x2)
        y2 = min(height, y2)
        return image[y1:y2, x1:x2]
    
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:  # CHW
            _, height, width = image.shape
            x2 = min(width, x2)
            y2 = min(height, y2)
            return image[:, y1:y2, x1:x2]
        else:
            raise ValueError("Tensor doit être au format CHW")
    
    else:
        raise TypeError(f"Type non supporté: {type(image)}")

# 🎨 AUGMENTATIONS
def apply_image_augmentation(
    image: Union[np.ndarray, Image.Image],
    augmentation_type: str,
    **params
) -> Union[np.ndarray, Image.Image]:
    """🎨 Applique une augmentation spécifique"""
    
    if augmentation_type == "brightness":
        return adjust_brightness(image, params.get("factor", 1.2))
    elif augmentation_type == "contrast":
        return adjust_contrast(image, params.get("factor", 1.2))
    elif augmentation_type == "saturation":
        return adjust_saturation(image, params.get("factor", 1.2))
    elif augmentation_type == "blur":
        return apply_blur(image, params.get("radius", 1))
    elif augmentation_type == "noise":
        return add_noise(image, params.get("intensity", 0.1))
    else:
        raise ValueError(f"Augmentation non supportée: {augmentation_type}")

def adjust_brightness(
    image: Union[np.ndarray, Image.Image],
    factor: float
) -> Union[np.ndarray, Image.Image]:
    """☀️ Ajuste la luminosité"""
    
    if isinstance(image, Image.Image):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    else:
        # NumPy
        if image.dtype == np.uint8:
            adjusted = image.astype(np.float32) * factor
            return np.clip(adjusted, 0, 255).astype(np.uint8)
        else:
            return np.clip(image * factor, 0, 1)

def adjust_contrast(
    image: Union[np.ndarray, Image.Image],
    factor: float
) -> Union[np.ndarray, Image.Image]:
    """🌓 Ajuste le contraste"""
    
    if isinstance(image, Image.Image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    else:
        # NumPy
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        
        if image.dtype == np.uint8:
            return np.clip(adjusted, 0, 255).astype(np.uint8)
        else:
            return np.clip(adjusted, 0, 1)

def adjust_saturation(
    image: Union[np.ndarray, Image.Image],
    factor: float
) -> Union[np.ndarray, Image.Image]:
    """🎨 Ajuste la saturation"""
    
    if isinstance(image, Image.Image):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    else:
        # NumPy - conversion HSV
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        else:
            return image  # Pas de saturation pour images en niveaux de gris

def apply_blur(
    image: Union[np.ndarray, Image.Image],
    radius: float
) -> Union[np.ndarray, Image.Image]:
    """🌫️ Applique un flou"""
    
    if isinstance(image, Image.Image):
        return image.filter(ImageFilter.GaussianBlur(radius))
    else:
        # NumPy avec OpenCV
        kernel_size = int(radius * 2) * 2 + 1  # Impair
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)

def add_noise(
    image: Union[np.ndarray, Image.Image],
    intensity: float
) -> Union[np.ndarray, Image.Image]:
    """📡 Ajoute du bruit gaussien"""
    
    if isinstance(image, Image.Image):
        # Convertir en NumPy, ajouter bruit, reconvertir
        img_array = np.array(image)
        noisy_array = add_noise(img_array, intensity)
        return Image.fromarray(noisy_array)
    else:
        # NumPy
        noise = np.random.normal(0, intensity * 255, image.shape)
        
        if image.dtype == np.uint8:
            noisy = image.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            noisy = image + (noise / 255.0)
            return np.clip(noisy, 0, 1)

# 🎯 VISUALISATION
def draw_bounding_boxes(
    image: Union[np.ndarray, Image.Image],
    boxes: List[List[float]],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_size: int = 16
) -> Union[np.ndarray, Image.Image]:
    """
    🎯 Dessine des boîtes englobantes sur une image
    
    Args:
        image: Image de base
        boxes: Liste de boîtes [x1, y1, x2, y2]
        labels: Labels des objets (optionnel)
        scores: Scores de confiance (optionnel)
        colors: Couleurs des boîtes (optionnel)
        thickness: Épaisseur des lignes
        font_size: Taille de la police
        
    Returns:
        Image avec annotations
    """
    
    if isinstance(image, np.ndarray):
        return _draw_boxes_opencv(image, boxes, labels, scores, colors, thickness, font_size)
    else:
        return _draw_boxes_pil(image, boxes, labels, scores, colors, thickness, font_size)

def _draw_boxes_pil(
    image: Image.Image,
    boxes: List[List[float]],
    labels: Optional[List[str]],
    scores: Optional[List[float]],
    colors: Optional[List[Tuple[int, int, int]]],
    thickness: int,
    font_size: int
) -> Image.Image:
    """🎯 Dessin avec PIL"""
    
    # Copie pour éviter modification originale
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Police
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Couleurs par défaut
    if colors is None:
        default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(boxes))]
    
    # Dessin des boîtes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        color = colors[i % len(colors)]
        
        # Rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Label et score
        if labels or scores:
            text_parts = []
            if labels and i < len(labels):
                text_parts.append(labels[i])
            if scores and i < len(scores):
                text_parts.append(f"{scores[i]:.2f}")
            
            if text_parts:
                text = " - ".join(text_parts)
                
                # Fond du texte
                bbox = draw.textbbox((x1, y1 - font_size - 2), text, font=font)
                draw.rectangle(bbox, fill=color)
                
                # Texte
                draw.text((x1, y1 - font_size - 2), text, fill=(255, 255, 255), font=font)
    
    return annotated

def _draw_boxes_opencv(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: Optional[List[str]],
    scores: Optional[List[float]],
    colors: Optional[List[Tuple[int, int, int]]],
    thickness: int,
    font_size: int
) -> np.ndarray:
    """🎯 Dessin avec OpenCV"""
    
    # Copie
    annotated = image.copy()
    
    # Couleurs par défaut
    if colors is None:
        default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(boxes))]
    
    # Dessin
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        color = colors[i % len(colors)]
        
        # Rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Label et score
        if labels or scores:
            text_parts = []
            if labels and i < len(labels):
                text_parts.append(labels[i])
            if scores and i < len(scores):
                text_parts.append(f"{scores[i]:.2f}")
            
            if text_parts:
                text = " - ".join(text_parts)
                
                # Taille du texte
                font_scale = font_size / 20.0
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Fond du texte
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - baseline - 2),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Texte
                cv2.putText(
                    annotated,
                    text,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1
                )
    
    return annotated

# 💾 SAUVEGARDE
def save_annotated_image(
    image: Union[np.ndarray, Image.Image],
    output_path: Union[str, Path],
    quality: int = 95,
    optimize: bool = True
) -> bool:
    """
    💾 Sauvegarde une image annotée
    
    Args:
        image: Image à sauvegarder
        output_path: Chemin de sortie
        quality: Qualité JPEG (0-100)
        optimize: Optimiser la taille
        
    Returns:
        Succès de la sauvegarde
    """
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(image, np.ndarray):
            # Conversion BGR si OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # PIL pour sauvegarde
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Options de sauvegarde selon format
        save_kwargs = {"optimize": optimize}
        
        if output_path.suffix.lower() in [".jpg", ".jpeg"]:
            save_kwargs["quality"] = quality
            save_kwargs["format"] = "JPEG"
        elif output_path.suffix.lower() == ".png":
            save_kwargs["format"] = "PNG"
        elif output_path.suffix.lower() == ".webp":
            save_kwargs["quality"] = quality
            save_kwargs["format"] = "WEBP"
        
        pil_image.save(output_path, **save_kwargs)
        
        logger.debug(f"💾 Image sauvegardée: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde {output_path}: {e}")
        return False

# 🔄 CONVERSIONS
def image_to_base64(
    image: Union[np.ndarray, Image.Image],
    format: str = "JPEG",
    quality: int = 85
) -> str:
    """🔄 Convertit une image en base64"""
    
    if isinstance(image, np.ndarray):
        # Conversion vers PIL
        if len(image.shape) == 3:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(image)
    
    # Sauvegarde en mémoire
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    buffer.seek(0)
    
    # Encodage base64
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

def base64_to_image(encoded: str) -> Image.Image:
    """🔄 Convertit base64 en image"""
    
    image_data = base64.b64decode(encoded)
    buffer = io.BytesIO(image_data)
    return Image.open(buffer)

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "resize_with_aspect_ratio",
    "normalize_image",
    "denormalize_image",
    "pad_to_square",
    "crop_image",
    "apply_image_augmentation",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_saturation",
    "apply_blur",
    "add_noise",
    "draw_bounding_boxes",
    "save_annotated_image",
    "image_to_base64",
    "base64_to_image"
]