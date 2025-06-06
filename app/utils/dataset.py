"""
📊 DATASET - UTILITAIRES POUR DATASETS ET AUGMENTATION DE DONNÉES
================================================================
Utilitaires pour la gestion des datasets de détection d'objets perdus

Fonctionnalités:
- Parsers pour différents formats d'annotations (COCO, YOLO, PASCAL VOC)
- Augmentations d'images spécialisées pour détection
- DataLoaders optimisés avec collate functions
- Stratégies de sampling et balancement
- Validation et nettoyage des datasets
- Support multi-formats et multi-sources

Formats supportés:
- COCO JSON: Standard industrie
- YOLO: Format texte simple
- PASCAL VOC: XML annotations
- Custom: Format maison configurable

Pipeline d'augmentation:
Image + Boxes → Transformations → Augmented Image + Boxes
     ↓              ↓                    ↓
 Original    Flip, Rotate, Scale    Synchronized
   Data      Color, Noise, etc.      Transforms
"""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# 📋 CLASSES UTILITAIRES
class DatasetUtils:
    """📋 Utilitaires généraux pour datasets"""
    
    @staticmethod
    def validate_dataset(
        images_dir: Path,
        annotations_file: Path,
        min_images: int = 10
    ) -> Dict[str, Any]:
        """
        ✅ Valide un dataset de détection
        
        Args:
            images_dir: Dossier des images
            annotations_file: Fichier d'annotations
            min_images: Nombre minimum d'images requis
            
        Returns:
            Rapport de validation
        """
        
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Vérifier existence des dossiers
            if not images_dir.exists():
                report["errors"].append(f"Dossier images non trouvé: {images_dir}")
                report["valid"] = False
            
            if not annotations_file.exists():
                report["errors"].append(f"Fichier annotations non trouvé: {annotations_file}")
                report["valid"] = False
            
            if not report["valid"]:
                return report
            
            # Charger annotations
            annotations = DatasetUtils.load_annotations(annotations_file)
            
            # Statistiques de base
            num_images = len(annotations)
            num_objects = sum(len(ann.get("boxes", [])) for ann in annotations)
            
            if num_images < min_images:
                report["errors"].append(f"Pas assez d'images: {num_images} < {min_images}")
                report["valid"] = False
            
            # Vérifier cohérence images/annotations
            missing_images = []
            for ann in annotations:
                image_path = images_dir / ann["image_name"]
                if not image_path.exists():
                    missing_images.append(ann["image_name"])
            
            if missing_images:
                report["warnings"].append(f"{len(missing_images)} images manquantes")
                if len(missing_images) > num_images * 0.1:  # Plus de 10% manquantes
                    report["errors"].append("Trop d'images manquantes")
                    report["valid"] = False
            
            # Statistiques
            report["stats"] = {
                "num_images": num_images,
                "num_objects": num_objects,
                "avg_objects_per_image": num_objects / max(1, num_images),
                "missing_images": len(missing_images)
            }
            
        except Exception as e:
            report["errors"].append(f"Erreur validation: {e}")
            report["valid"] = False
        
        return report
    
    @staticmethod
    def load_annotations(annotations_file: Path) -> List[Dict[str, Any]]:
        """📖 Charge les annotations selon le format"""
        
        file_ext = annotations_file.suffix.lower()
        
        if file_ext == ".json":
            return DatasetUtils.load_coco_annotations(annotations_file)
        elif file_ext == ".txt":
            return DatasetUtils.load_yolo_annotations(annotations_file)
        elif file_ext == ".xml":
            return DatasetUtils.load_voc_annotations(annotations_file)
        else:
            raise ValueError(f"Format d'annotation non supporté: {file_ext}")
    
    @staticmethod
    def load_coco_annotations(json_file: Path) -> List[Dict[str, Any]]:
        """📖 Charge annotations format COCO"""
        
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        # Index par image_id
        images_by_id = {img["id"]: img for img in coco_data["images"]}
        categories_by_id = {cat["id"]: cat for cat in coco_data["categories"]}
        
        # Grouper annotations par image
        annotations_by_image = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Conversion format unifié
        unified_annotations = []
        
        for image_id, image_info in images_by_id.items():
            image_anns = annotations_by_image.get(image_id, [])
            
            boxes = []
            labels = []
            
            for ann in image_anns:
                # COCO: [x, y, width, height]
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])  # Conversion vers xyxy
                labels.append(ann["category_id"])
            
            unified_annotations.append({
                "image_name": image_info["file_name"],
                "image_width": image_info["width"],
                "image_height": image_info["height"],
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id
            })
        
        return unified_annotations
    
    @staticmethod
    def load_yolo_annotations(txt_file: Path) -> List[Dict[str, Any]]:
        """📖 Charge annotations format YOLO"""
        
        # YOLO utilise un fichier par image
        # Pour simplifier, on assume une liste d'images
        
        annotations = []
        images_dir = txt_file.parent / "images"
        labels_dir = txt_file.parent / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError("Structure YOLO requiert dossiers 'images' et 'labels'")
        
        for image_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            boxes = []
            labels = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Conversion vers format absolu (à faire avec dimensions image)
                            # Pour l'instant, garder format YOLO normalisé
                            boxes.append([x_center, y_center, width, height])
                            labels.append(class_id)
            
            # Obtenir dimensions image
            try:
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
            except:
                img_width, img_height = 640, 640  # Valeurs par défaut
            
            annotations.append({
                "image_name": image_file.name,
                "image_width": img_width,
                "image_height": img_height,
                "boxes": boxes,
                "labels": labels,
                "format": "yolo_normalized"
            })
        
        return annotations
    
    @staticmethod
    def load_voc_annotations(xml_file: Path) -> List[Dict[str, Any]]:
        """📖 Charge annotations format PASCAL VOC"""
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Info image
        filename = root.find("filename").text
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)
        
        # Objets
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            
            x1 = int(bbox.find("xmin").text)
            y1 = int(bbox.find("ymin").text)
            x2 = int(bbox.find("xmax").text)
            y2 = int(bbox.find("ymax").text)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(name)  # Nom de classe au lieu d'ID
        
        return [{
            "image_name": filename,
            "image_width": img_width,
            "image_height": img_height,
            "boxes": boxes,
            "labels": labels
        }]

# 🎨 AUGMENTATION D'IMAGES
class ImageAugmentation:
    """🎨 Augmentations spécialisées pour détection d'objets"""
    
    def __init__(
        self,
        mode: str = "training",
        image_size: Tuple[int, int] = (640, 640),
        augment_prob: float = 0.5
    ):
        self.mode = mode
        self.image_size = image_size
        self.augment_prob = augment_prob
        
        # Transformations selon le mode
        if mode == "training":
            self.transform = self._get_training_transforms()
        elif mode == "validation":
            self.transform = self._get_validation_transforms()
        else:
            self.transform = self._get_inference_transforms()
        
        logger.debug(f"🎨 ImageAugmentation: {mode}, size={image_size}")
    
    def _get_training_transforms(self) -> A.Compose:
        """🏋️ Transformations d'entraînement"""
        
        return A.Compose([
            # Transformations géométriques
            A.LongestMaxSize(max_size=self.image_size[0]),
            A.PadIfNeeded(
                min_height=self.image_size[1],
                min_width=self.image_size[0],
                border_mode=cv2.BORDER_CONSTANT,
                value=114  # Gris foncé
            ),
            
            # Augmentations spatiales
            A.RandomResizedCrop(
                height=self.image_size[1],
                width=self.image_size[0],
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2),
                p=0.3
            ),
            
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.3
            ),
            
            # Augmentations de couleur
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),
            
            # Augmentations de bruit/texture
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.2),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.ISONoise(intensity=(0.1, 0.5)),
            ], p=0.2),
            
            # Normalisation finale
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
            
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _get_validation_transforms(self) -> A.Compose:
        """✅ Transformations de validation"""
        
        return A.Compose([
            A.LongestMaxSize(max_size=self.image_size[0]),
            A.PadIfNeeded(
                min_height=self.image_size[1],
                min_width=self.image_size[0],
                border_mode=cv2.BORDER_CONSTANT,
                value=114
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
    
    def _get_inference_transforms(self) -> A.Compose:
        """🔍 Transformations d'inférence"""
        
        return A.Compose([
            A.LongestMaxSize(max_size=self.image_size[0]),
            A.PadIfNeeded(
                min_height=self.image_size[1],
                min_width=self.image_size[0],
                border_mode=cv2.BORDER_CONSTANT,
                value=114
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: Optional[List[List[float]]] = None,
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """🔄 Applique les transformations"""
        
        # Préparation des inputs pour Albumentations
        transform_input = {"image": image}
        
        if boxes is not None and labels is not None:
            transform_input["bboxes"] = boxes
            transform_input["class_labels"] = labels
        
        # Application des transformations
        try:
            transformed = self.transform(**transform_input)
            
            result = {
                "image": transformed["image"],
                "original_size": image.shape[:2]
            }
            
            if "bboxes" in transformed:
                result["boxes"] = transformed["bboxes"]
                result["labels"] = transformed["class_labels"]
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur augmentation: {e}")
            # Fallback sans augmentation
            basic_transform = A.Compose([
                A.Resize(self.image_size[1], self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            transformed = basic_transform(image=image)
            return {
                "image": transformed["image"],
                "boxes": boxes or [],
                "labels": labels or [],
                "original_size": image.shape[:2]
            }

# 🗂️ DATASET CUSTOM
class DetectionDataset(Dataset):
    """🗂️ Dataset pour détection d'objets perdus"""
    
    def __init__(
        self,
        images_dir: Path,
        annotations: List[Dict[str, Any]],
        class_mapping: Dict[str, int],
        transform: Optional[ImageAugmentation] = None,
        cache_images: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.class_mapping = class_mapping
        self.transform = transform
        self.cache_images = cache_images
        
        # Cache pour images si activé
        self._image_cache = {} if cache_images else None
        
        # Validation
        self._validate_annotations()
        
        logger.info(f"🗂️ DetectionDataset: {len(annotations)} images, cache={cache_images}")
    
    def _validate_annotations(self):
        """✅ Valide les annotations"""
        
        valid_annotations = []
        
        for ann in self.annotations:
            image_path = self.images_dir / ann["image_name"]
            
            if not image_path.exists():
                logger.warning(f"⚠️ Image manquante: {image_path}")
                continue
            
            # Vérifier boîtes valides
            if "boxes" in ann and ann["boxes"]:
                valid_boxes = []
                valid_labels = []
                
                for box, label in zip(ann["boxes"], ann["labels"]):
                    # Vérifier format boîte
                    if len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                        # Convertir label si nécessaire
                        if isinstance(label, str) and label in self.class_mapping:
                            label = self.class_mapping[label]
                        
                        valid_boxes.append(box)
                        valid_labels.append(label)
                
                ann["boxes"] = valid_boxes
                ann["labels"] = valid_labels
            
            valid_annotations.append(ann)
        
        self.annotations = valid_annotations
        logger.info(f"✅ {len(valid_annotations)} annotations validées")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """🔍 Récupère un échantillon"""
        
        ann = self.annotations[idx]
        
        # Chargement image
        image = self._load_image(ann["image_name"])
        
        # Préparation des targets
        boxes = ann.get("boxes", [])
        labels = ann.get("labels", [])
        
        # Application des transformations
        if self.transform:
            transformed = self.transform(image, boxes, labels)
            
            return {
                "image": transformed["image"],
                "boxes": torch.tensor(transformed.get("boxes", []), dtype=torch.float32),
                "labels": torch.tensor(transformed.get("labels", []), dtype=torch.long),
                "image_id": idx,
                "original_size": transformed["original_size"]
            }
        else:
            # Sans transformation
            return {
                "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "image_id": idx,
                "original_size": image.shape[:2]
            }
    
    def _load_image(self, image_name: str) -> np.ndarray:
        """📖 Charge une image"""
        
        # Vérifier cache
        if self._image_cache and image_name in self._image_cache:
            return self._image_cache[image_name]
        
        # Chargement
        image_path = self.images_dir / image_name
        
        try:
            # Utiliser OpenCV pour cohérence avec Albumentations
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Cache si activé
            if self._image_cache:
                self._image_cache[image_name] = image
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement image {image_path}: {e}")
            # Image par défaut
            return np.zeros((480, 640, 3), dtype=np.uint8)

# 🔧 FONCTIONS UTILITAIRES
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    🔧 Fonction de collation pour DataLoader
    
    Gère les batches de tailles variables pour détection
    """
    
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample["image"])
        
        # Target par image
        target = {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "image_id": sample["image_id"],
            "original_size": sample["original_size"]
        }
        targets.append(target)
    
    # Stack des images
    images = torch.stack(images, dim=0)
    
    return {
        "images": images,
        "targets": targets
    }

def create_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """🔧 Crée un DataLoader optimisé pour détection"""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )

def parse_annotation_format(
    annotation_file: Path,
    format_type: str = "auto"
) -> str:
    """🔍 Détecte automatiquement le format d'annotation"""
    
    if format_type != "auto":
        return format_type
    
    file_ext = annotation_file.suffix.lower()
    
    if file_ext == ".json":
        # Vérifier si c'est du COCO
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                if "images" in data and "annotations" in data and "categories" in data:
                    return "coco"
        except:
            pass
        return "json"
    
    elif file_ext == ".xml":
        return "pascal_voc"
    
    elif file_ext == ".txt":
        return "yolo"
    
    else:
        raise ValueError(f"Format non reconnu pour {annotation_file}")

def split_dataset(
    annotations: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """✂️ Divise le dataset en train/val/test"""
    
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Les ratios doivent sommer à 1.0")
    
    random.seed(random_seed)
    shuffled = annotations.copy()
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]
    
    logger.info(f"✂️ Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_set, val_set, test_set

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "DatasetUtils",
    "ImageAugmentation",
    "DetectionDataset",
    "collate_fn",
    "create_data_loader",
    "parse_annotation_format",
    "split_dataset"
]