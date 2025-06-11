# app/api/endpoints/image_detection.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request, Depends
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import io

from app.schemas.detection import DetectionRequest, DetectionResponse
from app.utils.image_utils import decode_base64_to_image, get_image_info, validate_image
from app.core.model_manager import ModelManager

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_model_manager(request: Request) -> ModelManager:
    """Récupère le gestionnaire de modèles depuis l'état de l'application"""
    return request.app.state.model_manager

@router.post("/image", response_model=DetectionResponse)
async def detect_objects_in_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    model_name: Optional[str] = Form("stable_epoch_30"),
    confidence_threshold: Optional[float] = Form(0.5),
    nms_threshold: Optional[float] = Form(0.5),
    max_detections: Optional[int] = Form(50),
    enable_tracking: bool = Form(True),
    enable_lost_detection: bool = Form(True),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🔍 Détecte les objets dans une image
    
    **Entrées supportées:**
    - Upload de fichier image (multipart/form-data)
    - Image encodée en base64 (form field)
    
    **Paramètres:**
    - model_name: Nom du modèle à utiliser
    - confidence_threshold: Seuil de confiance (0.0-1.0)
    - nms_threshold: Seuil pour Non-Maximum Suppression
    - max_detections: Nombre maximum d'objets détectés
    - enable_tracking: Activer le suivi d'objets
    - enable_lost_detection: Activer la détection d'objets perdus
    """
    start_time = time.time()
    
    try:
        # Validation des entrées
        if not file and not image_base64:
            raise HTTPException(
                status_code=400,
                detail="Aucune image fournie. Utilisez 'file' ou 'image_base64'"
            )
        
        # Chargement de l'image
        if file:
            # Upload de fichier
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Type de fichier non supporté: {file.content_type}"
                )
            
            # Lecture du fichier
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Impossible de décoder l'image uploadée"
                )
        
        else:
            # Image base64
            try:
                image = decode_base64_to_image(image_base64)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Erreur décodage base64: {str(e)}"
                )
        
        # Validation de l'image
        if not validate_image(image):
            raise HTTPException(
                status_code=400,
                detail="Image invalide ou corrompue"
            )
        
        # Informations sur l'image
        img_info = get_image_info(image)
        logger.info(f"Image reçue: {img_info['width']}x{img_info['height']}, "
                   f"{img_info['size_bytes']/1024:.1f} KB")
        
        # Validation des paramètres
        if not (0.0 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="confidence_threshold doit être entre 0.0 et 1.0"
            )
        
        if not (0.0 <= nms_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="nms_threshold doit être entre 0.0 et 1.0"
            )
        
        if not (1 <= max_detections <= 100):
            raise HTTPException(
                status_code=400,
                detail="max_detections doit être entre 1 et 100"
            )
        
        # Récupération du détecteur
        try:
            detector = await model_manager.get_detector(model_name)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Modèle non disponible: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erreur interne: impossible de charger le modèle"
            )
        
        # Détection
        detection_start = time.time()
        
        try:
            objects, persons, alerts = detector.detect(
                image,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections,
                enable_tracking=enable_tracking,
                enable_lost_detection=enable_lost_detection
            )
        except Exception as e:
            logger.error(f"Erreur lors de la détection: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la détection: {str(e)}"
            )
        
        detection_time = (time.time() - detection_start) * 1000  # en ms
        total_time = (time.time() - start_time) * 1000  # en ms
        
        # Mise à jour des statistiques
        model_manager.update_inference_stats(model_name, detection_time)
        
        # Comptage des objets par statut
        lost_count = sum(1 for obj in objects if obj.status.value in ['lost', 'critical'])
        suspect_count = sum(1 for obj in objects if obj.status.value in ['suspect', 'surveillance'])
        
        # Création de la réponse
        response = DetectionResponse(
            success=True,
            timestamp=datetime.now(),
            processing_time=total_time,
            objects=objects,
            persons=persons,
            total_objects=len(objects),
            lost_objects=lost_count,
            suspect_objects=suspect_count,
            image_info=img_info,
            model_used=model_name
        )
        
        # Log des résultats
        logger.info(f"Détection terminée: {len(objects)} objets, {len(persons)} personnes, "
                   f"{len(alerts)} alertes en {total_time:.1f}ms")
        
        if alerts:
            logger.warning(f"🚨 {len(alerts)} alertes générées!")
            for alert in alerts:
                logger.warning(f"  - {alert.message}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue dans detect_objects_in_image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du serveur: {str(e)}"
        )

@router.post("/image/analyze", response_model=Dict[str, Any])
async def analyze_image_detailed(
    request: Request,
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("stable_epoch_30"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    🔬 Analyse détaillée d'une image avec métrics avancés
    
    Retourne des informations détaillées sur:
    - Qualité de l'image
    - Performance de détection
    - Analyse spatiale des objets
    - Recommandations d'optimisation
    """
    start_time = time.time()
    
    try:
        # Validation du fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté: {file.content_type}"
            )
        
        # Lecture de l'image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Impossible de décoder l'image"
            )
        
        # Analyse de qualité d'image
        quality_metrics = _analyze_image_quality(image)
        
        # Détection avec différents seuils
        detector = await model_manager.get_detector(model_name)
        
        # Test avec différents seuils de confiance
        confidence_tests = [0.3, 0.5, 0.7, 0.9]
        detection_results = {}
        
        for conf_threshold in confidence_tests:
            objects, persons, alerts = detector.detect(
                image,
                confidence_threshold=conf_threshold,
                enable_tracking=False,
                enable_lost_detection=False
            )
            
            detection_results[f"confidence_{conf_threshold}"] = {
                "objects_count": len(objects),
                "persons_count": len(persons),
                "alerts_count": len(alerts),
                "avg_confidence": np.mean([obj.confidence for obj in objects]) if objects else 0
            }
        
        # Analyse spatiale
        spatial_analysis = _analyze_spatial_distribution(image, objects if 'objects' in locals() else [])
        
        # Recommandations
        recommendations = _generate_recommendations(quality_metrics, detection_results, spatial_analysis)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "analysis_time_ms": total_time,
            "image_quality": quality_metrics,
            "detection_performance": detection_results,
            "spatial_analysis": spatial_analysis,
            "recommendations": recommendations,
            "model_used": model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur dans analyze_image_detailed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )

def _analyze_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Analyse la qualité d'une image"""
    h, w = image.shape[:2]
    
    # Calcul de la netteté (variance du Laplacien)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calcul du contraste
    contrast = gray.std()
    
    # Luminosité moyenne
    brightness = gray.mean()
    
    # Distribution des couleurs
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_diversity = (hist > 0).sum() / hist.size
    
    # Évaluation globale
    quality_score = 0
    quality_factors = []
    
    if sharpness > 100:
        quality_score += 25
        quality_factors.append("✅ Netteté correcte")
    else:
        quality_factors.append("⚠️ Image floue détectée")
    
    if 50 < brightness < 200:
        quality_score += 25
        quality_factors.append("✅ Luminosité correcte")
    else:
        quality_factors.append("⚠️ Problème de luminosité")
    
    if contrast > 30:
        quality_score += 25
        quality_factors.append("✅ Contraste suffisant")
    else:
        quality_factors.append("⚠️ Contraste faible")
    
    if color_diversity > 0.1:
        quality_score += 25
        quality_factors.append("✅ Diversité des couleurs")
    else:
        quality_factors.append("⚠️ Image monotone")
    
    return {
        "resolution": {"width": w, "height": h, "megapixels": (w * h) / 1e6},
        "sharpness": float(sharpness),
        "contrast": float(contrast),
        "brightness": float(brightness),
        "color_diversity": float(color_diversity),
        "quality_score": quality_score,
        "quality_factors": quality_factors,
        "quality_level": "Excellente" if quality_score >= 75 else 
                        "Bonne" if quality_score >= 50 else
                        "Moyenne" if quality_score >= 25 else "Faible"
    }

def _analyze_spatial_distribution(image: np.ndarray, objects: list) -> Dict[str, Any]:
    """Analyse la distribution spatiale des objets"""
    h, w = image.shape[:2]
    
    if not objects:
        return {
            "objects_count": 0,
            "coverage": 0,
            "density": 0,
            "zones": {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        }
    
    # Zones de l'image
    zones = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
    total_area = 0
    
    for obj in objects:
        bbox = obj.bounding_box
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2
        
        # Classification par zone
        if center_x < w/2 and center_y < h/2:
            zones["top_left"] += 1
        elif center_x >= w/2 and center_y < h/2:
            zones["top_right"] += 1
        elif center_x < w/2 and center_y >= h/2:
            zones["bottom_left"] += 1
        else:
            zones["bottom_right"] += 1
        
        # Aire totale couverte
        total_area += bbox.area()
    
    coverage = (total_area / (w * h)) * 100  # Pourcentage de couverture
    density = len(objects) / ((w * h) / 1000)  # Objets per 1000 pixels
    
    return {
        "objects_count": len(objects),
        "coverage_percent": float(coverage),
        "density": float(density),
        "zones": zones,
        "average_object_size": float(total_area / len(objects)) if objects else 0
    }

def _generate_recommendations(quality_metrics: dict, detection_results: dict, 
                            spatial_analysis: dict) -> List[str]:
    """Génère des recommandations d'optimisation"""
    recommendations = []
    
    # Recommandations qualité image
    if quality_metrics["quality_score"] < 50:
        recommendations.append("🔧 Améliorer la qualité d'image pour de meilleures détections")
        
        if quality_metrics["sharpness"] < 100:
            recommendations.append("📷 Réduire le flou (stabilisation, mise au point)")
        
        if quality_metrics["brightness"] < 50:
            recommendations.append("💡 Augmenter l'éclairage de la scène")
        elif quality_metrics["brightness"] > 200:
            recommendations.append("☀️ Réduire la surexposition")
        
        if quality_metrics["contrast"] < 30:
            recommendations.append("🔆 Améliorer le contraste de l'image")
    
    # Recommandations seuils détection
    conf_30 = detection_results.get("confidence_0.3", {})
    conf_70 = detection_results.get("confidence_0.7", {})
    
    if conf_30.get("objects_count", 0) > conf_70.get("objects_count", 0) * 2:
        recommendations.append("⚖️ Considérer un seuil de confiance plus élevé (0.6-0.7)")
    
    if conf_70.get("objects_count", 0) == 0 and conf_30.get("objects_count", 0) > 0:
        recommendations.append("📉 Considérer un seuil de confiance plus bas (0.4-0.5)")
    
    # Recommandations spatiales
    if spatial_analysis["coverage_percent"] > 50:
        recommendations.append("📦 Scène dense - considérer un NMS plus agressif")
    
    if spatial_analysis["density"] < 0.1:
        recommendations.append("🔍 Scène sparse - vérifier si tous les objets sont détectés")
    
    # Zone analysis
    zones = spatial_analysis.get("zones", {})
    max_zone = max(zones.values()) if zones else 0
    if max_zone > len(zones) * 0.7:  # Plus de 70% dans une zone
        recommendations.append("⚖️ Objets concentrés dans une zone - vérifier l'angle de caméra")
    
    return recommendations if recommendations else ["✅ Configuration optimale détectée"]