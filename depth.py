from PIL import Image
import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Tuple, Dict, Any
import time
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_DEPTH_CONFIG = {
    "model_id": "LiheYoung/depth-anything-large-hf",
    "use_fast": False,
    "interpolation_mode": "bicubic",
    "align_corners": False,
    "normalize_depth": True
}

# Définition du périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilisation du périphérique pour l'estimation de profondeur: {device}")

def load_depth_model() -> Tuple[AutoImageProcessor, AutoModelForDepthEstimation]:
    """
    Charge le modèle d'estimation de profondeur.
    
    Returns:
        Un tuple contenant le processeur et le modèle d'estimation de profondeur.
    """
    start_time = time.time()
    logger.info(f"Chargement du modèle d'estimation de profondeur depuis {DEFAULT_DEPTH_CONFIG['model_id']}")
    
    depth_processor = AutoImageProcessor.from_pretrained(
        DEFAULT_DEPTH_CONFIG["model_id"], 
        use_fast=DEFAULT_DEPTH_CONFIG["use_fast"]
    )
    
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEFAULT_DEPTH_CONFIG["model_id"])
    depth_model = depth_model.to(device)
    
    if device.type == 'cuda':
        logger.info("GPU détecté, conversion du modèle en demi-précision pour optimisation")
        depth_model = depth_model.half()
    
    depth_model.eval()
    
    loading_time = time.time() - start_time
    logger.info(f"Modèle d'estimation de profondeur chargé en {loading_time:.2f} secondes")
    
    return depth_processor, depth_model

# Chargement du modèle et du processeur
depth_processor, depth_model = load_depth_model()

def get_depth_image(
    image: Image.Image, 
    config: Dict[str, Any] = None
) -> Image.Image:
    """
    Génère une carte de profondeur à partir de l'image.
    
    Args:
        image: L'image à traiter.
        config: Configuration personnalisée (facultatif)
        
    Returns:
        La carte de profondeur sous forme d'image en niveau de gris.
    """
    # Fusionner la configuration par défaut avec la configuration personnalisée
    cfg = DEFAULT_DEPTH_CONFIG.copy()
    if config:
        cfg.update(config)
        logger.info(f"Configuration personnalisée appliquée pour l'estimation de profondeur: {config}")
    
    start_time = time.time()
    logger.info(f"Début de l'estimation de profondeur pour une image de taille {image.size}")
    
    original_size = image.size
    
    # Préparation des entrées pour le modèle
    prep_start = time.time()
    image_to_depth = depth_processor(images=image, return_tensors="pt").to(device)
    prep_time = time.time() - prep_start
    logger.debug(f"Préparation des entrées terminée en {prep_time:.2f} secondes")
    
    # Inférence du modèle
    inference_start = time.time()
    with torch.no_grad():
        depth_map = depth_model(**image_to_depth).predicted_depth
    inference_time = time.time() - inference_start
    logger.debug(f"Inférence du modèle terminée en {inference_time:.2f} secondes")
    
    # Redimensionnement à la taille d'origine
    resize_start = time.time()
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1).float(),
        size=(original_size[1], original_size[0]),
        mode=cfg["interpolation_mode"],
        align_corners=cfg["align_corners"],
    )
    resize_time = time.time() - resize_start
    logger.debug(f"Redimensionnement de la carte de profondeur terminé en {resize_time:.2f} secondes")
    
    # Normalisation de la carte de profondeur
    if cfg["normalize_depth"]:
        norm_start = time.time()
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        norm_time = time.time() - norm_start
        logger.debug(f"Normalisation de la carte de profondeur terminée en {norm_time:.2f} secondes")
    
    # Convertir directement en image en niveau de gris
    conversion_start = time.time()
    depth_np = depth_map.squeeze().cpu().numpy()
    depth_grayscale = (depth_np * 255.0).clip(0, 255).astype(np.uint8)
    conversion_time = time.time() - conversion_start
    logger.debug(f"Conversion en image en niveau de gris terminée en {conversion_time:.2f} secondes")
    
    total_time = time.time() - start_time
    logger.info(f"Estimation de profondeur terminée en {total_time:.2f} secondes")
    
    return Image.fromarray(depth_grayscale, mode='L')

def enhance_depth_map(
    depth_image: Image.Image,
    apply_denoise: bool = True,
    apply_contrast: bool = True,
    denoise_strength: int = 10,
    contrast_factor: float = 1.5
) -> Image.Image:
    """
    Améliore la carte de profondeur en appliquant des filtres de débruitage et d'amélioration de contraste.
    
    Args:
        depth_image: Image de profondeur en niveau de gris
        apply_denoise: Appliquer un filtre de débruitage
        apply_contrast: Améliorer le contraste
        denoise_strength: Force du débruitage (1-30)
        contrast_factor: Facteur d'amélioration du contraste
        
    Returns:
        Carte de profondeur améliorée
    """
    start_time = time.time()
    logger.info(f"Début de l'amélioration de la carte de profondeur (denoise={apply_denoise}, contrast={apply_contrast})")
    
    result = np.array(depth_image)
    
    if apply_denoise:
        denoise_start = time.time()
        # Utilisation d'un filtre bilatéral pour préserver les bords tout en réduisant le bruit
        result = cv2.bilateralFilter(result, d=denoise_strength, sigmaColor=50, sigmaSpace=50)
        denoise_time = time.time() - denoise_start
        logger.debug(f"Débruitage terminé en {denoise_time:.2f} secondes")
    
    if apply_contrast:
        contrast_start = time.time()
        # Amélioration du contraste par égalisation d'histogramme adaptative
        clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
        result = clahe.apply(result)
        contrast_time = time.time() - contrast_start
        logger.debug(f"Amélioration du contraste terminée en {contrast_time:.2f} secondes")
    
    total_time = time.time() - start_time
    logger.info(f"Amélioration de la carte de profondeur terminée en {total_time:.2f} secondes")
    
    return Image.fromarray(result, mode='L')