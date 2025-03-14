import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, List
from transformers import DPTForDepthEstimation, DPTImageProcessor
import torch.nn.functional as F

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales pour stocker les modèles et processeurs
depth_image_processor = None
depth_model = None
depth_min = 0.0
depth_max = 1.0

def setup() -> None:
    """Initialise les modèles d'estimation de profondeur
    
    Cette fonction charge le modèle MiDaS pour l'estimation de profondeur.
    """
    global depth_image_processor, depth_model
    
    logger.info("Initialisation du modèle d'estimation de profondeur...")
    
    # Utiliser directement MiDaS comme modèle principal, puisqu'il fonctionne bien
    try:
        # Utiliser MiDaS qui est plus stable et plus largement disponible
        depth_image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        
        logger.info("Modèle MiDaS chargé avec succès")
    except Exception as e:
        logger.error(f"Impossible de charger le modèle MiDaS: {str(e)}")
        raise RuntimeError("Aucun modèle de profondeur disponible") from e
    
    # Déplacer le modèle sur GPU si disponible
    if torch.cuda.is_available():
        depth_model = depth_model.to("cuda")

@torch.inference_mode()
@torch.autocast("cuda")
def get_depth_map(image: Image.Image) -> Image.Image:
    """Génère une carte de profondeur à partir d'une image
    
    Args:
        image (PIL.Image.Image): L'image d'entrée pour laquelle générer la carte de profondeur
        
    Returns:
        PIL.Image.Image: La carte de profondeur générée, convertie en image RGB
    """
    global depth_min, depth_max
    
    # Vérification de l'initialisation
    if depth_model is None or depth_image_processor is None:
        raise RuntimeError("Le module de profondeur n'a pas été initialisé. Appelez setup() d'abord.")
    
    # S'assurer que l'image est dans le format RGB requis
    if not isinstance(image, Image.Image):
        raise TypeError("L'image doit être une instance de PIL.Image.Image")
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    try:
        # Redimensionner l'image si elle est très grande (pour économiser la mémoire GPU)
        width, height = image.size
        max_size = 1024  # Limite de taille raisonnable pour l'estimation de profondeur
        if width > max_size or height > max_size:
            # Calculer le ratio pour redimensionner proportionnellement
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            logger.info(f"Redimensionnement de l'image de {width}x{height} à {new_size[0]}x{new_size[1]} pour l'estimation de profondeur")
            image_for_depth = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            image_for_depth = image
            
        # Prétraitement de l'image pour le modèle de profondeur
        width, height = image_for_depth.size
        inputs = depth_image_processor(images=image_for_depth, return_tensors="pt")
        
        # Déplacer sur GPU si disponible
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Obtenir la carte de profondeur du modèle en adaptant selon le type de modèle
        with torch.no_grad():
            # Pour modèle MiDaS
            depth_map = depth_model(**inputs).predicted_depth
                
            # Normalisation
            depth_min, depth_max = torch.min(depth_map), torch.max(depth_map)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Redimensionner la carte de profondeur aux dimensions de l'image d'origine
        depth_map = F.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convertir en RGB en répétant la carte sur 3 canaux
        depth_rgb = torch.cat([depth_map] * 3, dim=1)
        
        # Convertir en image PIL
        depth_image = depth_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
        depth_image = (depth_image * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_image)
        
        # Redimensionner à la taille de l'image originale si nécessaire
        if depth_image.size != image.size:
            depth_image = depth_image.resize(image.size, Image.Resampling.LANCZOS)
        
        return depth_image
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la carte de profondeur: {str(e)}")
        # Créer une carte de profondeur de secours basique (gradient simple)
        logger.warning("Création d'une carte de profondeur de secours basique (gradient)")
        
        # Créer un dégradé horizontal comme estimation très basique de la profondeur
        # C'est mieux que rien en cas d'échec complet du modèle
        fallback_depth = Image.new("L", image.size, color=0)
        for x in range(image.size[0]):
            # Créer un dégradé horizontal (de gauche à droite)
            value = int(255 * x / image.size[0])
            for y in range(image.size[1]):
                fallback_depth.putpixel((x, y), value)
        
        # Convertir en RGB
        fallback_depth_rgb = Image.merge("RGB", [fallback_depth, fallback_depth, fallback_depth])
        
        return fallback_depth_rgb

def enhance_depth(depth_image: Image.Image, gamma: float = 1.0, contrast: float = 1.0) -> Image.Image:
    """Améliore la carte de profondeur pour une meilleure utilisation avec ControlNet
    
    Args:
        depth_image (PIL.Image.Image): La carte de profondeur à améliorer
        gamma (float, optional): Correction gamma à appliquer. Défaut à 1.0.
        contrast (float, optional): Ajustement de contraste à appliquer. Défaut à 1.0.
        
    Returns:
        PIL.Image.Image: La carte de profondeur améliorée
    """
    # Convertir en array numpy
    depth_np = np.array(depth_image).astype(np.float32) / 255.0
    
    # Appliquer correction gamma
    if gamma != 1.0:
        depth_np = np.power(depth_np, gamma)
    
    # Appliquer ajustement de contraste
    if contrast != 1.0:
        mean_val = np.mean(depth_np)
        depth_np = (depth_np - mean_val) * contrast + mean_val
        depth_np = np.clip(depth_np, 0.0, 1.0)
    
    # Reconvertir en image PIL
    enhanced_depth = (depth_np * 255.0).astype(np.uint8)
    
    return Image.fromarray(enhanced_depth)