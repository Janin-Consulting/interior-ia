
import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, List
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales pour stocker les modèles et processeurs
depth_image_processor = None
depth_model = None

def setup() -> None:
    """Initialise les modèles d'estimation de profondeur
    
    Cette fonction charge le modèle Depth-Anything pour l'estimation de profondeur.
    Elle gère également les optimisations comme xformers si disponible.
    """
    global depth_image_processor, depth_model
    
    logger.info("Initialisation du modèle d'estimation de profondeur...")
    
    # Chargement du processeur d'image et du modèle pour l'estimation de profondeur
    depth_image_processor = AutoImageProcessor.from_pretrained(
        "LiheYoung/depth-anything-large-hf", 
        torch_dtype=torch.float16
    )
    
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "LiheYoung/depth-anything-large-hf", 
        torch_dtype=torch.float16
    )
    
    # Déplacer le modèle sur GPU si disponible
    if torch.cuda.is_available():
        depth_model = depth_model.to("cuda")
        
        # Tenter d'activer les optimisations xformers si disponibles
        try:
            logger.info("Tentative d'activation des optimisations xformers pour l'estimation de profondeur...")
            # Note: Depth-Anything ne prend pas directement en charge xformers, mais nous ajoutons
            # cette structure au cas où une future version le prendrait en charge
            if hasattr(depth_model, 'enable_xformers_memory_efficient_attention'):
                depth_model.enable_xformers_memory_efficient_attention()
                logger.info("Optimisations xformers activées avec succès pour l'estimation de profondeur")
        except (ModuleNotFoundError, ImportError):
            logger.warning("xformers n'est pas disponible, utilisation de l'attention standard pour l'estimation de profondeur")
        except Exception as e:
            logger.warning(f"Impossible d'activer xformers pour l'estimation de profondeur: {str(e)}")
    else:
        logger.warning("CUDA n'est pas disponible. L'inférence sera beaucoup plus lente sur CPU.")

@torch.inference_mode()
@torch.autocast("cuda")
def get_depth_map(image: Image.Image) -> Image.Image:
    """Génère une carte de profondeur à partir d'une image
    
    Args:
        image (PIL.Image.Image): L'image d'entrée pour laquelle générer la carte de profondeur
        
    Returns:
        PIL.Image.Image: La carte de profondeur générée, convertie en image RGB
    """
    global depth_image_processor, depth_model
    
    # Vérifier si les modèles sont chargés
    if depth_image_processor is None or depth_model is None:
        logger.info("Chargement des modèles d'estimation de profondeur...")
        setup()
    
    # S'assurer que l'image est dans le format RGB requis
    if not isinstance(image, Image.Image):
        logger.warning("L'entrée n'est pas une image PIL, tentative de conversion")
        try:
            image = Image.fromarray(np.array(image))
        except Exception as e:
            logger.error(f"Erreur lors de la conversion de l'image: {str(e)}")
            raise
    
    # S'assurer que l'image est en mode RGB
    if image.mode != "RGB":
        logger.info(f"Conversion de l'image du mode {image.mode} à RGB")
        image = image.convert("RGB")
    
    try:
        # Prétraitement de l'image
        width, height = image.size
        inputs = depth_image_processor(images=image, return_tensors="pt")
        
        # Déplacer les entrées sur le même appareil que le modèle
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Obtenir la carte de profondeur
        with torch.no_grad():
            depth_map = depth_model(**inputs).predicted_depth
        
        # Redimensionner la carte de profondeur aux dimensions de l'image d'origine
        depth_map = F.interpolate(
            depth_map.unsqueeze(1).float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Normaliser la carte de profondeur
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Convertir en RGB en répétant la carte sur 3 canaux
        depth_rgb = torch.cat([depth_map] * 3, dim=1)
        
        # Convertir en image PIL
        depth_image = depth_rgb.permute(0, 2, 3, 1).cpu().numpy()[0]
        depth_image = Image.fromarray((depth_image * 255.0).clip(0, 255).astype(np.uint8))
        
        return depth_image
    
    except Exception as e:
        logger.error(f"Erreur lors de l'estimation de la profondeur: {str(e)}")
        # Afficher plus d'informations sur l'image pour le débogage
        logger.error(f"Format de l'image: {image.format}, Mode: {image.mode}, Taille: {image.size}")
        raise

def enhance_depth(depth_image: Image.Image, gamma: float = 1.0, contrast: float = 1.0) -> Image.Image:
    """Améliore la carte de profondeur pour une meilleure utilisation avec ControlNet
    
    Args:
        depth_image (PIL.Image.Image): La carte de profondeur générée
        gamma (float): Facteur gamma pour améliorer certaines zones (>1 accentue les zones sombres)
        contrast (float): Facteur de contraste pour améliorer la différence entre zones
        
    Returns:
        PIL.Image.Image: La carte de profondeur améliorée
    """
    # Convertir en tableau numpy
    depth_np = np.array(depth_image).astype(np.float32) / 255.0
    
    # Appliquer la correction gamma
    if gamma != 1.0:
        depth_np = np.power(depth_np, gamma)
    
    # Appliquer l'ajustement de contraste
    if contrast != 1.0:
        mean_val = np.mean(depth_np)
        depth_np = (depth_np - mean_val) * contrast + mean_val
    
    # Recadrer les valeurs et convertir en image
    depth_np = np.clip(depth_np * 255.0, 0, 255).astype(np.uint8)
    enhanced_depth = Image.fromarray(depth_np)
    
    return enhanced_depth

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Générer une carte de profondeur à partir d\'une image')
    parser.add_argument('--image', type=str, required=True, help='Chemin de l\'image d\'entrée')
    parser.add_argument('--output', type=str, default="depth_map.png", help='Chemin du fichier de sortie')
    parser.add_argument('--gamma', type=float, default=1.0, help='Correction gamma (>1 accentue les zones sombres)')
    parser.add_argument('--contrast', type=float, default=1.0, help='Ajustement du contraste')
    
    args = parser.parse_args()
    
    # Initialiser les modèles
    setup()
    
    # Charger l'image
    image_path = Path(args.image)
    logger.info(f"Traitement de l'image: {args.image}")
    image = Image.open(str(image_path))
    
    # Générer la carte de profondeur
    depth_map = get_depth_map(image)
    
    # Améliorer la carte de profondeur si spécifié
    if args.gamma != 1.0 or args.contrast != 1.0:
        depth_map = enhance_depth(depth_map, args.gamma, args.contrast)
    
    # Sauvegarder la carte de profondeur
    output_path = args.output
    depth_map.save(output_path)
    
    logger.info(f"Carte de profondeur sauvegardée avec succès à: {output_path}")