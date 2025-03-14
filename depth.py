import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, List
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTForDepthEstimation, DPTImageProcessor
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
    
    Cette fonction charge le modèle Depth-Anything pour l'estimation de profondeur.
    Elle gère également les optimisations comme xformers si disponible.
    """
    global depth_image_processor, depth_model
    
    logger.info("Initialisation du modèle d'estimation de profondeur...")
    
    # Essayer d'abord le modèle depth-anything (haute qualité)
    try:
        # Chargement du processeur d'image et du modèle pour l'estimation de profondeur
        depth_image_processor = AutoImageProcessor.from_pretrained(
            "LiheYoung/depth-anything-large-hf", 
            torch_dtype=torch.float16
        )
        
        depth_model = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-large-hf", 
            torch_dtype=torch.float16
        )
        
        logger.info("Modèle depth-anything-large chargé avec succès")
    except Exception as e:
        logger.warning(f"Erreur lors du chargement de depth-anything: {str(e)}")
        
        # Essayer un modèle alternatif plus léger et plus largement disponible
        try:
            logger.info("Tentative de chargement du modèle MiDaS comme alternative...")
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            
            # Utiliser MiDaS (plus commun et présent dans plus d'environnements)
            depth_image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            
            logger.info("Modèle MiDaS chargé avec succès comme alternative")
        except Exception as e2:
            logger.error(f"Impossible de charger un modèle de profondeur alternatif: {str(e2)}")
            raise RuntimeError("depth_anything") from e2
    
    # Déplacer le modèle sur GPU si disponible
    if torch.cuda.is_available():
        depth_model = depth_model.to("cuda")
        
    # Essayer d'activer xformers pour les optimisations de mémoire (si disponible)
    try:
        import xformers
        logger.info("xformers détecté, activation des optimisations de mémoire")
        # Vérifier si le modèle supporte xformers
        if hasattr(depth_model, 'enable_xformers_memory_efficient_attention'):
            depth_model.enable_xformers_memory_efficient_attention()
        else:
            logger.info("Ce modèle ne prend pas en charge les optimisations xformers")
    except (ImportError, AttributeError) as e:
        logger.info(f"xformers non disponible ({str(e)})")
        # Utiliser l'attention slicing comme alternative si xformers n'est pas disponible
        # et si le modèle le supporte
        if hasattr(depth_model, 'enable_attention_slicing'):
            logger.info("Activation de l'attention slicing comme alternative")
            depth_model.enable_attention_slicing()
        else:
            logger.info("Ce modèle ne prend pas en charge l'attention slicing")

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
        
        # Déplacer sur GPU si disponible
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Obtenir la carte de profondeur du modèle
        with torch.no_grad():
            # Détection du type de modèle et adaptation en conséquence
            if "DPT" in depth_model.__class__.__name__:
                # Cas du modèle MiDaS
                depth_map = depth_model(**inputs).predicted_depth
                
                # Normalisation spécifique pour MiDaS
                depth_min, depth_max = torch.min(depth_map), torch.max(depth_map)
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                # Cas par défaut (depth-anything)
                depth_map = depth_model(**inputs).predicted_depth
                
                # Normalisation standard
                depth_min, depth_max = torch.min(depth_map), torch.max(depth_map)
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Redimensionner la carte de profondeur aux dimensions de l'image d'origine
        depth_map = F.interpolate(
            depth_map.unsqueeze(1).float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
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