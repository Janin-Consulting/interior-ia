# python upscale.py --image result.png --output result-upscale.png --type interior

import numpy as np
import torch
import os
import time
import logging
import math
import gc
import cv2

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

# Configuration du logger pour le suivi des opérations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration par défaut pour l'upscaling
DEFAULT_CONFIG = {
    "model_id": "stabilityai/stable-diffusion-x4-upscaler",  # Modèle d'upscaling à utiliser
    "prompt_strength": 0.75,                                 # Force du prompt dans le processus (0-1), augmentée pour plus d'impact
    "num_inference_steps": 20,                               # Nombre d'étapes de diffusion
    "noise_level": 15,                                       # Niveau de bruit réduit pour préserver les détails d'origine
    "guidance_scale": 9.0,                                   # Échelle de guidance augmentée pour renforcer l'influence du prompt
    "use_half_precision": True,                              # Utilisation de la précision FP16 pour économiser la mémoire
    "max_size": 512,                                         # Taille maximale d'image pour l'upscaling direct, réduite pour limiter la mémoire 
    "tile_size": 192,                                        # Taille des tuiles pour le traitement par morceaux, réduite pour limiter la mémoire
    "tile_overlap": 64,                                      # Chevauchement des tuiles, réduit pour limiter la mémoire
    "use_tiling": True,                                      # Activer le traitement par tuiles pour les grandes images
    "auto_resize": True                                      # Redimensionner automatiquement avant l'upscaling
}

# Prompts spécifiques pour différents types d'images
PROMPT_TEMPLATES = {
    "general": "image ultra haute définition, détails fins, netteté parfaite, éclairage naturel, texture réaliste, 8k",
    "interior": "intérieur détaillé, textures de matériaux réalistes, éclairage naturel, ombres douces, détails architecturaux précis, 8k",
    "portrait": "portrait haute définition, traits du visage nets, peau naturelle, détails des yeux, texture des cheveux réaliste, éclairage studio professionnel, 8k",
    "landscape": "paysage détaillé, profondeur de champ, détails des textures naturelles, lumière atmosphérique, couleurs vibrantes, clarté exceptionnelle, 8k"
}

def initialize_upscaler(config=None, device=None):
    """
    Initialise le modèle d'upscaling Stable Diffusion X4.
    
    Args:
        config: Configuration personnalisée (facultative)
        device: Périphérique à utiliser pour l'inférence ("cuda", "cpu", etc.)
        
    Returns:
        Le pipeline d'upscaling initialisé
    """
    # Fusionner la configuration par défaut avec la configuration personnalisée
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Détecter automatiquement le périphérique si non spécifié
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    start_time = time.time()
    logger.info(f"Initialisation du modèle d'upscaling {cfg['model_id']} sur {device}")
    
    try:
        # Préparation du type de tenseur en fonction de la configuration
        dtype = torch.float16 if cfg["use_half_precision"] and device == "cuda" else torch.float32
        
        # Initialisation du pipeline standard PyTorch
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            cfg["model_id"],
            torch_dtype=dtype
        )
        
        # Déplacement du modèle vers le périphérique approprié
        pipeline = pipeline.to(device)
        
        # Optimisations supplémentaires pour l'inférence
        if device == "cuda":
            # Activer les optimisations mémoire si disponibles
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing("max")
            
            # Activer d'autres optimisations de mémoire pour les grands modèles
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                
            # Activer le tiling de VAE pour réduire la consommation de mémoire
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
            
            # Activer la libération de mémoire CUDA à chaque étape
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                
            # Force la libération de mémoire après initialisation
            torch.cuda.empty_cache()
            gc.collect()
        
        init_time = time.time() - start_time
        logger.info(f"Modèle d'upscaling initialisé en {init_time:.2f} secondes")
        
        return pipeline
    
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
        raise RuntimeError(f"Erreur lors de l'initialisation du modèle: {str(e)}")

def create_tile_masks(tile_size, overlap, dtype=np.float32):
    """
    Crée des masques de fondu pour le mélange des tuiles avec une approche vectorisée.
    
    Args:
        tile_size: Taille des tuiles
        overlap: Zone de chevauchement entre les tuiles
        dtype: Type de données NumPy à utiliser
    
    Returns:
        Tuple de masques pour les 4 bords (haut, bas, gauche, droite)
    """
    # Création de gradients linéaires pour les zones de chevauchement
    mask_x = np.linspace(0, 1, overlap, dtype=dtype)
    mask_y = np.linspace(0, 1, overlap, dtype=dtype)
    
    # Création des masques pour chaque bord
    top_mask = np.ones((tile_size, tile_size), dtype=dtype)
    bottom_mask = np.ones((tile_size, tile_size), dtype=dtype)
    left_mask = np.ones((tile_size, tile_size), dtype=dtype)
    right_mask = np.ones((tile_size, tile_size), dtype=dtype)
    
    # Application des gradients avec des opérations de broadcasting correctes
    # Création des masques 2D pour chaque bord
    top_gradient = np.broadcast_to(mask_y[:, np.newaxis], (overlap, tile_size))
    bottom_gradient = np.broadcast_to(mask_y[::-1, np.newaxis], (overlap, tile_size))
    left_gradient = np.broadcast_to(mask_x[np.newaxis, :], (tile_size, overlap))
    right_gradient = np.broadcast_to(mask_x[::-1][np.newaxis, :], (tile_size, overlap))
    
    # Application des gradients aux masques
    top_mask[:overlap, :] = top_gradient
    bottom_mask[-overlap:, :] = bottom_gradient
    left_mask[:, :overlap] = left_gradient
    right_mask[:, -overlap:] = right_gradient
    
    return top_mask, bottom_mask, left_mask, right_mask

def upscale_image_with_tiling(input_image, prompt, pipeline, config=None):
    """
    Applique l'upscaling avec traitement par tuiles pour les grandes images.
    Optimisé avec la vectorisation NumPy pour améliorer les performances.
    
    Args:
        input_image: Image PIL à traiter
        prompt: Description textuelle pour guider l'upscaling
        pipeline: Pipeline d'upscaling initialisé
        config: Configuration personnalisée
        
    Returns:
        Image PIL upscalée
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    width, height = input_image.size
    tile_size = cfg["tile_size"]
    tile_overlap = cfg["tile_overlap"]
    
    # Conversion de l'image PIL en tableau NumPy pour un traitement vectorisé
    input_array = np.array(input_image)
    
    # Calcul du nombre de tuiles en largeur et hauteur
    num_tiles_x = math.ceil(width / (tile_size - tile_overlap))
    num_tiles_y = math.ceil(height / (tile_size - tile_overlap))
    
    # Calcul de la taille finale après upscaling
    final_width = width * 4
    final_height = height * 4
    
    # Création d'un tableau NumPy pour stocker le résultat au lieu d'une image PIL
    # Cela évite les conversions répétées entre PIL et NumPy
    result_array = np.zeros((final_height, final_width, 3), dtype=np.uint8)
    
    # Création d'un tableau pour suivre les zones déjà remplies (pour le mélange)
    coverage = np.zeros((final_height, final_width), dtype=np.float32)
    
    logger.info(f"Traitement par tuiles: {num_tiles_x}x{num_tiles_y} tuiles de {tile_size}x{tile_size} pixels")
    
    # Pré-calcul des masques de fusion pour éviter de les recalculer à chaque tuile
    top_mask, bottom_mask, left_mask, right_mask = create_tile_masks(tile_size, tile_overlap)
    
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calcul des coordonnées de la tuile
            x_start = x * (tile_size - tile_overlap)
            y_start = y * (tile_size - tile_overlap)
            
            # Ajustement pour la dernière tuile
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            
            # Ajustement pour garantir que la taille minimale est respectée
            x_start = max(0, min(x_start, width - tile_size))
            y_start = max(0, min(y_start, height - tile_size))
            
            logger.info(f"Traitement de la tuile ({x+1}/{num_tiles_x}, {y+1}/{num_tiles_y}): [{x_start}:{x_end}, {y_start}:{y_end}]")
            
            # Extraction de la tuile avec NumPy au lieu de PIL.crop
            # Optimisation: extraction directe depuis le tableau NumPy
            tile_array = input_array[y_start:y_end, x_start:x_end]
            tile = Image.fromarray(tile_array)
            
            try:
                # Libérer la mémoire avant traitement de chaque tuile
                torch.cuda.empty_cache()
                gc.collect()
                
                # Upscaling de la tuile
                output_tile = pipeline(
                    prompt=prompt,
                    image=tile,
                    num_inference_steps=cfg["num_inference_steps"],
                    guidance_scale=cfg["guidance_scale"],
                    noise_level=cfg["noise_level"]
                ).images[0]
                
                # Conversion de la tuile de sortie en tableau NumPy
                output_tile_array = np.array(output_tile)
                
                # Calcul des coordonnées dans l'image de sortie
                out_x = x_start * 4
                out_y = y_start * 4
                out_width = output_tile_array.shape[1]
                out_height = output_tile_array.shape[0]
                
                # Calcul des masques composites pour cette tuile
                composite_mask = np.ones((out_height, out_width), dtype=np.float32)
                
                # Appliquer les masques de bord appropriés selon la position de la tuile
                if x > 0:  # Si ce n'est pas la première colonne
                    left_mask_resized = cv2.resize(left_mask, (out_width, out_height))
                    composite_mask *= left_mask_resized
                if x < num_tiles_x - 1:  # Si ce n'est pas la dernière colonne
                    right_mask_resized = cv2.resize(right_mask, (out_width, out_height))
                    composite_mask *= right_mask_resized
                if y > 0:  # Si ce n'est pas la première ligne
                    top_mask_resized = cv2.resize(top_mask, (out_width, out_height))
                    composite_mask *= top_mask_resized
                if y < num_tiles_y - 1:  # Si ce n'est pas la dernière ligne
                    bottom_mask_resized = cv2.resize(bottom_mask, (out_width, out_height))
                    composite_mask *= bottom_mask_resized
                
                # S'assurer que la zone de destination est valide
                out_y_end = min(final_height, out_y + out_height)
                out_x_end = min(final_width, out_x + out_width)
                out_height = out_y_end - out_y
                out_width = out_x_end - out_x
                
                # Appliquer la tuile à l'image résultat avec le masque
                # Vectorisation: opération matricielle au lieu de boucles
                composite_mask = composite_mask[:out_height, :out_width, np.newaxis]
                
                # Mettre à jour la zone de couverture
                current_coverage = coverage[out_y:out_y_end, out_x:out_x_end]
                
                # Normaliser les poids pour éviter la surexposition
                total_weight = current_coverage[:, :, np.newaxis] + composite_mask
                total_weight = np.maximum(total_weight, 1e-6)  # Éviter la division par zéro
                
                # Calcul du mélange pondéré
                result_array[out_y:out_y_end, out_x:out_x_end] = (
                    (result_array[out_y:out_y_end, out_x:out_x_end] * current_coverage[:, :, np.newaxis]) + 
                    (output_tile_array[:out_height, :out_width] * composite_mask)
                ) / total_weight
                
                # Mettre à jour la couverture
                coverage[out_y:out_y_end, out_x:out_x_end] += composite_mask[:, :, 0]
                
                # Libération de la mémoire
                del output_tile, output_tile_array, composite_mask, tile, tile_array
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la tuile ({x+1}/{num_tiles_x}, {y+1}/{num_tiles_y}): {str(e)}")
                # Continuer avec la tuile suivante plutôt que d'abandonner tout le processus
                continue
    
    # Conversion du tableau résultat en image PIL
    result_img = Image.fromarray(np.uint8(result_array))
    
    # Libération de la mémoire des grandes matrices
    del result_array, coverage, input_array
    gc.collect()
    
    return result_img

def resize_for_upscaling(image, max_size=DEFAULT_CONFIG["max_size"]):
    """
    Redimensionne une image pour qu'elle soit adaptée à l'upscaling.
    Optimisé avec la vectorisation NumPy.
    
    Args:
        image: Image PIL à redimensionner
        max_size: Taille maximale (largeur ou hauteur) pour l'upscaling
        
    Returns:
        Tuple (image redimensionnée, facteur d'échelle)
    """
    width, height = image.size
    # Calcul du ratio pour le redimensionnement
    ratio = min(max_size / width, max_size / height)
    
    # Si l'image est déjà plus petite que max_size, ne pas la redimensionner
    if ratio >= 1:
        return image, 1.0
    
    # Calcul des nouvelles dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    logger.info(f"Redimensionnement préalable: {width}x{height} -> {new_width}x{new_height}")
    
    # Utilisation de NumPy pour le redimensionnement
    # Conversion en tableau NumPy
    img_array = np.array(image)
    
    # Utilisation de OpenCV pour un redimensionnement plus rapide que PIL
    import cv2
    resized_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Conversion du tableau NumPy en image PIL
    resized_image = Image.fromarray(resized_array)
    
    # Libération de la mémoire
    del img_array, resized_array
    gc.collect()
    
    return resized_image, ratio

def upscale_image(input_image, prompt="", pipeline=None, config=None, device=None, image_type="general"):
    """
    Applique l'upscaling 4x sur une image à l'aide du modèle Stable Diffusion X4 Upscaler.
    
    Args:
        input_image: Image à améliorer (PIL.Image ou chemin d'accès)
        prompt: Description textuelle pour guider l'upscaling (facultatif)
        pipeline: Pipeline d'upscaling préinitialisé (facultatif)
        config: Configuration personnalisée (facultative)
        device: Périphérique à utiliser pour l'inférence (facultatif)
        image_type: Type d'image pour utiliser un prompt spécifique ("general", "interior", "portrait", "landscape")
    
    Returns:
        L'image améliorée avec une résolution 4x supérieure
    """
    # Fusionner la configuration par défaut avec la configuration personnalisée
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Gestion des différents types d'entrée pour l'image
    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            logger.error(f"Le fichier image {input_image} n'existe pas")
            raise FileNotFoundError(f"Le fichier image {input_image} n'existe pas")
        
        # Chargement de l'image depuis le chemin d'accès
        logger.info(f"Chargement de l'image depuis {input_image}")
        input_image = Image.open(input_image).convert("RGB")
    
    elif not isinstance(input_image, Image.Image):
        # Conversion d'un tableau numpy en image PIL si nécessaire
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        else:
            logger.error(f"Type d'image non pris en charge: {type(input_image)}")
            raise TypeError(f"Type d'image non pris en charge: {type(input_image)}")
    
    # Vérification des dimensions de l'image
    width, height = input_image.size
    logger.info(f"Dimensions de l'image d'entrée: {width}x{height}")
    
    # Redimensionnement automatique si l'image est trop grande
    original_size = (width, height)
    ratio = 1.0
    
    if cfg["auto_resize"] and (width > cfg["max_size"] or height > cfg["max_size"]):
        input_image, ratio = resize_for_upscaling(input_image, cfg["max_size"])
        width, height = input_image.size
    
    # Initialisation du pipeline si non fourni
    if pipeline is None:
        pipeline = initialize_upscaler(config=cfg, device=device)
    
    # Création d'un prompt par défaut si non spécifié ou enrichissement du prompt existant
    if not prompt:
        prompt = PROMPT_TEMPLATES[image_type]
    else:
        # Enrichir le prompt utilisateur avec des termes de qualité supplémentaires s'ils ne sont pas déjà présents
        quality_terms = ["haute définition", "8k", "détaillé", "net"]
        if not any(term in prompt.lower() for term in quality_terms):
            prompt = f"{prompt}, {PROMPT_TEMPLATES['general']}"
    
    logger.info(f"Utilisation du prompt: '{prompt}'")
    
    try:
        start_time = time.time()
        logger.info(f"Début du processus d'upscaling avec le prompt: '{prompt}'")
        
        # Toujours forcer l'utilisation du tiling pour réduire la consommation de mémoire
        use_tiling = True
        
        if use_tiling:
            # Traitement par tuiles pour les grandes images
            output_image = upscale_image_with_tiling(input_image, prompt, pipeline, cfg)
        else:
            # Application de l'upscaling standard pour les petites images
            output_image = pipeline(
                prompt=prompt,
                image=input_image,
                num_inference_steps=cfg["num_inference_steps"],
                guidance_scale=cfg["guidance_scale"],
                noise_level=cfg["noise_level"]
            ).images[0]
        
        process_time = time.time() - start_time
        new_width, new_height = output_image.size
        
        logger.info(
            f"Upscaling terminé en {process_time:.2f} secondes. "
            f"Nouvelles dimensions: {new_width}x{new_height}"
        )
        
        # Si l'image a été redimensionnée avant le traitement, mentionner le ratio
        if ratio < 1.0:
            logger.info(f"Note: L'image originale ({original_size[0]}x{original_size[1]}) a été redimensionnée par un facteur de {ratio:.2f} avant l'upscaling.")
        
        return output_image
    
    except Exception as e:
        logger.error(f"Erreur lors de l'upscaling: {str(e)}")
        # Nettoyer la mémoire en cas d'erreur
        if pipeline is not None:
            del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(f"Erreur lors de l'upscaling: {str(e)}")

def get_upscaled_image(input_image_path, prompt="", config=None, device=None, image_type="general"):
    """
    Fonction principale pour l'upscaling d'une image avec Stable Diffusion X4.
    Gère le chargement, l'upscaling et le retour de l'image améliorée.
    
    Args:
        input_image_path: Chemin d'accès à l'image à améliorer
        prompt: Description textuelle pour guider l'upscaling (facultatif)
        config: Configuration personnalisée (facultative)
        device: Périphérique à utiliser pour l'inférence (facultatif)
        image_type: Type d'image pour utiliser un prompt spécifique ("general", "interior", "portrait", "landscape")
    
    Returns:
        L'image améliorée avec une résolution 4x supérieure
    
    Raises:
        FileNotFoundError: Si le fichier image n'existe pas
        RuntimeError: Si une erreur survient pendant le traitement
    """
    logger.info(f"Traitement de l'image pour upscaling: {input_image_path}")
    
    # Initialisation du pipeline (une seule fois)
    pipeline = initialize_upscaler(config=config, device=device)
    
    # Application de l'upscaling
    return upscale_image(
        input_image=input_image_path,
        prompt=prompt,
        pipeline=pipeline,
        config=config,
        device=device,
        image_type=image_type
    )

if __name__ == "__main__":
    # Exemple d'utilisation
    import argparse
    
    parser = argparse.ArgumentParser(description="Upscaling d'images avec Stable Diffusion X4")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image à améliorer")
    parser.add_argument("--output", type=str, help="Chemin pour sauvegarder l'image améliorée")
    parser.add_argument("--prompt", type=str, default="", help="Description textuelle pour guider l'upscaling")
    parser.add_argument("--device", type=str, default="cuda", help="Périphérique à utiliser (cuda/cpu)")
    parser.add_argument("--max_size", type=int, help="Taille maximale de l'image (largeur ou hauteur) avant upscaling")
    parser.add_argument("--tile_size", type=int, help="Taille des tuiles pour le traitement")
    parser.add_argument("--no_tiling", action="store_true", help="Désactiver le traitement par tuiles")
    parser.add_argument("--no_resize", action="store_true", help="Désactiver le redimensionnement automatique")
    parser.add_argument("--type", type=str, default="general", choices=["general", "interior", "portrait", "landscape"], 
                       help="Type d'image pour appliquer un prompt spécifique")
    
    args = parser.parse_args()
    
    # Préparation de la configuration personnalisée
    config = {}
    if args.max_size:
        config["max_size"] = args.max_size
    if args.tile_size:
        config["tile_size"] = args.tile_size
    if args.no_tiling:
        config["use_tiling"] = False
    if args.no_resize:
        config["auto_resize"] = False
    
    # Définir le chemin de sortie par défaut si non spécifié
    if not args.output:
        filename = os.path.basename(args.image)
        name, ext = os.path.splitext(filename)
        args.output = os.path.join(os.path.dirname(args.image), f"{name}_upscaled{ext}")
    
    # Appliquer l'upscaling
    result = get_upscaled_image(args.image, prompt=args.prompt, device=args.device, config=config, image_type=args.type)
    
    # Sauvegarder le résultat
    result.save(args.output)
    logger.info(f"Image améliorée sauvegardée sous: {args.output}")