# python predict.py --image "empty_room_input/image.png" --output result.png --prompt "living room, modern style" --depth_weight 0.3

import random
import logging
import torch
import numpy as np
import sys

from typing import Tuple, Union, List, Optional
from pathlib import Path
from PIL import Image

# Créer un fichier monkey_patch pour gérer les imports problématiques
from monkey_patch import apply_patches
apply_patches()

# Imports des bibliothèques principales après l'application des monkey patches
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.pipelines.controlnet import StableDiffusionControlNetInpaintPipeline
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from colors import ade20k_palette
from utils import map_colors_to_rgb
from depth import get_depth_map, setup as setup_depth
from controlnet_aux import MLSDdetector

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales pour stocker les modèles et paramètres
pipe = None
control_items = None
additional_quality_suffix = None
seg_image_processor = None
image_segmentor = None
mlsd_processor = None
depth_initialized = False

def filter_items(
    colors_list: Union[List[Tuple[int, int, int]], np.ndarray],
    items_list: Union[List[str], np.ndarray],
    items_to_remove: Union[List[str], np.ndarray],
) -> Tuple[Union[List[Tuple[int, int, int]], np.ndarray], Union[List[str], np.ndarray]]:
    """
    Filtre les éléments et leurs couleurs correspondantes à partir des listes données,
    en excluant les éléments spécifiés.

    Args:
        colors_list: Une liste ou tableau numpy de couleurs correspondant aux éléments.
        items_list: Une liste ou tableau numpy d'éléments.
        items_to_remove: Une liste ou tableau numpy d'éléments à supprimer.

    Returns:
        Un tuple de deux listes ou tableaux numpy: couleurs filtrées et éléments filtrés.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
            
    return filtered_colors, filtered_items

def setup() -> None:
    """Charge le modèle en mémoire pour rendre l'exécution de plusieurs prédictions efficace"""
    global pipe, control_items, additional_quality_suffix, seg_image_processor, image_segmentor, mlsd_processor, depth_initialized
    
    # Initialiser le module de profondeur d'abord
    try:
        setup_depth()
        depth_initialized = True
        logger.info("Module d'estimation de profondeur initialisé avec succès")
    except Exception as e:
        depth_initialized = False
        logger.warning(f"Impossible d'initialiser le module de profondeur: {str(e)}")
        # Ne pas propager l'erreur, nous allons continuer sans profondeur
    
    # Charger les controlnets en fonction de la disponibilité de la profondeur
    if depth_initialized:
        logger.info("Initialisation des 3 ControlNets (segmentation, MLSD, profondeur)")
        controlnet = [
            ControlNetModel.from_pretrained(
                "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            )
        ]
    else:
        logger.info("Initialisation des 2 ControlNets (segmentation, MLSD) sans profondeur")
        controlnet = [
            ControlNetModel.from_pretrained(
                "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
            )
        ]

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V3.0_VAE",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    # Rendre xformers optionnel
    try:
        logger.info("Tentative d'activation des optimisations xformers...")
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Optimisations xformers activées avec succès")
    except (ModuleNotFoundError, ImportError):
        logger.warning("xformers n'est pas disponible, utilisation de l'attention standard")
    except Exception as e:
        logger.warning(f"Impossible d'activer xformers: {str(e)}")
    
    pipe = pipe.to("cuda")

    control_items = [
        "windowpane;window",
        "column;pillar",
        "door;double;door"
    ]
    additional_quality_suffix = "interior design, 4K, high resolution, elegant, tastefully decorated, functional"
    seg_image_processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")
    
    try:
        setup_depth()
        depth_initialized = True
        logger.info("Module d'estimation de profondeur initialisé avec succès")
    except Exception as e:
        depth_initialized = False
        logger.warning(f"Impossible d'initialiser le module de profondeur: {str(e)}")

@torch.inference_mode()
@torch.autocast("cuda")
def segment_image(image: Image.Image) -> Image.Image:
    """
    Segmente une image en utilisant un modèle de segmentation sémantique.

    Args:
        image (PIL.Image): L'image d'entrée à segmenter.

    Returns:
        Image: L'image segmentée avec chaque segment coloré différemment selon
        sa classe identifiée.
    """
    global seg_image_processor, image_segmentor
    
    # S'assurer que l'image est dans le format RGB requis par le processeur
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
        pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)

        seg = seg_image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade20k_palette())
        
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
            
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert("RGB")
        
        return seg_image
    except Exception as e:
        logger.error(f"Erreur lors de la segmentation de l'image: {str(e)}")
        # Afficher plus d'informations sur l'image pour le débogage
        logger.error(f"Format de l'image: {image.format}, Mode: {image.mode}, Taille: {image.size}")
        raise

def resize_dimensions(dimensions: Tuple[int, int], target_size: int) -> Tuple[int, int]:
    """
    Redimensionne une image PIL à la taille cible tout en maintenant les proportions
    Si l'image est plus petite que la taille cible, la laisse telle quelle
    
    Args:
        dimensions: Tuple de (largeur, hauteur)
        target_size: La taille cible pour la plus grande dimension
        
    Returns:
        Tuple de nouvelles dimensions (largeur, hauteur)
    """
    width, height = dimensions

    # Vérifier si les deux dimensions sont plus petites que la taille cible
    if width < target_size and height < target_size:
        return dimensions

    # Déterminer le côté le plus grand
    if width > height:
        # Calculer le ratio d'aspect
        aspect_ratio = height / width
        # Redimensionner les dimensions
        return (target_size, int(target_size * aspect_ratio))
    else:
        # Calculer le ratio d'aspect
        aspect_ratio = width / height
        # Redimensionner les dimensions
        return (int(target_size * aspect_ratio), target_size)

def predict(
    image: Path,
    prompt: str,
    negative_prompt: str = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, extra, ugly, upholstered walls, fabric walls, plush walls, mirror, mirrored, functional, realistic",
    num_inference_steps: int = 50,
    guidance_scale: float = 15,
    prompt_strength: float = 0.8,
    seed: Optional[int] = None,
    depth_weight: float = 0.3,
) -> Image.Image:
    """
    Exécute une prédiction unique pour générer une image d'intérieur meublée
    
    Args:
        image: Chemin vers l'image d'entrée
        prompt: Texte de prompt pour le design
        negative_prompt: Texte de prompt négatif pour guider le design
        num_inference_steps: Nombre d'étapes de débruitage
        guidance_scale: Échelle pour le guidage sans classification
        prompt_strength: Force du prompt pour l'inpainting (1.0 = destruction complète de l'information dans l'image)
        seed: Graine aléatoire. Laisser None pour générer une graine aléatoire
        depth_weight: Poids pour le contrôle de la profondeur (entre 0.0 et 1.0)
        
    Returns:
        PIL.Image.Image: L'image générée
    """
    global pipe, control_items, additional_quality_suffix, mlsd_processor, depth_initialized
    
    # Si aucune seed n'est fournie, en générer une aléatoire
    if seed is None:
        seed = random.randint(0, 65535)
        
    img = Image.open(str(image))
    
    # Enrichir les prompts pour certains types de pièces
    if "bedroom" in prompt and "bed " not in prompt:
        prompt += ", with a queen size bed against the wall"
    elif "children room" in prompt or "children's room" in prompt:
        if "bed " not in prompt:
            prompt += ", with a twin bed against the wall"

    pos_prompt = prompt + f", {additional_quality_suffix}"

    # Redimensionner l'image d'entrée
    orig_w, orig_h = img.size
    new_width, new_height = resize_dimensions(img.size, 768)
    input_image = img.resize((new_width, new_height))

    # Prétraitement pour la segmentation controlnet
    real_seg = np.array(segment_image(input_image))
    unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_to_rgb(i) for i in unique_colors]
    chosen_colors, segment_items = filter_items(
        colors_list=unique_colors,
        items_list=segment_items,
        items_to_remove=control_items,
    )
    
    # Créer le masque pour l'inpainting
    mask = np.zeros_like(real_seg)
    for color in chosen_colors:
        color_matches = (real_seg == color).all(axis=2)
        mask[color_matches] = 1

    # Préparer les images pour l'inpainting
    image_np = np.array(input_image)
    image = Image.fromarray(image_np).convert("RGB")
    segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

    # Prétraitement pour mlsd controlnet
    mlsd_img = mlsd_processor(input_image)
    mlsd_img = mlsd_img.resize(image.size)
    
    if depth_initialized:
        try:
            logger.info("Génération de la carte de profondeur...")
            depth_img = get_depth_map(input_image)
            depth_img = depth_img.resize(image.size)
            
            # Définir les poids de contrôle en fonction des caractéristiques de l'image
            # Calculer le ratio d'aspect pour ajuster les poids
            aspect_ratio = orig_w / orig_h
            
            # Ajuster les poids en fonction du ratio d'aspect
            seg_weight = 0.4
            mlsd_weight = 0.2
            
            # Si la pièce est très large, augmenter l'importance de la profondeur et des lignes
            if aspect_ratio > 1.5:
                seg_weight = 0.35
                mlsd_weight = 0.25
                depth_weight = 0.4
            # Si la pièce est très haute, réduire l'importance des lignes
            elif aspect_ratio < 0.7:
                seg_weight = 0.45
                mlsd_weight = 0.15
                depth_weight = 0.4
                
            logger.info(f"Ratio d'aspect: {aspect_ratio:.2f}, Poids utilisés - Seg: {seg_weight}, MLSD: {mlsd_weight}, Depth: {depth_weight}")
            
            # Générer l'image avec le pipeline d'inpainting incluant la profondeur
            generated_image = pipe(
                prompt=pos_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                strength=prompt_strength,
                guidance_scale=guidance_scale,
                generator=[torch.Generator(device="cuda").manual_seed(seed)],
                image=image,
                mask_image=mask_image,
                control_image=[segmentation_cond_image, mlsd_img, depth_img],
                controlnet_conditioning_scale=[seg_weight, mlsd_weight, depth_weight],
                control_guidance_start=[0, 0.1, 0.05],
                control_guidance_end=[0.5, 0.25, 0.6],
            ).images[0]
            
            logger.info("Génération avec profondeur réussie")
        except Exception as e:
            logger.warning(f"Erreur lors de l'utilisation de la carte de profondeur: {str(e)}")
            logger.info("Retour au mode de génération sans carte de profondeur")
            
            # Fallback sans la carte de profondeur
            generated_image = pipe(
                prompt=pos_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                strength=prompt_strength,
                guidance_scale=guidance_scale,
                generator=[torch.Generator(device="cuda").manual_seed(seed)],
                image=image,
                mask_image=mask_image,
                control_image=[segmentation_cond_image, mlsd_img],
                controlnet_conditioning_scale=[0.4, 0.2],
                control_guidance_start=[0, 0.1],
                control_guidance_end=[0.5, 0.25],
            ).images[0]
    else:
        # Mode sans profondeur (2 ControlNets seulement)
        logger.info("Génération sans profondeur (non disponible)")
        generated_image = pipe(
            prompt=pos_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=[torch.Generator(device="cuda").manual_seed(seed)],
            image=image,
            mask_image=mask_image,
            control_image=[segmentation_cond_image, mlsd_img],
            controlnet_conditioning_scale=[0.4, 0.2],
            control_guidance_start=[0, 0.1],
            control_guidance_end=[0.5, 0.25],
        ).images[0]

    # Redimensionner l'image générée aux dimensions originales
    out_img = generated_image.resize(
        (orig_w, orig_h), Image.Resampling.LANCZOS
    )

    out_img = out_img.convert("RGB")

    return out_img

if __name__ == "__main__":
    import argparse
    
    # Création du parser d'arguments
    parser = argparse.ArgumentParser(description="Génération d'images basée sur un prompt et une image d'entrée")
    
    # Arguments principaux
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Chemin pour sauvegarder l'image générée")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt décrivant la scène à générer")
    
    # Arguments optionnels pour le contrôle de la génération
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la génération (défaut: 42)")
    parser.add_argument("--strength", type=float, default=0.85, help="Force du prompt (défaut: 0.85)")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Échelle de guidance (défaut: 9.0)")
    parser.add_argument("--steps", type=int, default=40, help="Nombre d'étapes d'inférence (défaut: 40)")
    parser.add_argument("--depth_weight", type=float, default=0.3, 
                       help="Poids de la profondeur pour ControlNet (défaut: 0.3, entre 0 et 1)")
    parser.add_argument("--verbose", action="store_true", help="Afficher les informations détaillées")
    
    # Analyse des arguments
    args = parser.parse_args()
    
    # Configuration du niveau de log en fonction de l'option verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Mode verbeux activé")
    
    try:
        # Chargement des modèles
        logger.info("Initialisation des modèles...")
        setup()
        
        # Préparation des chemins
        image_path = Path(args.image)
        output_path = Path(args.output)
        
        # Vérification de l'existence du fichier d'entrée
        if not image_path.exists():
            logger.error(f"L'image d'entrée n'existe pas: {image_path}")
            sys.exit(1)
        
        # Création du dossier parent pour le fichier de sortie si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Affichage des paramètres pour vérification
        logger.info(f"Traitement de l'image: {image_path}")
        logger.info(f"Prompt: {args.prompt}")
        if args.verbose:
            logger.debug(f"Paramètres: seed={args.seed}, strength={args.strength}, "
                        f"guidance_scale={args.guidance_scale}, steps={args.steps}, "
                        f"depth_weight={args.depth_weight}")
        
        # Prédiction
        result_image = predict(
            image=image_path, 
            prompt=args.prompt,
            seed=args.seed,
            prompt_strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            depth_weight=args.depth_weight
        )
        
        # Enregistrement de l'image générée
        result_image.save(output_path)
        logger.info(f"Image générée sauvegardée avec succès à: {output_path}")
    
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)