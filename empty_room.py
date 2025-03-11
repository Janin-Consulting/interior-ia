#!/usr/bin/env python

# empty_room.py - Génère des pièces vides de style européen français
# Usage: python empty_room.py --output_dir output_directory [--room_type type] [--count n] [--width w] [--height h]
# Example: python empty_room.py --output_dir pièces_vides --room_type salon --count 3

import argparse
import os
import random
import logging
import torch
import numpy as np

from PIL import Image, ImageDraw
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector, CannyDetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from colors import ade20k_palette

# Essayer d'importer les monkey patches s'ils sont disponibles
try:
    from monkey_patch import apply_patches
    apply_patches()
    logging.info("Monkey patches appliqués avec succès.")
except ImportError:
    logging.warning("Module monkey_patch non trouvé. Poursuite sans patches.")
except Exception as e:
    logging.warning(f"Erreur lors de l'application des monkey patches: {e}")

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales - Amélioration des prompts pour accentuer la profondeur
QUALITY_SUFFIX = "photorealistic, french european style, dramatic lighting, ultra detailed, depth of field, sense of depth, interior design, 4K, high resolution, elegant"
NEGATIVE_PROMPT = "furniture, people, closeup, wall only, floor only, flat, shallow space, blurry, out of focus, lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blur, surreal, extra, ugly, upholstered walls, fabric walls, plush walls, mirror, mirrored"

# Ajout de mots-clés liés à la profondeur dans les descriptions des pièces
ROOM_TYPES = {
    "salon": "ultra wide angle view of empty living room with depth, deep perspective, spacious, walls ceiling floor, decorative molding, large windows, sense of depth",
    "chambre": "ultra wide angle view of empty bedroom with depth, deep perspective, spacious, walls ceiling floor, decorative molding, tall windows, sense of depth",
    "salle_a_manger": "ultra wide angle view of empty dining room with depth, deep perspective, spacious, walls ceiling floor, decorative molding, large windows, sense of depth",
    "bureau": "ultra wide angle view of empty home office with depth, deep perspective, spacious, walls ceiling floor, decorative molding, large window, sense of depth",
    "cuisine": "ultra wide angle view of empty kitchen with depth, deep perspective, spacious, walls ceiling floor, decorative molding, windows, sense of depth",
    "salle_de_bain": "ultra wide angle view of empty bathroom with depth, deep perspective, spacious, walls ceiling floor, decorative molding, window, sense of depth"
}

# Initialiser les détecteurs et les processeurs
def init_processors():
    # Initialiser les processeurs pour les images de contrôle
    mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
    canny = CannyDetector()
    
    # Initialiser le segmenteur d'image si possible
    try:
        seg_image_processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        logger.info("Segmenteur d'image initialisé avec succès")
        return mlsd, canny, seg_image_processor, image_segmentor
    except Exception as e:
        logger.warning(f"Impossible d'initialiser le segmenteur d'image: {e}")
        return mlsd, canny, None, None

# Pipeline setup
def setup_pipeline():
    # Utiliser les mêmes modèles que dans predict.py
    controlnet = [
        ControlNetModel.from_pretrained(
            "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        )
    ]

    # Utiliser le même modèle de base que dans predict.py
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V3.0_VAE",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )

    # Utiliser le même scheduler que dans predict.py
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Activer l'optimisation de mémoire
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Using xformers for memory-efficient attention")
    except Exception as e:
        logger.warning(f"xformers not available, continuing without memory-efficient attention: {e}")
        try:
            pipe.enable_attention_slicing()
            logger.info("Using attention slicing instead")
        except:
            logger.warning("Could not enable attention slicing either")
    
    pipe = pipe.to("cuda")
    return pipe

# Segmenter l'image si possible
@torch.inference_mode()
def segment_image(image, seg_image_processor, image_segmentor):
    if seg_image_processor is None or image_segmentor is None:
        return None
    
    try:
        # S'assurer que l'image est en mode RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
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
        logger.error(f"Erreur lors de la segmentation de l'image: {e}")
        return None

# Générer une image initiale avec plus de perspective pour guider la génération
def create_initial_image(width, height, seed):
    np.random.seed(seed)
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Dessiner un rectangle plus large pour suggérer une perspective
    margin_h = width // 8
    margin_v = height // 8
    draw.rectangle([(margin_h, margin_v), (width - margin_h, height - margin_v)], outline="gray", width=2)
    
    # Ajouter des lignes de perspective pour suggérer la profondeur
    center_x, center_y = width // 2, height // 2
    draw.line([(margin_h, margin_v), (center_x, center_y - height//8)], fill="gray", width=1)
    draw.line([(width - margin_h, margin_v), (center_x, center_y - height//8)], fill="gray", width=1)
    draw.line([(margin_h, height - margin_v), (center_x, center_y + height//8)], fill="gray", width=1)
    draw.line([(width - margin_h, height - margin_v), (center_x, center_y + height//8)], fill="gray", width=1)
    
    # Ajouter un léger bruit pour la texture
    noise_img = Image.fromarray(np.random.randint(0, 20, (height, width, 3), dtype=np.uint8))
    return Image.blend(image, noise_img, 0.05)

# Générer une pièce vide avec plus de paramètres pour contrôler la profondeur
def generate_empty_room(pipe, output_dir, room_type, width=832, height=512, seed=None, steps=60, guidance_scale=9.5):
    if room_type not in ROOM_TYPES:
        logger.warning(f"Type de pièce {room_type} non trouvé. Utilisation du type 'salon' par défaut")
        room_type = 'salon'

    prompt = f"{ROOM_TYPES[room_type]}, {QUALITY_SUFFIX}"
    mlsd_processor, canny_processor, seg_image_processor, image_segmentor = init_processors()

    os.makedirs(output_dir, exist_ok=True)

    current_seed = seed if seed is not None else random.randint(1, 10000)
    generator = torch.Generator(device="cuda").manual_seed(current_seed)

    # Préparer les images de contrôle
    init_image = create_initial_image(width, height, current_seed)
    mlsd_image = mlsd_processor(init_image)
    
    # Tenter de segmenter l'image si le segmenteur est disponible
    seg_image = segment_image(init_image, seg_image_processor, image_segmentor)
    
    if seg_image is not None:
        # Si nous avons une image segmentée, utiliser MLSD et Segmentation comme contrôle
        control_images = [seg_image, mlsd_image]
    else:
        # Sinon utiliser MLSD et Canny
        canny_image = canny_processor(init_image)
        control_images = [mlsd_image, mlsd_image]  # Utiliser MLSD deux fois si pas de segmentation
    
    # Générer l'image avec les paramètres ajustés pour améliorer la profondeur
    output = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_images,
        controlnet_conditioning_scale=[0.7, 0.7],  # Échelles appropriées pour les contrôleurs
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height
    ).images[0]

    # Sauvegarder l'image
    output_path = f"{output_dir}/{room_type}_{current_seed}.png"
    output.save(output_path)
    logger.info(f"Saved: {output_path}")
    return output_path

# Script principal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de pièces vides en style européen français")
    parser.add_argument('--room_type', type=str, default="salon", help="Type de pièce à générer")
    parser.add_argument('--output_dir', required=True, help="Répertoire de sortie pour les images")
    parser.add_argument('--count', type=int, default=1, help="Nombre d'images à générer")
    parser.add_argument('--width', type=int, default=832, help="Largeur de l'image")
    parser.add_argument('--height', type=int, default=512, help="Hauteur de l'image")
    parser.add_argument('--seed', type=int, default=None, help="Seed pour la génération (optionnel)")
    parser.add_argument('--steps', type=int, default=60, help="Nombre d'étapes d'inférence")
    parser.add_argument('--guidance_scale', type=float, default=9.5, help="Échelle de guidance")

    args = parser.parse_args()

    # Configurer le pipeline une seule fois
    pipe = setup_pipeline()

    # Générer le nombre d'images spécifié
    for i in range(args.count):
        current_seed = random.randint(0, 99999) if args.seed is None else args.seed + i
        generate_empty_room(
            pipe, 
            args.output_dir, 
            args.room_type, 
            args.width, 
            args.height, 
            current_seed,
            args.steps,
            args.guidance_scale
        )