from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

def get_empty_room(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Génère une version de la pièce sans meubles en utilisant un modèle d'inpainting.
    
    Args:
        image: Image source contenant la pièce avec meubles
        mask: Masque d'inpainting (blanc pour les zones à remplacer)
        
    Returns:
        Image de la pièce vide après inpainting
    """
    # Mesure du temps d'exécution
    import time
    start_time = time.time()
    logger.info("Début de la génération de la pièce vide")
    
    try:
        # Chargement du modèle d'inpainting
        logger.debug("Chargement du modèle d'inpainting")
        
        # Désactiver les warnings spécifiques pendant le chargement du modèle
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*CLIPFeatureExtractor is deprecated.*")
            # Utiliser torch.float16 au lieu de torch.float32 pour réduire l'empreinte mémoire
            pipe = AutoPipelineForInpainting.from_pretrained('lykon/absolute-reality-1.6525-inpainting', 
                                                           torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
        # Utiliser le GPU si disponible, sinon le CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Définition des prompts pour l'inpainting
        inpaint_prompt = "Empty room, with only empty walls, floor, ceiling, doors, windows, photorealistic, high resolution, detailed texture"
        negative_prompt = "furnitures, sofa, cough, table, plants, rug, home equipment, music equipment, shelves, books, light, lamps, window, radiator, blurry, low quality, pixelated, artifacts, compression artifacts"
        
        # Préservation de la qualité d'image pendant le redimensionnement
        logger.debug("Préparation des images pour l'inpainting")
        
        # Calculer la taille optimale pour l'inpainting tout en préservant le ratio d'aspect
        width, height = image.size
        aspect_ratio = width / height
        
        # Déterminer la taille maximale possible en fonction de la mémoire disponible
        if torch.cuda.is_available():
            # Utiliser une taille plus grande si la mémoire GPU est suffisante
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            # Estimation empirique: 768x768 nécessite environ 4 Go
            if free_memory > 6 * 1024 * 1024 * 1024:  # Plus de 6 Go disponible
                inpaint_size = 768
            elif free_memory > 4 * 1024 * 1024 * 1024:  # Plus de 4 Go disponible
                inpaint_size = 640
            else:
                inpaint_size = 512
        else:
            # Sur CPU, utiliser une taille plus petite
            inpaint_size = 512
            
        logger.debug(f"Utilisation d'une taille d'inpainting de {inpaint_size}x{inpaint_size}")
        
        # Calculer les dimensions en préservant le ratio d'aspect
        if width >= height:
            new_width = inpaint_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = inpaint_size
            new_width = int(new_height * aspect_ratio)
            
        # S'assurer que les dimensions sont des multiples de 8 (requis par certains modèles)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Redimensionnement de haute qualité pour l'inpainting
        image_source_for_inpaint = image.resize((new_width, new_height), Image.LANCZOS)
        image_mask_for_inpaint = mask.resize((new_width, new_height), Image.LANCZOS)
        
        # Configuration du générateur pour la reproductibilité
        generator = [torch.Generator(device=device).manual_seed(20)]

        # Exécution de l'inpainting avec plus d'étapes pour une meilleure qualité
        logger.debug("Exécution de l'inpainting")
        try:
            image_inpainting_auto = pipe(
                prompt=inpaint_prompt, 
                negative_prompt=negative_prompt, 
                generator=generator, 
                strength=0.8,
                image=image_source_for_inpaint, 
                mask_image=image_mask_for_inpaint, 
                guidance_scale=10.0,
                num_inference_steps=20  # Augmentation du nombre d'étapes pour une meilleure qualité
            ).images[0]
            
            # Redimensionnement de haute qualité pour revenir à la taille d'origine
            logger.debug("Redimensionnement de l'image résultante")
            
            # Utiliser un redimensionnement de haute qualité pour préserver les détails
            if image_inpainting_auto.size[0] < width or image_inpainting_auto.size[1] < height:
                # Pour l'agrandissement, utiliser LANCZOS
                image_inpainting_auto = image_inpainting_auto.resize((width, height), Image.LANCZOS)
            else:
                # Pour la réduction, utiliser un algorithme qui préserve les détails
                import numpy as np
                import cv2
                
                # Convertir en numpy pour utiliser cv2
                img_np = np.array(image_inpainting_auto)
                
                # Redimensionner avec INTER_AREA qui est meilleur pour les réductions
                img_np = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)
                
                # Appliquer un léger rehaussement de netteté
                kernel = np.array([[-0.2, -0.2, -0.2], [-0.2, 2.8, -0.2], [-0.2, -0.2, -0.2]])
                img_np = cv2.filter2D(img_np, -1, kernel)
                
                # Convertir en PIL Image
                image_inpainting_auto = Image.fromarray(img_np)
            
        finally:
            # Libération immédiate de la mémoire des images temporaires
            del image_source_for_inpaint, image_mask_for_inpaint
            
            # Libération de la mémoire
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        elapsed_time = time.time() - start_time
        logger.info(f"Génération de la pièce vide terminée en {elapsed_time:.2f} secondes")
        
        return image_inpainting_auto
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la pièce vide: {str(e)}")
        # En cas d'erreur, retourner l'image originale
        return image