from PIL import Image
import torch
import numpy as np
import cv2
from segmentation_colors import ade20k_palette
import time
import logging
import functools
from typing import Tuple, Dict, Any
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Définition du périphérique (utilisation du même que dans depth.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilisation du périphérique pour la segmentation: {device}")

# Configuration par défaut
DEFAULT_CONFIG = {
    "model_id": "shi-labs/oneformer_ade20k_swin_large",
    "min_area_threshold": 50,
    "kernel_size": 3,
    "use_fast": False,
    "overlay_alpha": 0.5,
    "enable_morphological_cleanup": True,
    "cache_palette": True,
    # Configuration pour le débruitage
    "denoise_strength": 9,
    "denoise_sigma_color": 75,
    "denoise_sigma_space": 75
}

def load_model() -> Tuple[OneFormerProcessor, OneFormerForUniversalSegmentation]:
    """
    Charge le modèle de segmentation d'images.

    Returns:
        Un tuple contenant le processeur et le modèle de segmentation.
    """
    start_time = time.time()
    logger.info(f"Chargement du modèle de segmentation depuis {DEFAULT_CONFIG['model_id']}")

    processor = OneFormerProcessor.from_pretrained(
        DEFAULT_CONFIG["model_id"],
        use_fast=DEFAULT_CONFIG["use_fast"]
    )

    model = OneFormerForUniversalSegmentation.from_pretrained(DEFAULT_CONFIG["model_id"])
    model = model.to(device)

    if device.type == 'cuda':
        logger.info("GPU détecté, conversion du modèle en demi-précision pour optimisation")
        model = model.half()

    model.eval()

    elapsed_time = time.time() - start_time
    logger.info(f"Modèle de segmentation chargé en {elapsed_time:.2f} secondes")

    return processor, model

# Chargement du modèle et du processeur
processor, model = load_model()

# Mise en cache de la palette pour éviter de la recalculer à chaque appel
@functools.lru_cache(maxsize=1)
def get_cached_palette() -> np.ndarray:
    """
    Récupère la palette de couleurs ADE20K et la met en cache pour éviter de la recalculer.

    Returns:
        Palette de couleurs sous forme de tableau numpy
    """
    return np.array(ade20k_palette())

def colorize_segmentation(segmentation_map: np.ndarray) -> Image.Image:
    """
    Colorie une carte de segmentation en utilisant la palette de couleurs ADE20K.

    Args:
        segmentation_map: Carte de segmentation sous forme de tableau numpy

    Returns:
        Image colorée de la segmentation
    """
    if segmentation_map.ndim != 2:
        raise ValueError("La carte de segmentation doit être un tableau 2D")

    h, w = segmentation_map.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    palette = get_cached_palette()

    # Optimisation: utilisation de la vectorisation numpy au lieu d'une boucle
    unique_labels = np.unique(segmentation_map)
    for label in unique_labels:
        if label < len(palette):
            mask = (segmentation_map == label)
            color_seg[mask] = palette[label]

    return Image.fromarray(color_seg)

def create_overlay(original_image: Image.Image, segmentation_map: np.ndarray, alpha: float = DEFAULT_CONFIG["overlay_alpha"]) -> Image.Image:
    """
    Crée une superposition (overlay) de la segmentation sur l'image originale.

    Args:
        original_image: Image originale
        segmentation_map: Carte de segmentation sous forme de tableau numpy
        alpha: Niveau de transparence (0.0 à 1.0)

    Returns:
        Image avec la superposition de segmentation
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("Le paramètre alpha doit être compris entre 0.0 et 1.0")

    palette = get_cached_palette()

    h, w = segmentation_map.shape
    rgba_array = np.zeros((h, w, 4), dtype=np.uint8)

    # Optimisation: traitement par lot des classes
    unique_classes = np.unique(segmentation_map)
    for class_id in unique_classes:
        if class_id > 0:  # Ignorer l'arrière-plan (classe 0)
            if class_id < len(palette):
                color = palette[class_id]
                mask = (segmentation_map == class_id)
                rgba_array[mask] = [color[0], color[1], color[2], int(255 * alpha)]

    overlay = Image.fromarray(rgba_array, 'RGBA')

    result = original_image.copy().convert('RGBA')
    result.alpha_composite(overlay)

    return result

def apply_morphological_cleanup(
    segmentation: np.ndarray,
    kernel_size: int = DEFAULT_CONFIG["kernel_size"],
    min_area_threshold: int = DEFAULT_CONFIG["min_area_threshold"]
) -> np.ndarray:
    """
    Applique un nettoyage morphologique à la carte de segmentation pour réduire le bruit.

    Args:
        segmentation: Carte de segmentation sous forme de tableau numpy
        kernel_size: Taille du noyau pour les opérations morphologiques
        min_area_threshold: Seuil minimal pour considérer une région

    Returns:
        Carte de segmentation nettoyée
    """
    start_time = time.time()
    logger.debug(f"Début du nettoyage morphologique (kernel_size={kernel_size}, min_area_threshold={min_area_threshold})")

    # Créer une copie pour éviter de modifier l'original
    cleaned_segmentation = segmentation.copy()

    # Identifier les classes uniques
    unique_classes = np.unique(segmentation)

    # Créer un élément structurant pour les opérations morphologiques
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Traiter chaque classe séparément
    for class_id in unique_classes:
        if class_id == 0:  # Ignorer l'arrière-plan (généralement 0)
            continue

        # Créer un masque binaire pour la classe actuelle
        class_mask = (segmentation == class_id).astype(np.uint8)

        # Appliquer une ouverture morphologique pour éliminer les petits bruits
        # (érosion suivie de dilatation)
        opened_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

        # Appliquer une fermeture morphologique pour combler les petits trous
        # (dilatation suivie d'érosion)
        cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

        # Trouver les composantes connectées
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

        # Pour chaque composante connectée
        for i in range(1, num_labels):  # Commencer à 1 pour ignorer l'arrière-plan
            area = stats[i, cv2.CC_STAT_AREA]

            # Si la zone est trop petite, la supprimer
            if area < min_area_threshold:
                component_mask = (labels == i)

                # Trouver la classe la plus proche pour remplacer la petite région
                # Dilater la région pour trouver les voisins
                dilated_component = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=2)

                # Identifier les classes voisines (exclure la classe actuelle et l'arrière-plan)
                neighbor_mask = dilated_component & ~component_mask
                if np.any(neighbor_mask):
                    neighbor_classes = segmentation[neighbor_mask]
                    # Trouver la classe voisine la plus fréquente (hors classe actuelle)
                    neighbor_classes = neighbor_classes[neighbor_classes != class_id]
                    if len(neighbor_classes) > 0:
                        most_common_neighbor = np.bincount(neighbor_classes).argmax()
                        cleaned_segmentation[component_mask] = most_common_neighbor
                    else:
                        cleaned_segmentation[component_mask] = 0  # Arrière-plan si pas de voisin
                else:
                    cleaned_segmentation[component_mask] = 0  # Arrière-plan si pas de voisin

    # Appliquer un filtre médian pour lisser les frontières entre classes
    # Cela aide à éliminer les pixels isolés et à lisser les contours
    h, w = cleaned_segmentation.shape
    # Utiliser np.uint8, np.int16, ou np.float32 pour la compatibilité avec cv2.medianBlur
    cleaned_segmentation_dtype = np.uint8  # Choisir un type compatible
    cleaned_segmentation_for_blur = cleaned_segmentation.astype(cleaned_segmentation_dtype)

    median_filtered = cv2.medianBlur(cleaned_segmentation_for_blur, 3)
    cleaned_segmentation = median_filtered.astype(np.int64)

    elapsed_time = time.time() - start_time
    logger.debug(f"Nettoyage morphologique terminé en {elapsed_time:.4f} secondes")

    return cleaned_segmentation

def get_class_distribution(segmentation_map: np.ndarray) -> Dict[int, float]:
    """
    Calcule la distribution des classes dans une carte de segmentation.

    Args:
        segmentation_map: Carte de segmentation sous forme de tableau numpy

    Returns:
        Dictionnaire associant chaque classe à son pourcentage de pixels dans l'image
    """
    total_pixels = segmentation_map.size
    unique_classes, counts = np.unique(segmentation_map, return_counts=True)

    distribution = {int(class_id): float(count) / total_pixels * 100 for class_id, count in zip(unique_classes, counts)}

    logger.debug(f"Distribution des classes: {len(distribution)} classes trouvées")

    return distribution

def denoise_image(image: Image.Image,
                 strength: int = DEFAULT_CONFIG["denoise_strength"],
                 sigma_color: int = DEFAULT_CONFIG["denoise_sigma_color"],
                 sigma_space: int = DEFAULT_CONFIG["denoise_sigma_space"]) -> Image.Image:
    """
    Débruite l'image pour améliorer la qualité de la segmentation.
    Utilise une combinaison de filtres pour réduire le bruit tout en préservant les structures importantes.

    Args:
        image: L'image à débruiter.
        strength: Force du filtre bilatéral (paramètre d).
        sigma_color: Sigma dans l'espace de couleur.
        sigma_space: Sigma dans l'espace de coordonnées.

    Returns:
        L'image débruitée.
    """
    # Convertir en numpy pour le traitement
    image_cv = np.array(image)

    # Étape 1: Réduction du bruit avec préservation des bords
    # Le filtre bilatéral est excellent pour préserver les bords tout en lissant les textures
    denoised = cv2.bilateralFilter(image_cv, d=strength, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Étape 2: Légère réduction des détails fins qui peuvent perturber la segmentation
    # Appliquer un très léger flou gaussien pour réduire les textures trop détaillées
    # tout en préservant les structures importantes
    kernel_size = 3
    sigma = 0.8
    denoised = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), sigma)

    # Étape 3: Simplification de l'image pour faciliter la segmentation
    # Réduire légèrement la saturation pour que les variations de couleur trop subtiles
    # ne perturbent pas la segmentation
    hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
    # Réduire la saturation de 10%
    hsv[:,:,1] = hsv[:,:,1] * 0.9
    denoised = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Étape 4: Légère normalisation du contraste pour uniformiser les zones similaires
    # Cela aide le modèle à regrouper des zones de même classe
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Appliquer une égalisation d'histogramme très légère sur le canal L
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Fusionner les canaux
    lab = cv2.merge((l_channel, a_channel, b_channel))
    denoised = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(denoised)

def segment_image(
    image: Image.Image,
    processor: OneFormerProcessor = None,
    model: OneFormerForUniversalSegmentation = None,
    device: torch.device = None,
    config: Dict[str, Any] = None,
    apply_denoising: bool = True
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Segmente une image en utilisant le modèle OneFormer.

    Args:
        image: Image à segmenter
        processor: Processeur OneFormer (facultatif)
        model: Modèle OneFormer (facultatif)
        device: Appareil de calcul (CPU/GPU) (facultatif)
        config: Configuration personnalisée (facultatif)
        apply_denoising: Indique si le débruitage doit être appliqué avant la segmentation (par défaut: True)

    Returns:
        Tuple contenant (segmentation_tensor, image_segmentation_colorée)
    """
    # Fusionner la configuration par défaut avec la configuration personnalisée
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
        logger.info(f"Configuration personnalisée appliquée: {config}")

    start_time = time.time()
    logger.info(f"Début de la segmentation d'image de taille {image.size}")

    try:
        # Configuration des ressources si non fournies
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Utilisation du périphérique: {device}")

        if processor is None or model is None:
            # Ceci est pour une utilisation autonome, mais dans main.py ces paramètres seront passés
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

            if processor is None:
                logger.info(f"Chargement du processeur depuis {cfg['model_id']}")
                processor = OneFormerProcessor.from_pretrained(cfg['model_id'], use_fast=cfg['use_fast'])

            if model is None:
                logger.info(f"Chargement du modèle depuis {cfg['model_id']}")
                model = OneFormerForUniversalSegmentation.from_pretrained(cfg['model_id'])
                model = model.to(device)

                if device.type == 'cuda':
                    logger.info("GPU détecté, conversion du modèle en demi-précision pour optimisation")
                    model = model.half()

                model.eval()

        # Appliquer le débruitage si demandé
        if apply_denoising:
            logger.debug("Application du débruitage pour améliorer la segmentation")
            denoising_start = time.time()
            image_to_segment = denoise_image(
                image,
                strength=cfg["denoise_strength"],
                sigma_color=cfg["denoise_sigma_color"],
                sigma_space=cfg["denoise_sigma_space"]
            )
            denoising_time = time.time() - denoising_start
            logger.debug(f"Débruitage terminé en {denoising_time:.2f} secondes")
        else:
            logger.debug("Segmentation sans débruitage préalable")
            image_to_segment = image

        # Préparation des entrées pour le modèle
        prep_start = time.time()
        inputs = processor(images=image_to_segment, task_inputs=["semantic"], return_tensors="pt").to(device)
        prep_time = time.time() - prep_start
        logger.debug(f"Préparation des entrées terminée en {prep_time:.2f} secondes")

        # Inférence du modèle
        infer_start = time.time()
        # Convertir les entrées en float16 si le modèle est en float16
        if next(model.parameters()).dtype == torch.float16:
            inputs = {k: v.half() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        infer_time = time.time() - infer_start
        logger.debug(f"Inférence terminée en {infer_time:.2f} secondes")

        # Traitement des sorties du modèle
        post_start = time.time()
        segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        segmentation = segmentation.cpu().numpy()
        post_time = time.time() - post_start
        logger.debug(f"Post-traitement terminé en {post_time:.2f} secondes")

        # Nettoyage morphologique
        if cfg["enable_morphological_cleanup"]:
            morph_start = time.time()
            segmentation = apply_morphological_cleanup(
                segmentation,
                kernel_size=cfg["kernel_size"],
                min_area_threshold=cfg["min_area_threshold"]
            )
            morph_time = time.time() - morph_start
            logger.debug(f"Nettoyage morphologique terminé en {morph_time:.2f} secondes")

        # Colorisation de la segmentation
        color_start = time.time()
        colored_seg = colorize_segmentation(segmentation)
        color_time = time.time() - color_start
        logger.debug(f"Colorisation terminée en {color_time:.2f} secondes")

        total_time = time.time() - start_time
        logger.info(f"Segmentation terminée en {total_time:.2f} secondes")

        return segmentation, colored_seg

    except Exception as e:
        logger.error(f"Erreur lors de la segmentation: {e}")
        raise  # Relaisser l'exception pour la gestion dans main.py

# def main():
#     # Exemple d'utilisation (à remplacer par votre logique principale)
#     image_path = "inputs/02_before.png"  # Remplacez par le chemin de votre image
#     try:
#         image = Image.open(image_path)
#         segmentation, colored_seg = segment_image(image)

#         # Afficher ou sauvegarder les résultats
#         colored_seg.show()  # Afficher l'image segmentée
#         # colored_seg.save("outputs/segmented_image.png")  # Sauvegarder l'image segmentée

#         logger.info(f"Segmentation réussie pour {image_path}")
#     except FileNotFoundError:
#         logger.error(f"Image non trouvée: {image_path}")
#     except Exception as e:
#         logger.error(f"Erreur lors du traitement de {image_path}: {e}")

# if __name__ == "__main__":
#     main()
