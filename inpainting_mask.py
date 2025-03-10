from PIL import Image, ImageFilter
import numpy as np
from segmentation_colors import map_colors_rgb
from typing import List, Tuple, Set
from segmentation import colorize_segmentation
import logging
import time

# Configuration du logger
logger = logging.getLogger(__name__)

# Liste statique des éléments à conserver (murs, plafond, sol, escaliers, etc.)
# ELEMENTS_TO_KEEP = {
#     "wall", "floor;flooring", "ceiling", "stairway;staircase", "stairs;steps",
#     "escalator;moving;staircase;moving;stairway", "door;double;door", "windowpane;window",
#     "column;pillar", "railing;rail", "fireplace;hearth;open;fireplace", "shelf",
#     "light;light;source", "sconce", "painting;picture", "mirror", "cabinet", "sculpture",
#     "lamp", "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
#     "plant;flora;plant;life", "stool"
# }
ELEMENTS_TO_KEEP = {
    "wall", "floor;flooring", "ceiling", "stairway;staircase", "stairs;steps",
    "door;double;door", "windowpane;window", "column;pillar", "railing;rail",
    "fireplace;hearth;open;fireplace", "shelf"
}

def filter_items(
    colors_list: List[Tuple],
    items_list: List[str],
    items_to_remove: Set[str]
) -> Tuple[List, List]:
    """
    Filtre les couleurs et les éléments en fonction d'une liste d'éléments à supprimer.
    
    Args:
        colors_list: Liste des couleurs
        items_list: Liste des éléments correspondant aux couleurs
        items_to_remove: Liste ou ensemble des éléments à supprimer
        
    Returns:
        Tuple contenant les couleurs filtrées et les éléments filtrés
    """
    logger.debug(f"Filtrage de {len(items_list)} éléments, {len(items_to_remove)} éléments à conserver")
    
    # Utilisation de la compréhension de liste pour un code plus concis et plus rapide
    filtered_items_and_colors = [(color, item) for color, item in zip(colors_list, items_list) if item not in items_to_remove]
    
    # Déballage des résultats filtrés en deux listes
    filtered_colors, filtered_items = zip(*filtered_items_and_colors) if filtered_items_and_colors else ([], [])
    
    logger.debug(f"Résultat du filtrage: {len(filtered_colors)} éléments à masquer")
    return list(filtered_colors), list(filtered_items)

def get_unique_colors_and_items(seg_mask_array: np.ndarray) -> Tuple[List[Tuple], List[str]]:
    """
    Extrait les couleurs uniques et les éléments correspondants d'un masque de segmentation.
    
    Args:
        seg_mask_array: Tableau numpy représentant le masque de segmentation colorisé
        
    Returns:
        Tuple contenant (liste des couleurs uniques, liste des éléments correspondants)
    """
    # Mesure du temps d'exécution
    start_time = time.time()
    
    # Trouver les couleurs uniques
    unique_colors = np.unique(seg_mask_array.reshape(-1, seg_mask_array.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    
    # Mapper les couleurs aux éléments
    segment_items = [map_colors_rgb(color) for color in unique_colors]
    
    # Log du temps d'exécution et du nombre d'éléments trouvés
    elapsed_time = time.time() - start_time
    logger.debug(f"Extraction de {len(unique_colors)} couleurs uniques en {elapsed_time:.4f} secondes")
    
    return unique_colors, segment_items

def get_inpainting_mask(segmentation_mask: np.ndarray, image: Image.Image = None) -> Image.Image:
    """
    Génère un masque pour l'inpainting basé sur la segmentation.
    Le masque est blanc (255) pour les pixels à supprimer, noir (0) pour les pixels à conserver.
    Cible spécifiquement les meubles et objets qui ne sont pas sur les murs, au plafond ou des escaliers.
    
    Args:
        segmentation_mask: Masque de segmentation sous forme de tableau numpy
        image: Image originale (facultatif, non utilisé actuellement)
        
    Returns:
        Masque d'inpainting sous forme d'image PIL
    """
    # Mesure du temps d'exécution
    start_time = time.time()
    logger.info("Début de la génération du masque d'inpainting")
    
    # S'assurer que le masque de segmentation est au format uint8 avant de le convertir en Image
    segmentation_mask_uint8 = segmentation_mask.astype(np.uint8)
    
    # Utiliser colorize_segmentation pour convertir le masque de segmentation en une image colorée
    logger.debug("Colorisation du masque de segmentation")
    segmentation_only_mask = colorize_segmentation(segmentation_mask_uint8)
    
    # Trouver les couleurs uniques et les éléments correspondants
    logger.debug("Extraction des couleurs uniques et des éléments correspondants")
    seg_mask_array = np.array(segmentation_only_mask)
    
    # Extraction des couleurs uniques directement sans utiliser la fonction mise en cache
    unique_colors = np.unique(seg_mask_array.reshape(-1, seg_mask_array.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_rgb(color) for color in unique_colors]
    
    # Filtrer pour obtenir uniquement les éléments à masquer (meubles et autres objets)
    logger.debug("Filtrage des éléments à masquer")
    chosen_colors, segment_items = filter_items(
        colors_list=unique_colors,
        items_list=segment_items,
        items_to_remove=ELEMENTS_TO_KEEP
    )
    
    # Créer le masque initial basé sur les couleurs filtrées
    logger.debug(f"Création du masque pour {len(chosen_colors)} éléments")
    mask = np.zeros_like(seg_mask_array)
    
    # Optimisation: traiter toutes les couleurs en une seule passe si possible
    if chosen_colors:
        # Créer un masque pour chaque couleur et les combiner
        for color in chosen_colors:
            # Vectorisation de la comparaison des couleurs
            color_matches = np.all(seg_mask_array == color, axis=2)
            mask[color_matches] = 255
    
    # Convertir en image PIL
    mask_image = Image.fromarray(mask.astype(np.uint8))
    
    # Élargir la région du masque pour qu'elle efface également le voisinage des objets masqués
    logger.debug("Application du filtre MaxFilter pour élargir les régions du masque")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(15))

    # Log du temps d'exécution total
    elapsed_time = time.time() - start_time
    logger.info(f"Masque d'inpainting généré en {elapsed_time:.4f} secondes")
    
    return mask_image