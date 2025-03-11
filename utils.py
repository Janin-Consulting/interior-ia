import gc
import torch
import numpy as np

from PIL import Image
from scipy.signal import fftconvolve
from typing import Dict, Tuple, List
from palette import COLOR_MAPPING_RGBA
from colors import COLOR_MAPPING_HEX

# Module contenant des fonctions utilitaires pour le traitement d'images
# et la manipulation des couleurs dans le cadre de la segmentation
# Ces fonctions sont utilisées dans différents modules du projet

def to_rgb(color: str) -> Tuple[int, int, int]:
    """Convertit une couleur hexadécimale en RGB.
    Args:
        color (str): Couleur hexadécimale
    Returns:
        Tuple[int, int, int]: Couleur RGB
    """
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

def map_colors(color: str) -> str:
    """Associe une couleur à sa valeur hexadécimale.
    Args:
        color (str): Nom de la couleur
    Returns:
        str: Valeur hexadécimale
    """
    return COLOR_MAPPING_RGBA[color]

def map_colors_to_rgb(color: Tuple[int, int, int]) -> str:
    """Associe une couleur RGB à sa classe correspondante.
    Args:
        color (Tuple[int, int, int]): Couleur RGB
    Returns:
        str: Nom de la classe
    """
    return HEX_TO_RGB_CLASS_MAPPING[color]

def convolution(mask: Image.Image, size: int = 9) -> Image.Image:
    """Méthode pour flouter un masque.
    Args:
        mask (Image.Image): Image du masque
        size (int, optional): Taille du flou. Par défaut à 9.
    Returns:
        Image.Image: Masque flouté
    """
    mask = np.array(mask.convert("L"))
    conv = np.ones((size, size)) / size**2
    mask_blended = fftconvolve(mask, conv, 'same')
    mask_blended = mask_blended.astype(np.uint8).copy()

    border = size

    # Remplace les bords avec les valeurs originales
    mask_blended[:border, :] = mask[:border, :]
    mask_blended[-border:, :] = mask[-border:, :]
    mask_blended[:, :border] = mask[:, :border]
    mask_blended[:, -border:] = mask[:, -border:]

    return Image.fromarray(mask_blended).convert("L")

def flush() -> None:
    """Libère la mémoire GPU et nettoie le garbage collector."""
    gc.collect()
    torch.cuda.empty_cache()

def postprocess_image_masking(inpainted: Image.Image, image: Image.Image,
                              mask: Image.Image) -> Image.Image:
    """Méthode pour post-traiter l'image inpaintée.
    Args:
        inpainted (Image.Image): Image inpaintée
        image (Image.Image): Image originale
        mask (Image.Image): Masque
    Returns:
        Image.Image: Image inpaintée finalisée
    """
    final_inpainted = Image.composite(inpainted.convert("RGBA"),
                                      image.convert("RGBA"), mask)
    return final_inpainted.convert("RGB")

# Liste des noms de couleurs disponibles dans le mapping
COLOR_NAMES: List[str] = list(COLOR_MAPPING_HEX.keys())

# Conversion des codes hexadécimaux en valeurs RGB
# Inclut également les couleurs noir et blanc (0,0,0) et (255,255,255)
COLOR_RGB_VALUES: List[Tuple[int, int, int]] = [to_rgb(k) for k in COLOR_MAPPING_HEX.keys()] + [(0, 0, 0),
                                                         (255, 255, 255)]

# Dictionnaire associant les noms de classes à leurs valeurs RGB correspondantes
CLASS_TO_RGB_MAPPING: Dict[str, Tuple[int, int, int]] = {v: to_rgb(k) for k, v in COLOR_MAPPING_HEX.items()}

# Dictionnaire associant les couleurs RGB aux noms de classes
HEX_TO_RGB_CLASS_MAPPING: Dict[Tuple[int, int, int], str] = {to_rgb(k): v for k, v in COLOR_MAPPING_HEX.items()}