from typing import Dict
from colors import COLOR_MAPPING_HEX, COLOR_MAPPING_CATEGORY_HEX

# Ce module contient des fonctions pour convertir les codes de couleur hexadécimaux en format RGBA
# Il est utilisé pour préparer les mappages de couleurs pour la visualisation dans une interface web
# Les dictionnaires principaux sont importés de colors.py puis convertis au format RGBA

def convert_hex_to_rgba(hex_code: str) -> str:
    """Convertit un code hexadécimal en format rgba.
    Args:
        hex_code (str): Chaîne hexadécimale
    Returns:
        str: Chaîne au format rgba
    """
    hex_code = hex_code.lstrip('#')
    return "rgba(" + str(int(hex_code[0:2], 16)) + ", " + str(int(hex_code[2:4], 16)) + ", " + str(int(hex_code[4:6], 16)) + ", 1.0)"

def convert_dict_to_rgba(color_dict: Dict[str, str]) -> Dict[str, str]:
    """Convertit tous les codes hexadécimaux en format rgba pour tous les éléments d'un dictionnaire.
    Args:
        color_dict (Dict[str, str]): Dictionnaire de couleurs
    Returns:
        Dict[str, str]: Dictionnaire de couleurs avec valeurs rgba
    """
    updated_dict = {}
    for k, v in color_dict.items():
        updated_dict[convert_hex_to_rgba(k)] = v
    return updated_dict

def convert_nested_dict_to_rgba(nested_dict: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Convertit tous les codes hexadécimaux en format rgba pour tous les éléments d'un dictionnaire imbriqué.
    Args:
        nested_dict (Dict[str, Dict[str, str]]): Dictionnaire imbriqué de couleurs
    Returns:
        Dict[str, Dict[str, str]]: Dictionnaire imbriqué de couleurs avec valeurs rgba
    """
    updated_dict = {}
    for k, v in nested_dict.items():
        updated_dict[k] = convert_dict_to_rgba(v)
    return updated_dict

# Dictionnaires principaux transformés de hex en rgba pour utilisation dans l'interface
# COLOR_MAPPING_RGBA: Version RGBA du dictionnaire principal pour toutes les classes
# COLOR_MAPPING_CATEGORY_RGBA: Version RGBA du dictionnaire de catégories
COLOR_MAPPING_RGBA: Dict[str, str] = convert_dict_to_rgba(COLOR_MAPPING_HEX)
COLOR_MAPPING_CATEGORY_RGBA: Dict[str, Dict[str, str]] = convert_nested_dict_to_rgba(COLOR_MAPPING_CATEGORY_HEX)