import warnings
# Filtrer les avertissements spécifiques liés aux paramètres non valides dans transformers
# Important: placer les filtres avant toute importation de modules
warnings.filterwarnings("ignore", message="The following named arguments are not valid for.*and were ignored:.*")
warnings.filterwarnings("ignore", category=UserWarning)
# Filtrer spécifiquement le warning concernant CLIPFeatureExtractor
warnings.filterwarnings("ignore", category=FutureWarning, message=".*CLIPFeatureExtractor is deprecated.*")

from PIL import Image
import torch
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from typing import Dict, List, Tuple, Any
import traceback
import logging
import functools
import signal
import sys

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer les modules après avoir configuré les filtres d'avertissement
from segmentation import create_overlay, segment_image, processor, model, device
from depth import get_depth_image
from inpainting_mask import get_inpainting_mask
from empty_room import get_empty_room

# Le périphérique est maintenant importé du module segmentation
logger.info(f"Utilisation du périphérique pour le traitement principal : {device}")

# Variable globale pour suivre l'état d'interruption
interrupted = False

# Gestionnaire de signal pour Ctrl+C
def signal_handler(sig, frame):
    global interrupted
    if not interrupted:
        logger.warning("Signal d'interruption reçu (Ctrl+C). Arrêt en cours...")
        interrupted = True
    else:
        logger.warning("Deuxième signal d'interruption reçu. Arrêt forcé.")
        sys.exit(1)

# Enregistrer le gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configuration par défaut pour le traitement des images
DEFAULT_PROCESSING_CONFIG = {
    "resolution": 768,
}

@functools.lru_cache(maxsize=32)
def resize_image(input_image_path: str, resolution: int = DEFAULT_PROCESSING_CONFIG["resolution"]) -> Image.Image:
    """
    Redimensionne l'image pour qu'elle soit compatible avec le modèle de segmentation.
    Utilise des techniques avancées pour préserver la qualité et les détails de l'image.
    
    Args:
        input_image_path: Chemin d'accès à l'image à redimensionner.
        resolution: La résolution de sortie (par défaut : 768).
    
    Returns:
        L'image redimensionnée avec qualité préservée.
    """
    # Charger l'image depuis le chemin
    input_image = Image.open(input_image_path).convert("RGB")
    
    width, height = input_image.size
    aspect_ratio = width / height

    # Calculer les dimensions cibles
    if width >= height:
        new_width = resolution
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = resolution
        new_width = int(new_height * aspect_ratio)

    # S'assurer que les dimensions sont des multiples de 64 pour la compatibilité avec le modèle
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    # Si les dimensions sont identiques, retourner l'image originale
    if width == new_width and height == new_height:
        return input_image
    
    # Approche ultra-haute qualité pour le redimensionnement
    logger.debug(f"Redimensionnement de {width}x{height} à {new_width}x{new_height}")
    
    # Convertir en numpy pour un traitement avancé
    img_np = np.array(input_image)
    
    # Pour les agrandissements importants (plus de 2x)
    if new_width > width * 2 or new_height > height * 2:
        logger.debug("Agrandissement important détecté, utilisation d'une approche multi-étapes")
        
        # Étape 1: Prétraitement pour améliorer les détails
        # Convertir en LAB pour séparer la luminance des couleurs
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        
        # Améliorer les détails dans le canal de luminance avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Recombiner les canaux
        img_lab = cv2.merge((l_channel, a_channel, b_channel))
        img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Étape 2: Agrandissement progressif avec amélioration des détails
        current_width, current_height = width, height
        
        while current_width < new_width * 0.8 or current_height < new_height * 0.8:
            # Calculer la prochaine taille intermédiaire (max 1.6x)
            next_width = min(int(current_width * 1.6), new_width)
            next_height = min(int(current_height * 1.6), new_height)
            
            # Agrandir avec INTER_CUBIC
            img_np = cv2.resize(img_np, (next_width, next_height), interpolation=cv2.INTER_CUBIC)
            
            # Améliorer la netteté après chaque étape
            # Utiliser un filtre de netteté adaptatif qui préserve les bords
            img_np = cv2.detailEnhance(img_np, sigma_s=0.5, sigma_r=0.15)
            
            current_width, current_height = next_width, next_height
        
        # Étape 3: Dernier redimensionnement pour atteindre la taille exacte
        img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Étape 4: Post-traitement pour améliorer la netteté finale
        # Utiliser un rehaussement de détails adaptatif
        img_np = cv2.detailEnhance(img_np, sigma_s=0.3, sigma_r=0.2)
        
        return Image.fromarray(img_np)
    
    # Pour les réductions importantes (plus de 2.5x)
    elif width > new_width * 2.5 or height > new_height * 2.5:
        logger.debug("Réduction importante détectée, utilisation d'une approche pyramidale")
        
        # Étape 1: Prétraitement - Léger flou gaussien pour réduire le moiré
        sigma = 0.6
        img_np = cv2.GaussianBlur(img_np, (0, 0), sigma)
        
        # Étape 2: Réduction progressive avec pyramide gaussienne
        current_width, current_height = width, height
        
        while current_width > new_width * 1.8 or current_height > new_height * 1.8:
            # Réduire par un facteur de 1.5
            current_width = max(int(current_width / 1.5), new_width)
            current_height = max(int(current_height / 1.5), new_height)
            
            # Utiliser INTER_AREA pour la réduction
            img_np = cv2.resize(img_np, (current_width, current_height), interpolation=cv2.INTER_AREA)
            
            # Appliquer un léger rehaussement des contours après chaque étape
            if current_width > new_width * 1.2:
                img_np = cv2.detailEnhance(img_np, sigma_s=0.3, sigma_r=0.15)
        
        # Étape 3: Dernière étape avec INTER_AREA
        img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Étape 4: Post-traitement pour restaurer la netteté
        # Utiliser un rehaussement de détails adaptatif qui préserve les textures
        img_np = cv2.detailEnhance(img_np, sigma_s=0.3, sigma_r=0.15)
        
        # Étape 5: Amélioration des couleurs
        # Convertir en LAB pour améliorer séparément la luminance et les couleurs
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        
        # Améliorer le contraste dans le canal de luminance
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Recombiner les canaux
        img_lab = cv2.merge((l_channel, a_channel, b_channel))
        img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_np)
    
    # Pour les réductions modérées (entre 1.3x et 2.5x)
    elif width > new_width * 1.3 or height > new_height * 1.3:
        logger.debug("Réduction modérée détectée, utilisation d'une approche optimisée")
        
        # Étape 1: Redimensionnement avec INTER_AREA (optimal pour les réductions)
        img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Étape 2: Amélioration des détails
        img_np = cv2.detailEnhance(img_np, sigma_s=0.3, sigma_r=0.15)
        
        return Image.fromarray(img_np)
    
    # Pour les changements de taille mineurs (moins de 1.3x)
    else:
        logger.debug("Changement mineur de taille détecté")
        
        # Pour les agrandissements mineurs
        if new_width > width or new_height > height:
            # Utiliser INTER_LANCZOS4 pour un agrandissement de haute qualité
            img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Léger rehaussement des détails
            img_np = cv2.detailEnhance(img_np, sigma_s=0.2, sigma_r=0.1)
            
        # Pour les réductions mineures
        else:
            # Utiliser INTER_AREA pour les réductions
            img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Très léger rehaussement des détails
            img_np = cv2.detailEnhance(img_np, sigma_s=0.2, sigma_r=0.1)
        
        return Image.fromarray(img_np)

def process_single_image(img_path: str, save_dir: str) -> Tuple[str, bool]:
    """
    Traite une seule image et sauvegarde les résultats.
    
    Args:
        img_path: Chemin d'accès à l'image à traiter
        save_dir: Répertoire de sauvegarde des résultats
        
    Returns:
        Tuple contenant (chemin de l'image, succès du traitement)
    """
    start_time = time.time()
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    
    try:
        # Créer un dossier pour chaque image d'entrée
        img_save_dir = os.path.join(save_dir, base_name)
        os.makedirs(img_save_dir, exist_ok=True)
        
        # Vérifier si le traitement doit être interrompu
        if interrupted:
            logger.warning(f"Traitement interrompu avant de commencer {img_name}")
            return img_path, False
        
        # Copier l'image originale dans le dossier de sortie
        logger.debug(f"Copie de l'image originale {img_name}")
        original_image = Image.open(img_path).convert("RGB")
        original_image.save(os.path.join(img_save_dir, f"{base_name}_original.png"))
        
        # Charger et redimensionner l'image
        logger.debug(f"Redimensionnement de l'image {img_name}")
        resized_image = resize_image(img_path)
        resized_image.save(os.path.join(img_save_dir, f"{base_name}_resized.png"))
        
        # Segmenter l'image (le débruitage est maintenant intégré dans la fonction segment_image)
        logger.debug(f"Segmentation de l'image {img_name} (avec débruitage intégré)")
        with torch.no_grad():  # Réduire l'utilisation de mémoire pendant l'inférence
            # Configuration personnalisée pour la segmentation
            segmentation_config = {
                "min_area_threshold": 100,  # Augmenter le seuil pour éliminer plus de petites régions
                "kernel_size": 5,  # Augmenter la taille du noyau pour des opérations morphologiques plus efficaces
                "enable_morphological_cleanup": True  # S'assurer que le nettoyage morphologique est activé
            }
            segmentation, colored_seg = segment_image(
                resized_image, 
                processor, 
                model, 
                device, 
                config=segmentation_config,
                apply_denoising=True  # Activer le débruitage intégré
            )
            
        # Vérifier si le traitement doit être interrompu
        if interrupted:
            logger.warning(f"Traitement interrompu après la segmentation de {img_name}")
            return img_path, False
        
        # Sauvegarder l'image segmentée
        colored_seg.save(os.path.join(img_save_dir, f"{base_name}_segmentation.png"))
        
        # Créer et sauvegarder l'overlay global (utiliser l'image originale redimensionnée pour une meilleure qualité)
        logger.debug(f"Création de l'overlay pour {img_name}")
        overlay = create_overlay(resized_image, segmentation)
        overlay = overlay.convert('RGB')
        overlay.save(os.path.join(img_save_dir, f"{base_name}_overlay.png"))
        
        # Libérer la mémoire de l'overlay dès qu'il est sauvegardé
        del overlay
        
        # Vérifier si le traitement doit être interrompu
        if interrupted:
            logger.warning(f"Traitement interrompu après la création de l'overlay de {img_name}")
            return img_path, False
        
        # Créer une carte de profondeur
        logger.debug(f"Génération de la carte de profondeur pour {img_name}")
        with torch.no_grad():  # Réduire l'utilisation de mémoire pendant l'inférence
            depth_image = get_depth_image(resized_image)  # Utiliser l'image originale redimensionnée
        depth_image.save(os.path.join(img_save_dir, f"{base_name}_depth.png"))
        
        # Libérer la mémoire de la carte de profondeur dès qu'elle est sauvegardée
        del depth_image
        
        # Vérifier si le traitement doit être interrompu après chaque lot
        if interrupted:
            logger.warning(f"Traitement interrompu après la génération de la carte de profondeur de {img_name}")
            return img_path, False
        
        # Créer un masque d'inpainting
        logger.debug(f"Création du masque d'inpainting pour {img_name}")
        # Convertir le tenseur en numpy une seule fois et sur CPU pour économiser la mémoire GPU
        #seg_np = segmentation.cpu().numpy()
        seg_np = segmentation.numpy() if isinstance(segmentation, torch.Tensor) else segmentation
        # Libérer le tenseur GPU dès qu'il est converti en numpy
        del segmentation
        
        inpainting_mask = get_inpainting_mask(seg_np, resized_image)  # Utiliser l'image originale redimensionnée
        inpainting_mask.save(os.path.join(img_save_dir, f"{base_name}_inpainting_mask.png"))
        
        # Vérifier si le traitement doit être interrompu
        if interrupted:
            logger.warning(f"Traitement interrompu après la création du masque d'inpainting de {img_name}")
            return img_path, False
        
        # Générer une pièce vide en utilisant le masque d'inpainting
        logger.debug(f"Génération de la pièce vide pour {img_name}")
        empty_room_image = get_empty_room(resized_image, inpainting_mask)  # Utiliser l'image originale redimensionnée
        empty_room_image.save(os.path.join(img_save_dir, f"{base_name}_empty_room.png"))
        
        # Libérer la mémoire du masque d'inpainting et de la pièce vide dès qu'ils sont sauvegardés
        del inpainting_mask, seg_np, empty_room_image, original_image
        
        elapsed_time = time.time() - start_time
        logger.info(f"Traitement terminé pour {img_name} en {elapsed_time:.2f} secondes")
        
        # Libération explicite des objets volumineux restants
        del colored_seg, resized_image
        
        return img_path, True
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {img_path} : {str(e)}")
        traceback.print_exc()
        return img_path, False
    
    finally:
        # Nettoyage de la mémoire
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Mémoire GPU libérée après traitement de {img_name}")

def batch_process_images(image_paths: List[str], save_dir: str, max_workers: int = 4, batch_size: int = 10) -> Dict[str, Any]:
    """
    Traite un lot d'images en parallèle en utilisant un ThreadPoolExecutor.
    
    Args:
        image_paths: Liste des chemins d'accès aux images à traiter
        save_dir: Répertoire de sauvegarde des résultats
        max_workers: Nombre maximum de workers en parallèle
        batch_size: Taille des lots d'images à traiter
        
    Returns:
        Dictionnaire contenant les statistiques de traitement
    """
    global interrupted
    
    start_time = time.time()
    total_images = len(image_paths)
    successful_images = 0
    failed_images = 0
    failed_paths = []
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Déterminer le nombre optimal de workers en fonction de la mémoire disponible
    if torch.cuda.is_available():
        # Obtenir la mémoire totale du GPU
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Obtenir la mémoire actuellement utilisée
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        
        # Calculer la mémoire libre (utiliser la mémoire allouée pour une estimation plus conservatrice)
        free_memory = total_memory - allocated_memory
        
        # Estimer la mémoire nécessaire par worker (valeur empirique)
        estimated_memory_per_worker = 2 * 1024 * 1024 * 1024  # 2 GB par worker
        
        # Calculer le nombre optimal de workers
        optimal_workers = max(1, int(free_memory / estimated_memory_per_worker))
        
        # Limiter le nombre de workers au maximum spécifié
        max_workers = min(max_workers, optimal_workers)
        
        logger.info(f"Mémoire GPU - Totale: {total_memory / 1e9:.2f} GB, "
                   f"Réservée: {reserved_memory / 1e9:.2f} GB, "
                   f"Allouée: {allocated_memory / 1e9:.2f} GB, "
                   f"Libre: {free_memory / 1e9:.2f} GB")
        logger.info(f"Utilisation de {max_workers} workers en parallèle (optimal pour la mémoire disponible)")
    
    # Traiter les images par lots pour éviter de surcharger la mémoire
    for i in range(0, len(image_paths), batch_size):
        # Vérifier si le traitement doit être interrompu
        if interrupted:
            logger.warning(f"Traitement interrompu avant le lot {i//batch_size + 1}")
            break
            
        batch = image_paths[i:i+batch_size]
        logger.info(f"Traitement du lot {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} ({len(batch)} images)")
        
        # Utiliser un ThreadPoolExecutor pour le traitement parallèle
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre les tâches
            future_to_path = {executor.submit(process_single_image, path, save_dir): path for path in batch}
            
            # Collecter les résultats au fur et à mesure qu'ils sont terminés
            for future in tqdm(as_completed(future_to_path), total=len(batch), desc="Progression"):
                path = future_to_path[future]
                try:
                    # Vérifier si le traitement doit être interrompu
                    if interrupted:
                        # Annuler les tâches restantes
                        for f in future_to_path:
                            f.cancel()
                        logger.warning("Traitement interrompu, annulation des tâches restantes")
                        break
                        
                    # Récupérer le résultat
                    img_path, success = future.result()
                    
                    if success:
                        successful_images += 1
                    else:
                        failed_images += 1
                        failed_paths.append(img_path)
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération du résultat pour {path}: {str(e)}")
                    failed_images += 1
                    failed_paths.append(path)
        
        # Vérifier si le traitement doit être interrompu après chaque lot
        if interrupted:
            logger.warning(f"Traitement interrompu après le lot {i//batch_size + 1}")
            break
            
        # Nettoyage forcé de la mémoire après chaque lot
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Mémoire GPU libérée après le traitement du lot {i//batch_size + 1}")
            
            # Afficher l'état actuel de la mémoire
            reserved_memory = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory_allocated = total_memory - allocated_memory
            free_memory_reserved = total_memory - reserved_memory
            
            # Log détaillé de l'état de la mémoire
            logger.info(f"Mémoire GPU après lot - Réservée: {reserved_memory / 1e9:.2f} GB, "
                       f"Allouée: {allocated_memory / 1e9:.2f} GB, Libre (réservée): {free_memory_reserved / 1e9:.2f} GB, "
                       f"Libre (allouée): {free_memory_allocated / 1e9:.2f} GB")
            
            # Recalculer le nombre optimal de workers pour le prochain lot
            optimal_workers = max(1, int(free_memory_allocated / estimated_memory_per_worker))
            new_max_workers = min(max_workers, optimal_workers)
            
            if new_max_workers != max_workers:
                logger.info(f"Ajustement du nombre de workers: {max_workers} -> {new_max_workers} pour le prochain lot")
                max_workers = new_max_workers
                
    total_time = time.time() - start_time
    
    # Statistiques finales
    stats = {
        "total_images": total_images,
        "successful_images": successful_images,
        "failed_images": failed_images,
        "failed_paths": failed_paths,
        "total_time": total_time
    }
    
    if interrupted:
        logger.warning("Traitement interrompu par l'utilisateur après avoir traité "
                      f"{successful_images + failed_images}/{total_images} images.")
    else:
        logger.info(f"Traitement terminé en {total_time:.2f} secondes")
        logger.info(f"Images traitées avec succès : {successful_images}/{total_images}")
    
    if failed_images > 0:
        logger.warning(f"Images en échec : {failed_images}/{total_images}")
        for path in failed_paths:
            logger.warning(f"  - {path}")
    
    # Nettoyage final de la mémoire
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Nettoyage final de la mémoire GPU effectué")
    
    return stats

if __name__ == "__main__":
    try:
        # Récupérer tous les fichiers d'images dans le répertoire d'entrée
        input_dir = "inputs"
        output_dir = "outputs"
        
        # Vérifier si le répertoire d'entrée existe
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            logger.warning(f"Le répertoire d'entrée '{input_dir}' n'existait pas et a été créé.")
            logger.warning(f"Veuillez placer vos images dans le répertoire '{input_dir}' et relancer le script.")
            sys.exit(0)
        
        # Récupérer les chemins des images
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(input_dir, ext)))
            
        # Vérifier s'il y a des images à traiter
        if not image_paths:
            logger.warning(f"Aucune image trouvée dans le répertoire '{input_dir}'.")
            logger.warning(f"Formats supportés : {', '.join(image_extensions)}")
            sys.exit(0)
            
        # Traiter les images par lots
        logger.info(f"Traitement de {len(image_paths)} images...")
        stats = batch_process_images(image_paths, output_dir)
        
        # Afficher un résumé final
        if stats["failed_images"] == 0:
            logger.info("Toutes les images ont été traitées avec succès !")
        else:
            logger.warning(f"{stats['failed_images']} images n'ont pas pu être traitées.")
            
    except KeyboardInterrupt:
        # Cette partie sera exécutée si l'utilisateur appuie sur Ctrl+C en dehors des fonctions gérées
        logger.warning("Interruption manuelle du script (Ctrl+C).")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue : {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Nettoyage final, quoi qu'il arrive
        logger.info("Nettoyage final des ressources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Script terminé.")