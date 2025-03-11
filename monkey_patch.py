import os
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monkey_patch")

def apply_patches():
    """Applique les patches nécessaires"""
    # Import huggingface_hub avant toute chose
    import huggingface_hub
    
    # Vérifier si cached_download existe déjà
    if not hasattr(huggingface_hub, "cached_download"):
        # Créer une fonction de remplacement
        def cached_download(*args, **kwargs):
            # Utiliser hf_hub_download comme substitut
            return huggingface_hub.hf_hub_download(*args, **kwargs)
        
        # Ajouter la fonction au module
        huggingface_hub.cached_download = cached_download
        logger.info("Patched: huggingface_hub.cached_download")
    
    # Créer le module constants s'il n'existe pas
    if not hasattr(huggingface_hub, "constants"):
        # Créer un nouveau module
        import types
        constants_module = types.ModuleType("constants")
        
        # Définir le répertoire de cache HF dans le dossier utilisateur
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        constants_module.HF_HOME = cache_dir
        constants_module.HUGGINGFACE_HUB_CACHE = cache_dir
        
        # Ajouter le module au package
        huggingface_hub.constants = constants_module
        logger.info("Patched: création de huggingface_hub.constants")
    else:
        # Vérifier si HF_HOME existe dans constants
        if not hasattr(huggingface_hub.constants, "HF_HOME"):
            # Créer une constante de remplacement basée sur le cache existant
            if hasattr(huggingface_hub.constants, "HUGGINGFACE_HUB_CACHE"):
                huggingface_hub.constants.HF_HOME = huggingface_hub.constants.HUGGINGFACE_HUB_CACHE
            else:
                # Fallback sur un répertoire par défaut
                huggingface_hub.constants.HF_HOME = os.path.expanduser("~/.cache/huggingface")
            logger.info("Patched: huggingface_hub.constants.HF_HOME")
    
    # Vérifier si model_info existe
    if not hasattr(huggingface_hub, "model_info"):
        # Créer une fonction de remplacement
        def model_info(*args, **kwargs):
            # Vérifier si HfApi existe
            if hasattr(huggingface_hub, "HfApi"):
                # Utiliser HfApi().model_info comme substitut si disponible
                return huggingface_hub.HfApi().model_info(*args, **kwargs)
            else:
                # Fonction factice qui ne fait rien
                logger.warning("model_info n'est pas disponible et ne peut pas être patché")
                return None
        
        # Ajouter la fonction au module
        huggingface_hub.model_info = model_info
        logger.info("Patched: huggingface_hub.model_info")
    
    # Patcher d'autres fonctions spécifiques si nécessaire
    # Pour diffusers/utils/dynamic_modules_utils.py
    
    logger.info("Tous les patches ont été appliqués avec succès")
    
    # Patcher directement le module diffusers.utils.dynamic_modules_utils
    try:
        from importlib.util import find_spec
        if find_spec("diffusers.utils.dynamic_modules_utils"):
            import diffusers.utils.dynamic_modules_utils as dmu
            
            # Remplacer les imports problématiques
            if not hasattr(dmu, "cached_download"):
                dmu.cached_download = huggingface_hub.cached_download
            if not hasattr(dmu, "hf_hub_download"):
                dmu.hf_hub_download = huggingface_hub.hf_hub_download
            if not hasattr(dmu, "model_info"):
                dmu.model_info = huggingface_hub.model_info
                
            logger.info("Patched: diffusers.utils.dynamic_modules_utils imports")
    except Exception as e:
        logger.warning(f"Erreur lors du patch de diffusers: {str(e)}")