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
    
    # Patch pour les avertissements de timm
    try:
        # Vérifier si timm est installé
        if find_spec("timm"):
            # Importer les modules nécessaires de timm d'abord
            # Cela permet de s'assurer que les imports se font dans le bon ordre
            import timm.layers
            import timm.models
            
            # Patcher les imports dépréciés pour rediriger vers les bons chemins
            import sys
            
            # Créer un module fantôme timm.models.layers qui redirige vers timm.layers
            if "timm.models.layers" in sys.modules:
                # Si le module est déjà importé, on le met à jour
                sys.modules["timm.models.layers"].__dict__.update(timm.layers.__dict__)
            else:
                # Sinon on crée un nouveau module qui est simplement une référence à timm.layers
                import types
                layers_module = types.ModuleType("timm.models.layers")
                layers_module.__dict__.update(timm.layers.__dict__)
                sys.modules["timm.models.layers"] = layers_module
            
            # Même chose pour timm.models.registry
            if "timm.models.registry" in sys.modules:
                sys.modules["timm.models.registry"].__dict__.update(timm.models.__dict__)
            else:
                import types
                registry_module = types.ModuleType("timm.models.registry")
                registry_module.__dict__.update(timm.models.__dict__)
                sys.modules["timm.models.registry"] = registry_module
            
            logger.info("Patched: timm imports dépréciés redirigés vers les nouveaux modules")
    except Exception as e:
        logger.warning(f"Erreur lors du patch de timm: {str(e)}")
    
    # Patch pour les avertissements de controlnet_aux et les conflits de registre
    try:
        # Vérifier si controlnet_aux est installé
        if find_spec("controlnet_aux"):
            # Importer les modules nécessaires
            import controlnet_aux
            
            # Patcher la méthode de register_model pour éviter les warnings de réécriture
            # Ce patch doit être appliqué avant que controlnet_aux ne soit importé
            import timm.models as timm_models
            
            if hasattr(timm_models, "register_model"):
                # Sauvegarder la fonction originale
                original_register_model = timm_models.register_model
                
                # Créer une fonction wrapper qui ignore silencieusement les réécritures
                def patched_register_model(fn_or_name):
                    if callable(fn_or_name):
                        # Cas normal d'utilisation comme décorateur
                        model_name = fn_or_name.__name__
                        if model_name in timm_models._model_entrypoints:
                            # Le modèle existe déjà, on le retourne simplement sans warning
                            return fn_or_name
                        return original_register_model(fn_or_name)
                    else:
                        # Cas d'utilisation avec un nom spécifié
                        def wrap_register_model(fn):
                            if fn_or_name in timm_models._model_entrypoints:
                                # Le modèle existe déjà, on le retourne simplement sans warning
                                return fn
                            return original_register_model(fn_or_name)(fn)
                        return wrap_register_model
                
                # Remplacer la fonction originale par notre version patchée
                timm_models.register_model = patched_register_model
                logger.info("Patched: timm.models.register_model pour éviter les conflits de registre")

                # En plus du patch général, on applique un patch spécifique pour segment_anything
                try:
                    # Précharger tous les modèles problématiques pour éviter les warnings
                    # Ceci va enregistrer les modèles avant que controlnet_aux ne tente de les réenregistrer
                    import timm
                    _ = timm.create_model("tiny_vit_5m_224", pretrained=False)
                    _ = timm.create_model("tiny_vit_11m_224", pretrained=False)
                    _ = timm.create_model("tiny_vit_21m_224", pretrained=False)
                    _ = timm.create_model("tiny_vit_21m_384", pretrained=False)
                    _ = timm.create_model("tiny_vit_21m_512", pretrained=False)
                    logger.info("Préchargé les modèles tiny_vit pour éviter les conflits avec SAM")
                except Exception as e:
                    logger.warning(f"Erreur lors du préchargement des modèles tiny_vit: {str(e)}")
                
                # Patch plus agressif directement sur UserWarning pour supprimer les derniers warnings
                import warnings
                
                # Sauvegarder la fonction d'avertissement originale
                original_warn = warnings.warn
                
                # Créer une fonction wrapper qui filtre les avertissements spécifiques
                def filtered_warn(message, *args, **kwargs):
                    # Filtrer les avertissements spécifiques liés aux conflits de registre
                    if isinstance(message, str) and "Overwriting" in message and "in registry" in message:
                        # Supprimer silencieusement ces warnings
                        return
                    # Pour tous les autres warnings, utiliser la fonction originale
                    return original_warn(message, *args, **kwargs)
                
                # Remplacer la fonction originale par notre version filtrée
                warnings.warn = filtered_warn
                logger.info("Patched: warnings.warn pour filtrer les messages de conflit de registre")
    except Exception as e:
        logger.warning(f"Erreur lors du patch de controlnet_aux: {str(e)}")
    
    logger.info("Tous les patches ont été appliqués avec succès")