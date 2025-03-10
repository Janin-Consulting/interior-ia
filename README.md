# Interior-IA : Traitement d'images d'intérieur

Ce projet fournit un ensemble d'outils pour l'analyse et le traitement d'images d'intérieur de maisons, en utilisant des modèles d'intelligence artificielle avancés.

## Fonctionnalités principales

- **Segmentation sémantique** : Identification et classification des objets dans les images d'intérieur en utilisant OneFormer avec le dataset ADE20K
- **Génération de cartes de profondeur** : Estimation de la profondeur à partir d'images 2D avec le modèle Depth Anything
- **Création de masques d'inpainting** : Génération automatique de masques pour supprimer les meubles et objets des pièces
- **Génération de pièces vides** : Suppression des meubles et objets pour créer des versions vides des pièces

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/interior-ia.git
cd interior-ia

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Traitement d'une seule image

Pour traiter une seule image et générer tous les résultats (segmentation, overlay, carte de profondeur et masque d'inpainting) :

```bash
python main.py --image chemin/vers/votre/image.jpg
```

### Traitement par lots

Pour traiter plusieurs images en parallèle :

```bash
python main.py --input_dir chemin/vers/dossier/images --output_dir outputs --max_workers 4 --batch_size 10
```

Paramètres optionnels :
- `--max_workers` : Nombre maximum de threads parallèles (par défaut : 4)
- `--batch_size` : Nombre d'images à traiter par lot pour économiser la mémoire (par défaut : 10)

## Structure du projet

- **main.py** : Script principal pour le traitement d'images et la gestion du flux de travail
- **segmentation.py** : Fonctions pour la segmentation d'images avec OneFormer
- **depth.py** : Génération de cartes de profondeur avec Depth Anything
- **inpainting_mask.py** : Création de masques pour l'inpainting basés sur la segmentation
- **empty_room.py** : Génération de pièces vides en supprimant les meubles
- **segmentation_colors.py** : Mappages des couleurs pour la segmentation ADE20K

## Résultats

Pour chaque image traitée, un dossier portant le nom de l'image est créé dans le répertoire `outputs/` avec les fichiers suivants :

- **{nom_image}_segmentation.png** : Image segmentée colorée selon les classes ADE20K
- **{nom_image}_overlay.png** : Superposition de la segmentation sur l'image originale
- **{nom_image}_depth.png** : Carte de profondeur en niveaux de gris
- **{nom_image}_inpainting_mask.png** : Masque pour l'inpainting (blanc pour les zones à remplacer)
- **{nom_image}_empty_room.png** : Version de la pièce sans meubles générée par inpainting

## Modèles utilisés

### Segmentation

Le projet utilise le modèle OneFormer pré-entraîné sur le dataset ADE20K pour la segmentation sémantique :
- **Modèle** : `shi-labs/oneformer_ade20k_swin_large`
- **Classes** : Plus de 150 classes d'objets adaptées aux scènes d'intérieur

### Estimation de profondeur

Pour l'estimation de profondeur, le projet utilise le modèle Depth Anything :
- **Modèle** : `LiheYoung/depth-anything-large-hf`
- **Caractéristiques** : Estimation de profondeur monoculaire de haute qualité

### Inpainting

Pour la génération de pièces vides, le projet utilise un modèle d'inpainting :
- **Modèle** : `lykon/absolute-reality-1.6525-inpainting`
- **Caractéristiques** : Inpainting guidé par prompts pour remplacer les meubles par des surfaces vides

## Optimisations

Le projet inclut plusieurs optimisations pour améliorer les performances :

### Optimisations de traitement
- **Traitement parallèle** : Utilisation de `ThreadPoolExecutor` pour le traitement par lots
- **Optimisation GPU** : Conversion des modèles en demi-précision sur GPU

### Optimisations de mémoire
- **Traitement par lots** : Division des grandes collections d'images en lots plus petits pour éviter la surcharge mémoire
- **Ajustement dynamique des workers** : Adaptation automatique du nombre de workers en fonction de la mémoire GPU disponible
- **Libération progressive des ressources** : Suppression des objets volumineux dès qu'ils ne sont plus nécessaires
- **Utilisation de torch.no_grad()** : Réduction de l'empreinte mémoire pendant l'inférence
- **Nettoyage forcé entre les lots** : Appels explicites à `gc.collect()` et `torch.cuda.empty_cache()`
- **Surveillance de la mémoire** : Journalisation de l'utilisation de la mémoire GPU pour identifier les fuites

Ces optimisations permettent de traiter efficacement de grands ensembles d'images, même sur des machines avec des ressources GPU limitées.

## Dépendances principales

- **torch** et **torchvision** : Pour les modèles d'apprentissage profond
- **transformers** : Pour les modèles pré-entraînés de Hugging Face
- **diffusers** : Pour les modèles de diffusion utilisés dans l'inpainting
- **PIL** et **opencv-python** : Pour le traitement d'images
- **numpy** et **scipy** : Pour les calculs numériques

Pour la liste complète des dépendances, voir le fichier `requirements.txt`.

## Performances et limitations

- Les modèles de segmentation fonctionnent mieux sur des images bien éclairées avec des objets clairement visibles
- L'estimation de profondeur est relative (pas d'échelle métrique absolue) mais précise pour les relations spatiales
- L'inpainting peut parfois créer des artefacts visuels, surtout dans les zones complexes
- Les temps de traitement dépendent de la taille des images et des ressources matérielles disponibles
- **Utilisation de la mémoire** : Avec les optimisations implémentées, le projet peut traiter efficacement des lots importants d'images, mais la mémoire GPU reste un facteur limitant pour les très grands ensembles

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
