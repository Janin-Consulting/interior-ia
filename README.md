# Interior-IA : Traitement d'images d'intérieur

Ce projet fournit un ensemble d'outils pour la génération et l'amélioration d'images d'intérieur de maisons, en utilisant des modèles d'intelligence artificielle avancés.

## Fonctionnalités principales

- **Génération d'intérieurs meublés** : Transformation de pièces vides en intérieurs meublés avec ControlNet et modèles de diffusion
- **Segmentation sémantique** : Utilisée pour identifier les zones à meubler dans les images d'intérieur
- **Détection de lignes structurelles** : Préservation de l'architecture de la pièce lors de la génération
- **Upscaling haute résolution** : Amélioration de la qualité des images générées avec un upscaling x4 optimisé

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/interior-ia.git
cd interior-ia

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Génération d'images d'intérieur

Pour générer une image d'intérieur meublé à partir d'une pièce vide :

```bash
python predict.py --image empty_room_input/02-salon-sans-logo.png --output result.png --prompt "living room, modern style"
```

### Upscaling d'images

Pour améliorer la résolution et la qualité d'une image générée :

```bash
python upscale.py --image result.png --output result-upscale.png --type interior
```

### Configuration personnalisée

Chaque module dispose d'un dictionnaire `DEFAULT_CONFIG` pour personnaliser :
- Les paramètres des modèles d'IA (nombre d'étapes d'inférence, guidance scale, etc.)
- Les stratégies d'optimisation selon le matériel disponible
- Les préférences de qualité et de performance

## Structure du projet

- **predict.py** : Module principal pour la génération d'images d'intérieur meublées
- **upscale.py** : Module d'amélioration de la résolution des images avec traitement par tuiles
- **colors.py** : Palettes de couleurs et mappages pour la segmentation ADE20K
- **utils.py** : Fonctions utilitaires pour le traitement d'images et la conversion de formats
- **monkey_patch.py** : Correctifs pour assurer la compatibilité entre les bibliothèques
- **palette.py** : Définition des palettes de couleurs pour la visualisation

## Architecture du code

Le projet suit une architecture modulaire avec les caractéristiques suivantes :

- **Séparation des responsabilités** : Chaque module gère une fonctionnalité spécifique
- **Configuration par défaut** : Dictionnaires `DEFAULT_CONFIG` pour une personnalisation facile
- **Gestion intelligente des ressources** : Optimisation de l'utilisation mémoire et calcul GPU/CPU
- **Vectorisation numpy** : Utilisation intensive d'opérations vectorisées pour améliorer les performances
- **Adaptation GPU/CPU** : Détection automatique et optimisation selon le matériel disponible

## Optimisations récentes

Le projet a été récemment optimisé pour améliorer les performances et la maintenance :

### Améliorations techniques
- **Configuration flexible** : Introduction de dictionnaires DEFAULT_CONFIG remplaçant les variables globales
- **Gestion des seeds** : Solution pour le problème "manual_seed expected a long, but got NoneType"
- **Support CPU/GPU flexible** : Meilleure adaptation à différents types de matériel
- **Modèle spécialisé** : Utilisation de modèles optimisés pour le design d'intérieur

### Documentation et maintenabilité
- **Commentaires détaillés** : Documentation en français expliquant chaque étape des processus
- **Variables explicites** : Renommage pour une meilleure compréhension du code
- **Gestion d'erreurs améliorée** : Meilleure robustesse face aux différents types d'entrées

## Modèles utilisés

### Génération d'images
- **Modèle** : `SG161222/Realistic_Vision_V3.0_VAE` avec ControlNet pour la segmentation et les lignes
- **ControlNet** : `BertChristiaens/controlnet-seg-room` et `lllyasviel/sd-controlnet-mlsd`

### Segmentation
- **Modèle** : `nvidia/segformer-b5-finetuned-ade-640-640` pour la segmentation sémantique

### Upscaling
- **Modèle** : `stabilityai/stable-diffusion-x4-upscaler` pour l'amélioration de la résolution

## Priorités de développement

1. **Conversion ONNX** : Exportation des modèles vers le format ONNX pour une inférence optimisée
2. **Pipeline asynchrone** : Implémentation d'une architecture asynchrone pour un meilleur parallélisme
3. **Vectorisation numpy** : Poursuite de l'optimisation des opérations avec numpy

## Dépendances principales

- **torch** et **torchvision** : Pour les modèles d'apprentissage profond
- **transformers** : Pour les modèles pré-entraînés de Hugging Face
- **diffusers** : Pour les modèles de diffusion utilisés dans la génération d'images
- **controlnet-aux** : Pour les modèles ControlNet auxiliaires
- **PIL** et **opencv-python** : Pour le traitement d'images
- **numpy** : Pour les calculs numériques vectorisés

Pour la liste complète des dépendances, voir le fichier `requirements.txt`.

## Limitations connues

- Les performances dépendent des ressources matérielles disponibles (GPU recommandé)
- Le temps de traitement varie selon la taille et la complexité des images
- Les résultats peuvent varier en fonction des prompts utilisés et du contexte de l'image

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
