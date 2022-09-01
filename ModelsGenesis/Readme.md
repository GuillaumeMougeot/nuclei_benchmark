# Read me to use Models Genesis

Please cite the following paper when using <font color='red'>**Models Genesis**</font>
Zongwei Zhou, Vatsal Sodha, Jiaxuan Pang, Michael B. Gotway, and Jianming Liang
https://arxiv.org/abs/2004.07882

If you have questions or suggestions, feel free to open an issue at 
https://github.com/MrGiovanni/ModelsGenesis.git

# Pré-entrainement de Models Genesis
Pour pré-entrainé Models Genesis sur votre dataset, vous devez au préalable le diviser en données d'entrainement et en données de validation.
La commande suivante exécute le pré-entrainement et ne dépend pas du format de vos images (nifti ou tiff).
``./run_pretraining.sh trainset_path validset_path``
**trainset_path** : Chemin vers le répertoire des images utilisées pour l'entrainement.
**validset_path** : Chemin vers le répertoire des images utilisées pour la validation.

# Ré-entrainement
Une fois que le pré-entrainement est terminé, vous pouvez affiner votre modèle en le réentrainant. Nous avons obtenus de bons résultats en utilisant nnUNet (https://github.com/MIC-DKFZ/nnUNet) pour le ré-entrainement.
Placez-vous dans le dossier ``configure_docker_for_nnUNet/nnUNet`` puis suivez les instructions détaillées dans la documentation rédigée pour nnUNet.
Tout est fait pour que nnUNet reentraine le modèle en utilisant le modèle pré-entrainé généré par Models Genesis.