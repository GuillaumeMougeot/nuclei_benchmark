# Installation et utilisation de nnUNet
*Please cite the following paper when using nnUNet :
Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nat Methods (2020).*
[https://www.nature.com/articles/s41592-020-01008-z](https://www.nature.com/articles/s41592-020-01008-z) 

Je détaille ici comment j'ai configuré nnUNet pour pouvoir l’entraîner sur nos images de noyaux.
Vous pouvez trouver plus de détails en suivant le lien suivant [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) 

## Installation

Vous pouvez installer nnUNet localement ou le configurer en utilisant docker. Pour cela il faudra :

**Installer une version de Pytorch antérieure à la version 1.6**
Installer ensuite la version de nnUNet qui permet le ré-entraîneement. 
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

##  Configuration
Avant de commencer, il est important d'avoir le répertoire de travail sous le format adapté à nnUNet

![Image de la configuration du répertoire de travail.](/images/directories_to_make.png)
Il faudra donc avoir la meme structure que sur l'image au dessus
Vous devez aussi modifier les variables img_folder (dossier contenant les images) et msk_folder (dossier contenant les masks) pour les adapter à votre configuration.
Définir quelques variables d'environnement
	- export nnUNet_raw_data_base='nnUNet_raw_data_base'
	- export nnUNet_preprocessed='nnUNet_preprocessed'
	- export RESULTS_FOLDER='nnUNet_trained_models'
* Ces variables d'environnement sont définies en se basant sur la mème structure présentée dans limage plus haut.
* Le chiffre 500 dans le nom du dossier 'Task500_Nucleus' rreprésente la variable appelée TaskID. Dans le cas où vous utilisez vos propres images, sa valeur doit ètre supérieure ou égale à 500 pour éviter la confusion avec les identifiants des données utilisées pour les modèles pré-entraînées.

## Utilisation
Si vous utilisez docker, il faudra ajouter l'option ``- - ipc = host`` dans la commande run de docker :
<font color='red'>``docker run --gpus all -it -v /home/nanaa/nnUNet/Notebooks:/notebooks/ -p 8890:8888 --ipc=host modelgenesis``</font>
Si cette option est omise vous aurez l'erreur ci-dessous ou quelque chose de similaire.
![Erreur](/images/eror.png)

### Entraînement du modèle et prédiction
#### Preprocessing
Le code pour passer nos images au format adapté à nnUNet et les repartir en données de train et de test est fourni.

Dans un premier temps, nnU-Net extrait une empreinte d'ensemble de données (un ensemble de propriétés spécifiques à l'ensemble de données telles que les tailles d'image, les espacements de pixels, les informations d'intensité, etc). Pour réaliser cette étape, tapez la commande suivante : <font color='red'>``nnUNet_plan_and_preprocess -t xxx --verify_dataset_integrity``</font> avec 500 le TaskID
#### Entrainement
Pour lancer l'entrainement du modèle tapez  les commandes suivantes (avec xxx le TaskID) :
avec xxx le TaskID.
<font color='red'>``nnUNet_train 3d_fullres nnUNetTrainerV2 500 0 --npz``</font>  
<font color='red'>``nnUNet_train 3d_fullres nnUNetTrainerV2 500 1 --npz``</font>  
<font color='red'>``nnUNet_train 3d_fullres nnUNetTrainerV2 500 2 --npz``</font>  
<font color='red'>``nnUNet_train 3d_fullres nnUNetTrainerV2 500 3 --npz``</font>  
<font color='red'>``nnUNet_train 3d_fullres nnUNetTrainerV2 500 4 --npz``</font>  

Dans certains cas,  il sera nécessaire de supprimer tous les fichiers .npy générés par la commande lancée lors du preprocessing pour que l'entraînement marche. Vous pouvez les supprimer en tapant, avant de lancer l'entraînement, la commande suivante : 
 <font color='red'>``rm -f nnUNet_preprocessed/Task500_Nucleus/nnUNetData_plans_v2.1_stage0/*.npy``</font> sous linux. Si dans votre cas vous avez dù passer par là, il faudra préciser l'option  <font color='red'>``--use_compressed``</font>  à la fin de la commande pour lancer l'entraînement.
 
#### Prédiction

Pour tester le modèle ainsi entrainé sur les images de test, il faut lancer une prédiction pour chaque valeur de fold.

<font color='red'>``nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Nucleus/imagesTs/ -o output_for_fold_0 -t 500 -tr nnUNetTrainerV2 -m 3d_fullres -f 0'``</font>  
<font color='red'>``nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Nucleus/imagesTs/ -o output_for_fold_1 -t 500 -tr nnUNetTrainerV2 -m 3d_fullres -f 0'``</font>  
<font color='red'>``nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Nucleus/imagesTs/ -o output_for_fold_2 -t 500 -tr nnUNetTrainerV2 -m 3d_fullres -f 0'``</font>   
<font color='red'>``nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Nucleus/imagesTs/ -o output_for_fold_3 -t 500 -tr nnUNetTrainerV2 -m 3d_fullres -f 0'``</font>   
<font color='red'>``nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Nucleus/imagesTs/ -o output_for_fold_4 -t 500 -tr nnUNetTrainerV2 -m 3d_fullres -f 0'``</font>     

Ensuite on lance une autre prédiction qui effectuer une prédiction en moyennant les prédictions précédentes.

<font color='red'>``nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Nucleus/imagesTs/ -o output_directory -t 500 -tr nnUNetTrainerV2 -m 3d_fullres``</font>  

##### Remarque
Il est possible d'augmenter ou de diminuer le nombre d'époques.
Pour cela il faut créer une nouvelle classe quui hérite de la classe nnUNetTrainerV2 et indiquer dans cette nouvelle classe le nombre d'époques qu'on souhaite. Il suffira ensuite lors de l'entrainement et de la prédiction de lancer les commandes en remplaçant nnUNetTrainerV2 par le nom de votre classe.
