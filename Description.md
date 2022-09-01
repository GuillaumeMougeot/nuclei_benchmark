
# Description du repertoire de travail
Ce fichier est utile pour savoir se retrouver dans mon répertoire de travail. 
Tous les dossiers et fichiers importants sont dans ``/mnt/sdb2/Adama/``.
Les dossiers suivant contiennent les modèles.
``ModelsGenesis_on_Task502_Lung``   
``ModelsGenesis_on_Task503_Pancreas``   
``TansVW_on_Task502_Lung``  
``TansVW_on_Task503_Pancreas``  

Le dossier ``conversion`` conient les scripts qui permettent la conversion des images de nifti à tiff. 

Les  ``configure_docker_for_modelsgenesis``  contient l'emballage de ModelsGenesis. Il ya une version plus sobre sur github.  
Le sossier ``configure_docker_for_transvw`` contient tout les scripts que j'ai écrit pour réentrainer le modèle préeentrainé de TransVW. TransVW n'a pas été conteneurisé.  

Le dossier  ``nnUNet_trained_models`` contient le modèle entrainé de nnUNet.  

La conteneurisation de nnUNet est dans le dossier ``nuclei_benchmark/nnUNet/configure_docker_for_nnUNet/``