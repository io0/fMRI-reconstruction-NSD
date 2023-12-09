#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.

pip install numpy matplotlib tqdm scikit-image jupyterlab
pip install conda-forge accelerate

pip install clip-retrieval webdataset clip pandas matplotlib ftfy regex kornia umap-learn
pip install dalle2-pytorch

pip install torchvision==0.15.2 torch==2.0.1
pip install diffusers==0.13.0

pip install info-nce-pytorch==0.1.0
pip install pytorch-msssim

mkdir ../train_logs/models
wget https://dl.fbaipublicfiles.com/vicregl/convnext_xlarge_alpha0.75_fullckpt.pth
wget https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/sd_image_var_autoenc.pth

mkdir -p nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf
cd nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf
for i in {0..40}
do
   # Add leading zeros to the session number
   session_number=$(printf "%02d" $i)
   
   wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/betas_session${session_number}.nii.gz
done