#conda create -n ampseminer python=3.10
#conda activate ampseminer
#module load cuda/12.1
#export LD_LIBRARY_PATH=/home/lwh/miniconda3/envs/ampseminer/lib:$LD_LIBRARY_PATH

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge tensorboard -y
pip install git+https://github.com/facebookresearch/esm.git && echo esm installed!
conda install pyg -c pyg -y && echo pyg installed!
conda install -c anaconda scikit-learn -y && echo scikit-learn installed!
conda install -c anaconda scipy -y && echo scipy installed!
pip install git+https://github.com/aqlaboratory/openfold.git && echo openfold installed!
pip install git+https://github.com/huggingface/transformers && echo hf_transformers installed!
conda install -c anaconda ipykernel -y && echo ipykernel installed!
conda install -c conda-forge einops -y && echo einops installed!
pip install dm-tree
conda install -c anaconda cudnn -y
conda install -c conda-forge biopython -y
conda install -c conda-forge modelcif -y
conda install -c conda-forge ml-collections -y
conda install -c conda-forge omegaconf -y
conda install -c bioconda bioawk -y
conda install -c conda-forge biotite -y
conda install -c anaconda pandas -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda seaborn -y
conda install -c bioconda samtools -y
conda install -c bioconda seqtk -y
conda install -c bioconda seqkit -y
conda install -c bioconda mafft -y