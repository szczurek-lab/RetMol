# RetMol with Jointformer backbone

## Setup the environment

One can run RetMol with Jointformer using the original PyTorch version by using the same image as for RetMol with Chemformer [**from here**](https://gitlab.com/havaker/retmol). Additionally required packages can be installed using an overlay image, I have created one and stored it in the Lustre folder (/lustre/groups/aih/jointformer/retmol_container/retmold_overlay.img). The command to run shell inside the container would be 
'apptainer shell --nv --overlay /path/to/overlay.img /path/to/container/image.sif'

A more modern environment for RetMol is obtained by using the image provided at 'havaker/retmol:jointformer-pipenv'. A built image can be found in the lustre folder. In the same folder is also a script to pull docker images on Helmholtz cluster, one just needs to change the name and location of where the image should be pulled from accordingly.  

Before running any code, it is import to set the path to the tokenizer configuration file in 'RetMol/MolBART/csv_data_retrieval_jointformer.py' correctly.

## Training RetMol with Jointformer

The script which contains the training is 'RetMol/MolBART/train_retrieval_jointformer.py'.
Before training a path to the model configuration, tokenizer configuration and to the jointformer checkpoint have to be inserted in the corresponding part of the script. Some variables like the path to the checkpoints saved during training can be changed in 'RetMol/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism/megatron/global_vars.py'. The training settings are the same as for RetMol currently, but if there are OOM errors during training on Helmholtz cluster, it helped a lot to shrink the batch size.


## Inference with Jointformer

To run the guacamol experiments with jointformer, it is first necessary to add the paths of the model checkpoints and tokenizer configuration in 'RetMol/guacamol/run_retrieval_ga_jointformer.py'.  