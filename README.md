# RetMol with Jointformer backbone

Before setting up the enviromnent for RetMol, the necessary data sets and checkpoints have to be downloaded. Following the instructions from the [**original RetMol repository**](https://github.com/NVlabs/RetMol) and downloading data via scripts in the folder download_scripts does not quite work, as the files are too big to be downloaded, but one can still execute the scripts, which will output a link on terminal, via which one can manually download the data.
## Setting up RetMol
1. Download the image provided by Michal Sala as described in 
<br>
[**this GitLab repo**](https://gitlab.com/havaker/retmol).
In the folder /lustre/groups/aih/jointformer/retmol_container is one build of the image for the original RetMol environment ('retmol_old.sif'), as well as an overlay image for running the guacamol experiment and including the jointformer backbone with that env ('retmold_overlay.img'). 
Furthermore there is one build of an image for a RetMol env ('retmol_pipenv.sif') with more recent versions of the packages, namely PyTorch.

2. Once the image is downloaded, you can run a shell in the container with
'apptainer shell --nv /path/to/your/image.sif'
(the --nv flag is to ensure GPU support)
Then run '/bin/bash'

3. Apptainer, which is used on Helmholtz cluster will not automatically source the correct .bashrc file, so we have to do that manually. Go to /root/ and do 'source .bahsrc'

4. Then you should be in an enviromnent where you can run RetMol training and inference. For some experiments additional packages may be necessary, according to the above mentioned GitLab repo, we should first try installng them via conda, then pip (if the old version of RetMol is used). 

5. Batch jobs on the Helmholtz cluster which are supposed to run in an Apptainer environment can be split into two scripts, one which is executed with 'sbatch' and runs something like 
'apptainer exec --nv /path/to/your/container/image.sif /path/to/helper/script.slurm'
and a helper script which contains all the code which is supposed to be executed in the container enviromnent.