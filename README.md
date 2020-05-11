# deep_learning_project
Repository for NYU Deep Learning SP20 Final Project - Team 12

| Team Member Name | NetID  |
| ---------------- | ------ |
| William Huang    | wh629  |
| Haoyue Ping      | hp1326 |
| Bilal Munawar    | bm2515 |



The below are instructions to execute training and pretraining on NYU's Prince computer cluster.



# Setup

1. Clone github repository using command 

   ``git clone https://github.com/wh629/deep_learning_project.git``

2. Change current directory to ``/deep_learning_project``

3. Load modules to session using commands

   ``module purge
   module load anaconda3/5.3.1
   module load cuda/10.0.130
   module load gcc/6.3.0``

4. Set up ``DL`` environment with ``.yml`` using command

   ``conda env create -f environment.yml``

5. Create results directory in repository with command

   ``mkdir results``

6. Create data directory in repository with command

   ``mkdir data``

7. Load project data to data directory



# Execute Training

1. In repository, execute training with command

   ``sbatch train.sbatch``

2. Logs, weights, and results will appear in ``deep_learning_project/results`` upon completion

   * Run logs appear in ``results/log``
   * Weights appear in ``results/logged/<experiment name>/best.pt``
   * Results appear in ``results/results.jsonl``



# Execute Pre-training

1. In repository, execute pre-training with command

   ``sbatch pretrain.sbatch``

2. Logs and weights will appear in ``deep_learning_project/results`` upon completion

   * Run logs appear in ``results/log``
   * Weights appear in ``results/<experiment name>_best.pt``