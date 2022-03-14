#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --job-name=ExampleJob
#SBATCH --time=04-00:00:00
#SBATCH --mem=48000M
#SBATCH --output=slurm_output_%A.out

nvidia-smi

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
# Activate your environment
source activate giant-xrt
which python
# Run your code
# Download data
cd data/proc_data_multi_task
dataset=ogbn-arxiv
bash download_data.sh ${dataset}
cd ../../
# Process data
bash proc_data_multi_task.sh ${dataset}
# Move files to scratch node
echo "Moving data to scratch..."
tmp_dir="$TMPDIR"/tuanh_scratch
mkdir ${tmp_dir}
cp -r data ${tmp_dir}
data_dir=${tmp_dir}/data/proc_data_multi_task/${dataset}
model_dir=${tmp_dir}/models/multi_task_models
experiment_dir=${tmp_dir}/experiments/multi_task_models
# No matter what happens, we copy the temp output folders back to our login node
trap 'cp -r ${model_dir} $HOME/UvA_Thesis_pecosEXT/models; cp -r ${experiment_dir} $HOME/UvA_Thesis_pecosEXT/experiments;' EXIT
# Run train-val-test pipeline
params_path=data/proc_data_multi_task/params_mtask_${dataset}.json
bash multi_task_train.sh ${data_dir} ${model_dir} ${experiment_dir} ${params_path}
