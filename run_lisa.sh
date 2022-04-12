#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --job-name=ExampleJob
#SBATCH --time=00-12:00:00
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
export WANDB_DIR=$HOME
experiment_name=freezing_scheme_sweep
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
# Copy data folder
cp -r data ${tmp_dir}
# Copy model folder if it already exists
if [ -d "models/${experiment_name}" ]
then
  mkdir ${tmp_dir}/models
  cp -r models/${experiment_name} ${tmp_dir}/models
fi
# Copy experiment folder if it already exists
if [ -d "experiments/${experiment_name}" ]
then
  mkdir ${tmp_dir}/experiments
  cp -r experiments/${experiment_name} ${tmp_dir}/experiments
fi
# Copy cache folder if it already exists
if [ -d "models/cache" ]
then
  cp -r models/cache ${tmp_dir}/models
fi
data_dir=${tmp_dir}/data/proc_data_multi_task/${dataset}
model_dir=${tmp_dir}/models/${experiment_name}
experiment_dir=${tmp_dir}/experiments/${experiment_name}
cache_dir=${tmp_dir}/models/cache
runs=5
# No matter what happens, we copy the temp output folders back to our login node
trap 'cp -r ${model_dir} $HOME/UvA_Thesis_pecosEXT/models; cp -r ${experiment_dir} $HOME/UvA_Thesis_pecosEXT/experiments; cp -r ${cache_dir} $HOME/UvA_Thesis_pecosEXT/models;' EXIT
# Run train-val-test pipeline
params_path=data/proc_data_multi_task/params_mtask_${dataset}.json
#bash multi_task_pipeline.sh ${data_dir} ${model_dir} ${experiment_dir} ${cache_dir} ${params_path} ${runs}
bash hyperparams_sweep.sh ${data_dir} ${model_dir} ${experiment_dir} ${cache_dir} ${params_path} sweep_configs/${experiment_name}