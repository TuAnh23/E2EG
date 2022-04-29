
dataset=$1
subset=$2
if [ ${dataset} != "ogbn-arxiv" ] && [ ${dataset} != "ogbn-products" ]; then
    echo "dataset=${dataset} is not yet supported!"
    exit
fi

data_root_dir=./data
multi_task_data_dir=./data/proc_data_multi_task
max_degree=1000

python -u proc_data_multi_task.py \
    --raw-text-path ${multi_task_data_dir}/${dataset}/X.all.txt \
    --vectorizer-config-path ${multi_task_data_dir}/vect_config.json \
    --data-root-dir ${data_root_dir} \
    --multi-task-data-dir ${multi_task_data_dir} \
    --dataset ${dataset} \
    --subset "${subset}" \
    --max-deg ${max_degree}

