# E2EG: End-to-end node classification using graph topology and text-based node attribute

Master thesis conducted at the [University of Amsterdam](https://www.uva.nl/en), in collaboration with [Socialdatabase](https://www.socialdatabase.com/).

The repository is derived from [PECOS](https://github.com/amzn/pecos).

## 1. Requirements and Installation

* Python (==3.8)
* Pip (>=19.3)

Create a conda environment:
```bash
conda create -n "giant-xrt" python=3.8
conda activate giant-xrt
```

Install pytorch:

With cuda:
```bash
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
```
Without cuda:
```bash
conda install pytorch==1.9.0 -c pytorch
```
Install GNN related packages

With cuda:
```bash
ptcu_version="1.9.0+cu102"
```
Without cuda:
```bash
ptcu_version="1.9.0+cpu"
```

```bash
pip install torch-scatter -f "https://pytorch-geometric.com/whl/torch-${ptcu_version}.html"
pip install torch-sparse -f "https://pytorch-geometric.com/whl/torch-${ptcu_version}.html"
pip install torch-cluster -f "https://pytorch-geometric.com/whl/torch-${ptcu_version}.html"
pip install torch-spline-conv -f "https://pytorch-geometric.com/whl/torch-${ptcu_version}.html"
pip install torch-geometric
pip install ogb==1.3.2
```

Set up and install PECOS dependencies:
```bash
python3 -m pip install --editable ./
```

## 2. Reproduce results

### 2.1. Baseline models

#### 2.1.1. Train and evaluate graphSAGE on node degree:
```bash
dataset=ogbn-arxiv  # or ogbn-products
mkdir experiments/graphSAGE_nodedegree
python -u baseline_models/gnn.py \
  --use_sage \
  --node_feature degree \
  --dataset_name ${dataset} \
  --hidden_channels 50 | tee -a experiments/graphSAGE_nodedegree/train.log
```

#### 2.1.2. Train and evaluate transformer using only node text attribute:
```bash
# Download data
cd data/proc_data_multi_task
dataset=ogbn-arxiv  # or ogbn-products
bash download_data.sh ${dataset}
cd ../../
# Train model
mkdir experiments/bert_classifier
mkdir models/bert_classifier
source activate giant-xrt
python -u baseline_models/bert_classifier.py \
  --model_dir models/bert_classifier \
  --experiment_dir experiments/bert_classifier \
  --raw-text-path data/proc_data_multi_task/ogbn-arxiv/X.all.txt \
  --text_tokenizer_path data/proc_data_multi_task/ogbn-arxiv/xrt_models/text_encoder/text_tokenizer \
  --dataset ${dataset} \
  | tee -a experiments/bert_classifier/train.log
```

#### 2.1.3. Train and evaluate different settings for GIANT+MLP
```bash
# Download data
cd data/proc_data_multi_task
dataset=ogbn-arxiv  # or ogbn-products
bash download_data.sh ${dataset}
cd ../../
# Process data
source activate giant-xrt
bash proc_data_multi_task.sh ${dataset}
# Train model
experiment_name=giant_mlp
runs=10
data_dir=data/proc_data_multi_task/${dataset}
model_dir=models/${experiment_name}
experiment_dir=experiments/${experiment_name}
cache_dir=models/cache
params_path=data/proc_data_multi_task/params_xrt_${dataset}.json
bash GIANT_pipeline.sh ${data_dir} ${model_dir} ${experiment_dir} ${cache_dir} ${params_path} ${runs}
```

Change `GIANT_pipeline.sh` for different settings of GIANT+MLP:
- GIANT+MLP with BERT: set variable in line 89 to `--model-shortcut "bert-base-uncased"`
- GIANT+MLP with DistilBERT: set variable in line 89 to `--model-shortcut "distilbert-base-uncased"`
- GIANT+MLP with DistilRoberta SentenceBERT: set variable in line 89 to `--model-shortcut "sentence-transformers/all-distilroberta-v1"`
- GIANT+MLP in transductive setting: set variable in line 90 to `--include-Xval-Xtest-for-training "true"`
- GIANT+MLP in inductive setting: set variable in line 90 to `--include-Xval-Xtest-for-training "false"`

### 2.2. Proposed E2EG models
Note: in this repository, we make a clear distinction between `multi-label` classification tasks and `multi-class` classification tasks (whereas in real-life, the term "label" and "class" might be used interchangeably)

- `Multi-class` classification: each input will have only one output class. This is the case of the main node classification task.
- `Multi-label` classification: each input can have multiple output labels. This is the case of the auxiliary neighborhood prediction task.

The proposed model learns both the `Multi-class` classification task and the `Multi-label` classification task, and backpropagate the loss in tandem for both tasks

#### 2.2.1. Train and evaluate E2EG models
```bash
# Download data
cd data/proc_data_multi_task
dataset=ogbn-arxiv  # or ogbn-products
bash download_data.sh ${dataset}
cd ../../
# Process data
source activate giant-xrt
bash proc_data_multi_task.sh ${dataset}
# Train model
experiment_name=E2EG
runs=10
data_dir=data/proc_data_multi_task/${dataset}
model_dir=models/${experiment_name}
experiment_dir=experiments/${experiment_name}
cache_dir=models/cache
params_path=data/proc_data_multi_task/params_mtask_${dataset}.json
bash multi_task_pipeline_${dataset}.sh ${data_dir} ${model_dir} ${experiment_dir} ${cache_dir} ${params_path} ${runs}
```
No need to change anything if want to use best model setting.

Change `multi_task_pipeline_ogbn-arxiv.sh` or `multi_task_pipeline_ogbn-products.sh` for different settings of E2EG:
- Methods to delay learning the main task (only for `ogbn-arxiv`):
  - Freeze main classification head at early rounds: set variable in line 102 to `--freeze-mclass-head-range "0|2"`
  - Excluding main classification loss at early rounds: set variable in line 97 to `--weight-loss-strategy "include_mclass_loss_later_at_round_2"`
- Additional fine-tuning round that learns only the main node classification task:
  - Fine-tune the whole model: set variable in line 105 to `--include-additional-mclass-round "true"`
  - Fine-tune with the transformer component frozen: set variable in line 105 to `--include-additional-mclass-round-HEAD "true"`
- E2EG with BERT: set variable in line 104 to `--model-shortcut "bert-base-uncased"`
- E2EG with DistilBERT: set variable in line 104 to `--model-shortcut "distilbert-base-uncased"`
- E2EG with DistilRoberta SentenceBERT: set variable in line 104 to `--model-shortcut "sentence-transformers/all-distilroberta-v1"`
- E2EG in transductive setting: set variable in line 103 to `--include-Xval-Xtest-for-training "true"`
- E2EG in inductive setting: set variable in line 103 to `--include-Xval-Xtest-for-training "false"`

#### 2.2.2. Hyperparams sweep using [Weights & Biases](https://wandb.ai/site)
Have to create sweep config folder and define the sweep YAML file first. For more instructions see [wandb quickstart](https://docs.wandb.ai/guides/sweeps/quickstart).
```bash
dataset=ogbn-arxiv  # or ogbn-products
# Run sweep
experiment_name=E2EG_sweep
data_dir=data/proc_data_multi_task/${dataset}
model_dir=models/${experiment_name}
experiment_dir=experiments/${experiment_name}
cache_dir=models/cache
params_path=data/proc_data_multi_task/params_mtask_${dataset}.json
bash hyperparams_sweep.sh ${data_dir} ${model_dir} ${experiment_dir} ${cache_dir} ${params_path} sweep_configs/${experiment_name}
```

#### 2.2.3. Use E2EG for text embedding
After training an E2EG model according to [2.2.1](#221-train-and-evaluate-e2eg-models), run the following to get the text embedding using E2EG:
```bash
dataset=ogbn-arxiv  # or ogbn-products
experiment_name=E2EG
runs=1
data_dir=data/proc_data_multi_task/${dataset}
model_dir=models/${experiment_name}
experiment_dir=experiments/${experiment_name}
cache_dir=models/cache
params_path=data/proc_data_multi_task/params_mtask_${dataset}.json
bash encode_mtask.sh ${data_dir} ${model_dir} ${experiment_dir} ${cache_dir} ${params_path} ${runs}
```

The encoding is then saved to `models/E2EG/run0/X.all.e2eg-emb.npy`.

This encoding can then be used the same way as the encoding from `GIANT-XRT`. 

For the experiments in our paper, we replace the embedding from `GIANT-XRT` with our embedding from `E2EG` in the top 1 pipelines for `ogbn-arxiv` and `ogbn-products` on the OGB leaderboard (Date: 3 June 2022).
- For `ogbn-arxiv`: use E2EG's embedding and follows the instruction of the [top 1 pipeline's DRGCN repo](https://github.com/anonymousaabc/DRGCN).
- For `ogbn-products`: use E2EG's embedding and follows the instruction of the [top 1 pipeline's SCR repo](https://github.com/THUDM/SCR).