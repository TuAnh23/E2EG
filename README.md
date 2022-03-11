# End-to-end node classification with text-based node attribute

Master thesis conducted at the University of Amsterdam, in collaboration with Socialdatabase.

The repository is derived from [PECOS](https://github.com/amzn/pecos).

## Requirements and Installation

* Python (==3.8)
* Pip (>=19.3)

### Supporting Platforms
* Ubuntu 18.04 and 20.04
* Amazon Linux 2

### Installation from Source

#### Prerequisite builder tools
* For Ubuntu (18.04, 20.04):
``` bash
sudo apt-get update && sudo apt-get install -y build-essential git python3 python3-distutils python3-venv
```
* For Amazon Linux 2 Image:
``` bash
sudo yum -y install python3 python3-devel python3-distutils python3-venv && sudo yum -y install groupinstall 'Development Tools' 
```
One needs to install at least one BLAS library to compile PECOS, e.g. `OpenBLAS`:
* For Ubuntu (18.04, 20.04):
``` bash
sudo apt-get install -y libopenblas-dev
```
* For Amazon Linux 2 Image and AMI:
``` bash
sudo amazon-linux-extras install epel -y
sudo yum install openblas-devel -y
```

#### Install and develop locally
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

## Reproduce results

### Baseline models
Train graphSAGE on node degree:
```bash
mkdir experiments/graphSAGE_nodedegree
python -u baseline_models/gnn.py \
  --use_sage \
  --node_feature degree \
  --hidden_channels 50 | tee -a experiments/graphSAGE_nodedegree/train.log
```

Train transformer using only node text attribute:
```bash
# Download data
cd data/proc_data_multi_task
dataset=ogbn-arxiv
bash download_data.sh ${dataset}
cd ../../
# Train model
mkdir experiments/bert_classifier
mkdir models/bert_classifier
source activate giant-xrt
python -u baseline_models/bert_classifier.py \
  --model_dir models/bert_classifier \
  --raw-text-path data/proc_data_multi_task/ogbn-arxiv/X.all.txt \
  --text_tokenizer_path data/proc_data_multi_task/ogbn-arxiv/xrt_models/text_encoder/text_tokenizer \
  | tee -a experiments/bert_classifier/train.log
```

### Proposed models
Note: in this repository, we make a clear distinction between `multi-label` classification tasks and `multi-class` classification tasks (whereas in real-life, the term "label" and "class" might be used interchangeably)

- `Multi-class` classification: each input will have only one output class. This is the case of the main node classification task.
- `Multi-label` classification: each input can have multiple output labels. This is the case of the auxiliary neighborhood prediction task.

The proposed model learns both the `Multi-class` classification task and the `Multi-label` classification task, and backpropagate the loss in tandem for both tasks
```bash
# Download data
cd data/proc_data_multi_task
dataset=ogbn-arxiv
bash download_data.sh ${dataset}
cd ../../
# Process data
source activate giant-xrt
bash proc_data_multi_task.sh ${dataset}
# Train model
data_dir=data/proc_data_multi_task/${dataset}
params_path=data/proc_data_multi_task/params_mtask_${dataset}.json
bash multi_task_train.sh ${data_dir} ${params_path}
```