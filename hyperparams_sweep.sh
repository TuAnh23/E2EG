#================= inputs =====================
data_dir=$1                             # e.g., .data/proc_data_multi_task/ogbn-arxiv
model_dir=$2
experiment_dir=$3
cache_dir=$4
params_path=$5
sweep_config_dir=$6
if [ -z ${data_dir} ] || [ ! -d ${data_dir} ]; then
    echo "DATA_DIR does not exist: ${data_dir}"
    exit
fi

Y_trn_neighbor_path=${data_dir}/Y_neighbor.trn.npz        # training label matrix
Y_trn_main_path=${data_dir}/Y_main.trn.npy                # training class matrix
X_trn_txt_path=${data_dir}/X.trn.txt                      # training text
X_trn_npz_path=${data_dir}/X.trn.tfidf.npz                # training tfidf feature
X_trn_pt_path=${data_dir}/X.trn                           # save trn tensors here

Y_val_neighbor_path=${data_dir}/Y_neighbor.val.npz        # validation label matrix
Y_val_main_path=${data_dir}/Y_main.val.npy                # validation class matrix
X_val_txt_path=${data_dir}/X.val.txt                      # validation text
X_val_npz_path=${data_dir}/X.val.tfidf.npz                # validation tfidf feature
X_val_pt_path=${data_dir}/X.val                           # save val tensors here

Y_test_neighbor_path=${data_dir}/Y_neighbor.test.npz        # test label matrix
Y_test_main_path=${data_dir}/Y_main.test.npy                # test class matrix
X_test_txt_path=${data_dir}/X.test.txt                      # test text
X_test_npz_path=${data_dir}/X.test.tfidf.npz                # test tfidf feature
X_test_pt_path=${data_dir}/X.test                           # save test tensors here

tree_path=${data_dir}/HierarchialLabelTree                  # save Hierarchial Label Tree here

mkdir -p ${cache_dir}

seed=0
echo "Running with seed $seed"
#================== outputs ===================
mkdir -p ${experiment_dir}/run${seed}
mkdir -p ${model_dir}/run${seed}
if [[ -z "${TMPDIR}" ]]; then
  # TMPDIR environment variable is not yet defined, so we define one
  TMPDIR=${model_dir}/run${seed}/tmp
  mkdir -p ${TMPDIR}
  export TMPDIR=${model_dir}/run${seed}/tmp
else
  echo "TMPDIR is "$TMPDIR""
fi
#================== wandb config ===================
username=tuanh
projectname="UvA_Thesis"
# Save fixed-parsed-arguments of the training scripts (which do not need sweeping over) to a json file
python -m args_to_json \
	  --trn-text-path ${X_trn_txt_path} \
    --trn-feat-path ${X_trn_npz_path} \
    --trn-label-path ${Y_trn_neighbor_path} \
    --trn-class-path ${Y_trn_main_path} \
    --val-text-path ${X_val_txt_path} \
    --val-feat-path ${X_val_npz_path} \
    --val-label-path ${Y_val_neighbor_path} \
    --val-class-path ${Y_val_main_path} \
    --test-text-path ${X_test_txt_path} \
    --test-feat-path ${X_test_npz_path} \
    --test-label-path ${Y_test_neighbor_path} \
    --test-class-path ${Y_test_main_path} \
    --model-dir ${model_dir}/run${seed} \
    --experiment-dir ${experiment_dir}/run${seed} \
    --cache-dir ${cache_dir} \
    --params-path ${params_path} \
    --verbose-level 3 \
    --seed ${seed} \
    --tree-path ${tree_path} \
    --memmap "true" \
    --dataset $(basename $data_dir) \
    --saved-trn-pt ${X_trn_pt_path} \
    --saved-val-pt ${X_val_pt_path} \
    --weight-loss-strategy "None" \
    --include-Xval-Xtest-for-training "true" \
    --model-shortcut "distilbert-base-uncased" \
    --include-additional-mclass-round-HEAD "true" \
    --test-portion-for-training "None" \
    --val-portion-for-training "None" \
    --wandb-username tuanh \
    --wandb-sweep yes \
    --save_json_path ${sweep_config_dir}/training_config.json \
    --swept_args "numb_layers_mclass_pred|mclass_pred_dropout_prob|mclass_pred_batchnorm|mclass_pred_hidden_size"

##==================== setup sweep ===================
if [ ! -f ${sweep_config_dir}/sweep_info.txt ]; then
  wandb sweep --entity ${username} --project ${projectname} ${sweep_config_dir}/sweep.yaml |& tee ${sweep_config_dir}/sweep_info.txt
fi
sweepID=$(grep 'wandb: Created sweep with ID: ' ${sweep_config_dir}/sweep_info.txt | sed 's/^.*: //')
echo "sweepID ${sweepID}"
#==================== start sweep ===================
wandb agent --count 4 ${username}/${projectname}/${sweepID}
