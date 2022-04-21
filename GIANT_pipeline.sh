#================= inputs =====================
data_dir=$1                             # e.g., .data/proc_data_multi_task/ogbn-arxiv
model_dir=$2
experiment_dir=$3
cache_dir=$4
params_path=$5
runs=$6
if [ -z ${data_dir} ] || [ ! -d ${data_dir} ]; then
    echo "DATA_DIR does not exist: ${data_dir}"
    exit
fi
if [ -z "$runs" ]
then
  echo "Number of runs is not specified, set to 1"
  runs=1
fi

X_txt_path=${data_dir}/X.all.txt                          # all text

Y_trn_neighbor_path=${data_dir}/Y_neighbor.trn.npz        # training label matrix
Y_trn_main_path=${data_dir}/Y_main.trn.npy                # training class matrix
X_trn_txt_path=${data_dir}/X.trn.txt                      # training text
X_trn_npz_path=${data_dir}/X.trn.tfidf.npz                # training tfidf feature
X_trn_pt_path=${data_dir}/X.trn.pt                        # save trn tensors here

Y_val_neighbor_path=${data_dir}/Y_neighbor.val.npz        # validation label matrix
Y_val_main_path=${data_dir}/Y_main.val.npy                # validation class matrix
X_val_txt_path=${data_dir}/X.val.txt                      # validation text
X_val_npz_path=${data_dir}/X.val.tfidf.npz                # validation tfidf feature
X_val_pt_path=${data_dir}/X.val.pt                        # save val tensors here

Y_test_neighbor_path=${data_dir}/Y_neighbor.test.npz        # test label matrix
Y_test_main_path=${data_dir}/Y_main.test.npy                # test class matrix
X_test_txt_path=${data_dir}/X.test.txt                      # test text
X_test_npz_path=${data_dir}/X.test.tfidf.npz                # test tfidf feature
X_test_pt_path=${data_dir}/X.test.pt                        # save test tensors here

mkdir -p ${cache_dir}

start_seed=0
end_seed=runs-1
for (( seed=$start_seed; seed<=$end_seed; seed++ ))
do
	if [ -d "${experiment_dir}/run${seed}" ]
	then
	  echo "Results for run${seed} exists, skip this run"
	  continue 1
	fi
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
  #==================== train GIANT ===================
  # Store the timestamp of the start of this run to use as runID for wandb
  timestamp=$(date +%s)
  python -m pecos.xmc.xtransformer.train \
      --trn-text-path ${X_trn_txt_path} \
      --trn-feat-path ${X_trn_npz_path} \
      --trn-label-path ${Y_trn_neighbor_path} \
      --tst-text-path ${X_val_txt_path} \
      --tst-feat-path ${X_val_npz_path} \
      --tst-label-path ${Y_val_neighbor_path} \
      --model-dir ${model_dir}/run${seed} \
      --experiment-dir ${experiment_dir}/run${seed} \
      --cache-dir ${cache_dir} \
      --params-path ${params_path} \
      --verbose-level 3 \
      --seed ${seed} \
      --wandb-username tuanh \
      --wandb-run-id ${timestamp} \
      |& tee ${experiment_dir}/run${seed}/train.log
  #==================== embed text using GIANT ===================
  python -m pecos.xmc.xtransformer.encode \
      -t ${X_txt_path} \
      -m ${model_dir}/run${seed}/last \
      -o ${model_dir}/run${seed}/X.all.xrt-emb.npy \
      --batch-size 64 \
      --verbose-level 3 \
      |& tee ${experiment_dir}/run${seed}/predict.log
  #==================== train MLP ===================
  python -u baseline_models/mlp.py \
      --runs 1 \
      --data_root_dir ${data_dir}/../ \
      --node_emb_path ${model_dir}/run${seed}/X.all.xrt-emb.npy \
      |& tee ${experiment_dir}/run${seed}/final_scores.txt
done
#==================== combine results ===================
python -m results_combiner \
  --experiment_dir ${experiment_dir} \
  --runs ${runs} \
  | tee ${experiment_dir}/final_scores.txt