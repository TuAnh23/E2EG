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

start_seed=0
end_seed=runs-1
for (( seed=$start_seed; seed<=$end_seed; seed++ ))
do
	if [ -f "${model_dir}/run${seed}/X.all.e2eg-emb.npy" ]
	then
	  echo "Text encoding for run${seed} exists, skip this run"
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
  #==================== embed text using best multi-task model ===================
  best_round=$(tac ${experiment_dir}/run${seed}/train.log | grep -m 1 "Best validation accuracy at round " | cut -d\   -f6)
  python -m pecos.xmc.xtransformer.encode \
      -t ${X_txt_path} \
      -m ${model_dir}/run${seed}/round${best_round} \
      -o ${model_dir}/run${seed}/X.all.e2eg-emb.npy \
      --memmap "true" \
      --batch-size 64 \
      --verbose-level 3 \
      --mtask "true" \
      |& tee ${experiment_dir}/run${seed}/predict.log
done