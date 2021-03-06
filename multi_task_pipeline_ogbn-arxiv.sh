#================= inputs =====================
data_dir=$1                             # e.g., .data/proc_data_multi_task/ogbn-arxiv
model_dir=$2
experiment_dir=$3
cache_dir=$4
params_path=$5
runs=$6
wandb_username="None"  # Replace with your username if use wandb 
if [ -z ${data_dir} ] || [ ! -d ${data_dir} ]; then
    echo "DATA_DIR does not exist: ${data_dir}"
    exit
fi
if [ -z "$runs" ]
then
  echo "Number of runs is not specified, set to 1"
  runs=1
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

Y_test_all_neighbor_path=${data_dir}/Y_neighbor.test_all.npz        # test_all label matrix
Y_test_all_main_path=${data_dir}/Y_main.test_all.npy                # test_all class matrix
X_test_all_txt_path=${data_dir}/X.test_all.txt                      # test_all text
X_test_all_npz_path=${data_dir}/X.test_all.tfidf.npz                # test_all tfidf feature
X_test_all_pt_path=${data_dir}/X.test_all                           # save test_all tensors here

tree_path=${data_dir}/HierarchialLabelTree                  # save Hierarchial Label Tree here

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
  #==================== train ===================
  # Store the timestamp of the start of this run to use as runID for wandb
  timestamp=$(date +%s%N)
  python -m pecos.xmc.xtransformer.train \
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
      --tree-path ${tree_path} \
      --memmap "true" \
      --dataset $(basename $data_dir) \
      --saved-trn-pt ${X_trn_pt_path} \
      --saved-val-pt ${X_val_pt_path} \
      --seed ${seed} \
      --wandb-username ${wandb_username} \
      --wandb-run-id ${timestamp} \
      --weight-loss-strategy "include_mclass_loss_later_at_round_2" \
      --numb-layers-mclass-pred 1 \
      --mclass-pred-dropout-prob 0.2 \
      --mclass-pred-batchnorm "yes" \
      --mclass-pred-hidden-size 256 \
      --freeze-mclass-head-range "None" \
      --include-Xval-Xtest-for-training "true" \
      --model-shortcut "sentence-transformers/all-distilroberta-v1" \
      --include-additional-mclass-round-HEAD "true" \
      --test-portion-for-training "None" \
      --val-portion-for-training "None" \
      |& tee ${experiment_dir}/run${seed}/train.log

  #==================== test ===================
  best_round=$(tac ${experiment_dir}/run${seed}/train.log | grep -m 1 "Best validation accuracy at round " | cut -d\   -f6)
  # Restructure the saved models
  for dir_path in ${model_dir}/run${seed}/round* ${model_dir}/run${seed}/last
  do
    dir="$(basename -- ${dir_path})"
    if [ -d ${model_dir}/run${seed}/${dir}/text_tokenizer ]; then
      mkdir -p ${model_dir}/run${seed}/${dir}_tmp/text_encoder
      mv ${model_dir}/run${seed}/${dir}/* ${model_dir}/run${seed}/${dir}_tmp/text_encoder
      rm -r ${model_dir}/run${seed}/${dir}
      mv ${model_dir}/run${seed}/${dir}_tmp ${model_dir}/run${seed}/${dir}
    fi
  done
  # Perform tests using the model from the fine-tuning round with best validation accuracy
  dir_path=${model_dir}/run${seed}/round${best_round}
  dir="$(basename -- ${dir_path})"
  python -m pecos.xmc.xtransformer.predict \
    --text-path ${X_test_all_txt_path} \
    --saved-test-pt ${X_test_all_pt_path} \
    --memmap "true" \
    --model-folder ${model_dir}/run${seed}/${dir} \
    --save-pred-path-mlabel ${experiment_dir}/run${seed}/${dir}_prediction_mlabel \
    --save-pred-path-mclass ${experiment_dir}/run${seed}/${dir}_prediction_mclass \
    --multi-task \
    --seed ${seed}
  # Remove everything except for the best round
  find ./${model_dir}/run${seed}/ -mindepth 1 -maxdepth 1 -type d -not -name round${best_round} -exec rm -rf '{}' \;
  
  #==================== eval ===================
  # Calculate the test scores
  echo ${dir} | tee -a ${experiment_dir}/run${seed}/test_scores.txt
  python -m pecos.xmc.xtransformer.evaluate \
    --y-class-true ${Y_test_all_main_path} \
    --y-class-pred ${experiment_dir}/run${seed}/${dir}_prediction_mclass \
    --dataset $(basename $data_dir) \
    | tee -a ${experiment_dir}/run${seed}/test_scores.txt
  # Calculate and log the final best validation score and corresponding test score from this run
  python -m pecos.xmc.xtransformer.final_metrics_collection \
    --experiment_dir ${experiment_dir}/run${seed} \
    --wandb-username ${wandb_username} \
    --wandb-run-id ${timestamp} \
    | tee ${experiment_dir}/run${seed}/final_scores.txt
done
#==================== combine results ===================
python -m results_combiner \
  --experiment_dir ${experiment_dir} \
  --runs ${runs} \
  | tee ${experiment_dir}/final_scores.txt