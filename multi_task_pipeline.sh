#================= inputs =====================
data_dir=$1                             # e.g., .data/proc_data_multi_task/ogbn-arxiv
model_dir=$2
experiment_dir=$3
params_path=$4
runs=$5
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
  python -m pecos.xmc.xtransformer.train \
      --trn-text-path ${X_trn_txt_path} \
      --trn-feat-path ${X_trn_npz_path} \
      --trn-label-path ${Y_trn_neighbor_path} \
      --trn-class-path ${Y_trn_main_path} \
      --tst-text-path ${X_val_txt_path} \
      --tst-feat-path ${X_val_npz_path} \
      --tst-label-path ${Y_val_neighbor_path} \
      --tst-class-path ${Y_val_main_path} \
      --model-dir ${model_dir}/run${seed} \
      --params-path ${params_path} \
      --verbose-level 3 \
      --seed ${seed} \
      |& tee ${experiment_dir}/run${seed}/train.log
  grep -o "saving model.*" ${experiment_dir}/run${seed}/train.log >> ${experiment_dir}/run${seed}/val_performance_per_round.log
  grep -o "mclass-test-acc.*" ${experiment_dir}/run${seed}/train.log | tail -1 >> ${experiment_dir}/run${seed}/val_performance_per_round.log

  #==================== test ===================
  # Restructure the saved models
  for dir_path in ${model_dir}/run${seed}/round* ${model_dir}/run${seed}/final
  do
    dir="$(basename -- ${dir_path})"
    if [ -d ${model_dir}/run${seed}/${dir}/text_tokenizer ]; then
      mkdir -p ${model_dir}/run${seed}/${dir}_tmp/text_encoder
      mv ${model_dir}/run${seed}/${dir}/* ${model_dir}/run${seed}/${dir}_tmp/text_encoder
      rm -r ${model_dir}/run${seed}/${dir}
      mv ${model_dir}/run${seed}/${dir}_tmp ${model_dir}/run${seed}/${dir}
    fi
  done
  # Perform tests using every model from each fine-tuning round
  for dir_path in ${model_dir}/run${seed}/round* ${model_dir}/run${seed}/final
  do
    dir="$(basename -- ${dir_path})"
    python -m pecos.xmc.xtransformer.predict \
      --feat-path ${X_test_npz_path} \
      --text-path ${X_test_txt_path} \
      --model-folder ${model_dir}/run${seed}/${dir} \
      --save-pred-path-mlabel ${experiment_dir}/run${seed}/${dir}_prediction_mlabel \
      --save-pred-path-mclass ${experiment_dir}/run${seed}/${dir}_prediction_mclass \
      --multi-task
  done

  #==================== eval ===================
  for dir_path in ${model_dir}/run${seed}/round* ${model_dir}/run${seed}/final
  do
    dir="$(basename -- ${dir_path})"
    echo ${dir} | tee -a ${experiment_dir}/run${seed}/test_scores.txt
    python -m pecos.xmc.xtransformer.evaluate \
      --y-class-true ${Y_test_main_path} \
      --y-class-pred ${experiment_dir}/run${seed}/${dir}_prediction_mclass \
      | tee -a ${experiment_dir}/run${seed}/test_scores.txt
  done
done
#==================== combine results ===================
python -m results_combiner \
  --experiment_dir ${experiment_dir} \
  --runs ${runs} \
  | tee ${experiment_dir}/final_scores.txt