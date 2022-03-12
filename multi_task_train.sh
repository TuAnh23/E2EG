#================= inputs =====================
data_dir=$1                             # e.g., .data/proc_data_multi_task/ogbn-arxiv
params_path=$2
if [ -z ${data_dir} ] || [ ! -d ${data_dir} ]; then
    echo "DATA_DIR does not exist: ${data_dir}"
    exit
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

#================== outputs ===================
model_dir=models/multi_task_models
experiment_dir=experiments/multi_task_models
mkdir -p ${experiment_dir}
mkdir -p ${model_dir}
TMPDIR=${model_dir}/tmp
mkdir -p ${TMPDIR}
export TMPDIR=${model_dir}/tmp

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
    --model-dir ${model_dir} \
    --params-path ${params_path} \
    --verbose-level 3 \
    |& tee ${experiment_dir}/train.log

#==================== test ===================
python -m pecos.xmc.xtransformer.predict \
    --feat-path ${X_test_npz_path} \
    --text-path ${X_test_txt_path} \
    --model-folder ${model_dir}/final \
    --save-pred-path-mlabel ${experiment_dir}/prediction_mlabel \
    --save-pred-path-mclass ${experiment_dir}/prediction_mclass \
    --multi-task

#==================== eval ===================
python -m pecos.xmc.xtransformer.evaluate \
    --y-class-true ${Y_test_main_path} \
    --y-class-pred ${experiment_dir}/prediction_mclass \
    --save-score-path ${experiment_dir}/scores.txt