#!/bin/bash

VENV=/tmp/venv/
TVM_DIR=/home/ga87puy/src/tvm_schedules/tvm
OUT=/tmp/vankempen/out
MODEL_ZOO=/tmp/vankempen/model_zoo
MODEL_DIRS=($MODEL_ZOO/mlonmcu)
ALLOWED_EXTENSIONS=(tflite onnx)
IGNORE_LIST=()

export PYTHONPATH=$TVM_DIR/python

source $VENV/bin/activate

find_models() {
    BASE_DIR=$1
    # echo "BASE_DIR=$BASE_DIR"
    for ext in "${ALLOWED_EXTENSIONS[@]}"
    do
        # echo "ext=$ext"
        cd $BASE_DIR
        # CANDIDATES=$(find $BASE_DIR -iname "*.$ext")
        find . -iname "*.$ext"
        cd -
        #echo $CANDIDATES
        # echo "CANDIDATES=$CANDIDATES"
    done
}

invoke_tvmc() {
    MODEL_PATH=$1
    echo "MODEL_PATH=$MODEL_PATH"
    OUT_PATH=$2
    # echo "OUT_PATH=$OUT_PATH"
    TVMC_OUT_MLF=$OUT_PATH/mlf.tar
    TVMC_OUT_MLF_DIR=$OUT_PATH/mlf
    TVMC_ARGS="compile $MODEL_PATH --target c --runtime=crt --executor=aot --executor-aot-unpacked-api=1 --executor-aot-interface-api c --pass-config tir.disable_vectorize=1 -f mlf -o $TVMC_OUT_MLF"
    # echo "TVMC_ARGS=$TVMC_ARGS"
    # TODO: optional usmp?
    TVMC_OUT_LOG=$OUT_PATH/tvmc.log
    # echo python -m tvm.driver.tvmc $TVMC_ARGS
    python -m tvm.driver.tvmc $TVMC_ARGS > $TVMC_OUT_LOG
    mkdir -p $TVMC_OUT_MLF_DIR
    tar xf $TVMC_OUT_MLF -C $TVMC_OUT_MLF_DIR
}

echo "START"

for d in "${MODEL_DIRS[@]}"
do
    # echo "d=$d"
    FOUND_MODELS=($(find_models $d))
    for m in "${FOUND_MODELS[@]}"
    do
        # echo "m=$m"
        o=$OUT/$d/${m%.*}
        # echo "o=$o"
        mkdir -p $o
        cp $d/$m $OUT/$d/$m
        invoke_tvmc $m $o
    done
done


echo "STOP"

