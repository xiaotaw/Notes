#!/usr/bin/env bash


TF_HOME=../release/r1.8/
TF_INC=$TF_HOME/inc/
TF_LIB=$TF_HOME/lib/linux64_cuda10.0_cudnn7.6_glibc2.12/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TF_LIB 






target_lst=(test_hello test_matmul test_load_pb)

for target in ${target_lst[@]}; do
    echo -e "\n**************** compile $target ... ****************"
    g++ -std=c++0x -o $target \
        -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w \
        -I$TF_INC \
        -L$TF_LIB \
        -lrt -lpthread -ltensorflow_cc -ltensorflow_framework \
        $target.c

    echo -e "\n**************** run $target ... ****************"
    ./$target && rm ./$target
done

