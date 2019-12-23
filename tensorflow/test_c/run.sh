#!/usr/bin/env bash

case $1 in
    r1.8)
        echo "use tensorflow r.8"
        TF_HOME=../release_c/r1.8/
        TF_INC=-I$TF_HOME/inc/
        TF_LIB=$TF_HOME/lib/linux64_cuda10.0_cudnn7.6_glibc2.12/
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TF_LIB 
        linkopt="-lrt -lpthread -ltensorflow_cc -ltensorflow_framework"
        ;;
    r2.0)    
        echo "use tensorflow r2.0"
        TF_HOME=../release_c/r2.0/
        #TF_INC="-I$TF_HOME/inc/"
        TF_INC="-I$TF_HOME/inc/ \
                -I$TF_HOME/inc/external/eigen_archive/ \
                -I$TF_HOME/inc/external/com_google_absl \
                -I$TF_HOME/inc/external/com_google_protobuf/src"
        TF_LIB=$TF_HOME/lib/linux64_cuda10.0_cudnn7.6_glibc2.12_glibcxx3.4.22/
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TF_LIB 
        linkopt="-lrt -lpthread -ltensorflow_cc"
        ;;
    *)       # unknown version, use r.18
        echo "unknown tf version, use default tensorflow_version = r1.8"
        TF_HOME=../release_c/r1.8/
        TF_INC=-I$TF_HOME/inc/
        TF_LIB=$TF_HOME/lib/linux64_cuda10.0_cudnn7.6_glibc2.12/
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TF_LIB 
        linkopt="-lrt -lpthread -ltensorflow_cc -ltensorflow_framework"
        ;;
esac


#TF_HOME=../release_c/r1.8/
#TF_INC=$TF_HOME/inc/
#TF_LIB=$TF_HOME/lib/linux64_cuda10.0_cudnn7.6_glibc2.12/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TF_LIB 



target_lst=(test_hello test_matmul test_load_pb)

for target in ${target_lst[@]}; do
    echo -e "\n**************** compile $target ... ****************"
    g++ -std=c++0x -o $target \
        -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w \
        $TF_INC \
        -L$TF_LIB \
        $linkopt \
        $target.c

    echo -e "\n**************** run $target ... ****************"
    ./$target && rm ./$target
done

