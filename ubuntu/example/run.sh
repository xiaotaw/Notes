#!/usr/bin/env bash
echo -e "\n\nRun test_json for nlohmann json. \n"
g++ -std=c++11 test_json.cpp -o test_json
chmod u+x test_json && ./test_json  && rm ./test_json


echo -e "\n\nRun test_c_str for std::string.c_str. \n"
gcc_ver=`gcc --version | grep gcc`
if [[ $gcc_ver =~ 4.9.2 ]]; then
    gcc -std=c++0x -lstdc++ test_c_str.cpp -o test_c_str
    chmod u+x test_c_str && ./test_c_str && rm ./test_c_str
else
    echo -e "$gcc_ver is detected."
    echo -e "only gcc=4.9.2 is tested for this example."
    echo -e "higher version such as 6.5.0 seems to solved this problem. \n"
fi

