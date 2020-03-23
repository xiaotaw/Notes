#!/usr/bin/env bash
g++ -std=c++11 test_json.cpp -o test_json

chmod u+x test_json && ./test_json  && rm ./test_json
