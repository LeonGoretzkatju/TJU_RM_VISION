#!/bin/bash

cd $(dirname $0)/../
timeout 8 git pull
cd make
cmake ..
make -j8
