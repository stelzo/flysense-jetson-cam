#!/bin/bash

git submodule update --recursive --init

cd GPUJPEG && cmake . -Bbuild && cmake --build build

autogen.sh
configure
make
