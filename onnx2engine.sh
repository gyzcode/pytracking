#!/bin/sh
trtexec --onnx=pytracking/networks/dimp50_training_output.onnx --workspace=10240 --fp16 --saveEngine=pytracking/networks/dimp50_training_output.engine

trtexec --onnx=pytracking/networks/dimp50_test_output.onnx --workspace=10240 --fp16 --saveEngine=pytracking/networks/dimp50_test_output.engine
