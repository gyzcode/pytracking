#!/bin/sh
trtexec --onnx=pytracking/networks/dimp18_training_output.onnx --workspace=4096 --fp16 --saveEngine=pytracking/networks/dimp18_training_output.engine

trtexec --onnx=pytracking/networks/dimp18_test_output.onnx --workspace=4096 --fp16 --saveEngine=pytracking/networks/dimp18_test_output.engine
