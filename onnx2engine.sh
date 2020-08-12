#!/bin/sh
trtexec --onnx=dimp50_output.onnx --workspace=10240 --fp16 --saveEngine=dimp50_output.engine
