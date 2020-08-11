#!/bin/sh
trtexec --onnx=test.onnx --workspace=10240 --fp16 --saveEngine=dimp50.engine
