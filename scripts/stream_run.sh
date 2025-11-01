#!/bin/bash
#export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTHONPATH=/Users/liwei.1868/PycharmProjects/InfTS-LLM:$PYTHONPATH

#echo "Converting Dataset to stream format..."
#cd preprocess
#python /Users/liwei.1868/PycharmProjects/StreamTS-Agents/dataset2stream.py

echo "Starting Flink job..."
cd stream_flink
python flink_app.py