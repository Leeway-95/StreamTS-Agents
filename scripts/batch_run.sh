#!/bin/bash
#export PYTHONPATH=/path/to/project_root:$PYTHONPATH
#export PYTHONPATH=/Users/liwei.1868/PycharmProjects/InfTS-LLM:$PYTHONPATH
export PYTHONPATH=/Users/liwei.1868/PycharmProjects/InfTS-LLM:$PYTHONPATH

echo "0.Clearing datasets..."
cd preprocess
python dataset_init.py
cd ..

echo "1.Converting datasets to stream format..."
cd preprocess
python dataset2stream.py
cd ..

cd model_monitor/claSS
echo "2.Starting the Monitoring Agent..."
python monitor.py
cd ..

echo "3.Starting the Reasoning Agent..."
cd agent_reasoner
python reaoner.py
cd ..

echo "4.Starting Postprocessing Analysis..."
cd postprocess
python analyze_results.py