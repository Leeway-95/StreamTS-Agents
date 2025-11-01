<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/387015cb-2f6f-4b61-81e4-10815b364528" style="width:65;height:30px;"> Understanding and Reasoning Streaming Time Series with Specialized LLM Agents </b></h2>
</div>

![Topic](https://img.shields.io/badge/Streaming%20Time%20Series%20-%20LLM--Agents-blueviolet)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/Leeway2027/Qwen3-TSVL-4B)
[![DOI](https://zenodo.org/badge/DOI/Datasets.svg)](https://doi.org/10.5281/zenodo.14349206)
![Stars](https://img.shields.io/github/stars/Leeway-95/StreamTS-Agents)
![Forks](https://img.shields.io/github/forks/Leeway-95/StreamTS-Agents)

<!--[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Qwen3-TSVL-4B-Web%20Demo-blue)](https://huggingface.co/spaces/Leeway2027/Qwen3-TSVL-4B)-->

This repository provides the code for our paper, which introduces StreamTS-Agents, a multi-agent framework that enhances LLM-based reasoning over streaming time series.

>  ✨ If you find our work useful for your research, please consider giving it a <strong>star ⭐ on GitHub</strong> to stay updated with future releases.

## Demonstration
<!--
https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

## Key Features
StreamTS-Agents can be directly applied to any LLMs without retraining:
- ✅ **Native support for multivariate time series**
-->

### Example Demonstration
Here is an example of specialized LLM agents operating over streaming time series in gold trading. The monitoring agent observes the gold price, records captions at detected temporal patterns, and conveys them to the reasoning agent, which answers the user's questions.

<p align="center">
  <img width="700" height="600" alt="image" src="https://github.com/user-attachments/assets/88d697bc-7577-42a4-a303-511e13b93c7a" />
</p>

## Abstract
Streaming time series are collected sequentially over time in real-world applications, such as real-time financial trading and database usage monitoring. Multi-agent collaboration enhances long-term dependencies in LLM-based reasoning over streaming time series. However, existing studies often overlook specialization among agents with distinct yet interdependent objectives. To bridge this gap, this paper introduces a multi-agent specialized paradigm, in which agents pursue complementary goals such as low-cost monitoring and high-quality reasoning. This setting faces three major challenges: (1) continuous temporal understanding, (2) effective cooperation among specialized agents, and (3) temporal context collapse. To address these challenges, this paper introduces StreamTS-Agents, a multi-agent framework comprising: (a) a monitoring agent that generates descriptive and analytical captions to enable continuous temporal understanding over streaming time series, (b) a caption storage that maps captions to patterns for retrieval, thereby facilitating multi-agent cooperation, and (c) a reasoning agent that performs caption-guided chain-of-thought (CoT) reasoning and maintains an insight playbook to mitigate temporal context collapse.

## Motivation
Existing works typically overlook specialization among agents with complementary goals. We categorize existing work into two paradigms: ① Single-agent Unified Paradigm and ② Multi-agent Cooperative Paradigm. However, Paradigm ① omits middle steps for complex tasks, and Paradigm ② lacks specialization for distinct objectives. To overcome these limitations, we define ③ Multi-agent Specialized Paradigm, designed for LLM agents with complementary goals, such as low-cost monitoring and high-quality reasoning.

<p align="center">
  <img width="650" height="480" alt="image" src="https://github.com/user-attachments/assets/53f2cad0-3ef9-4317-a4f9-c0914c5c094f" />
</p>

## Case Study
StreamTS-Agents employ specialized agents for monitoring and reasoning, supported by a pattern-mapping caption storage, to overcome challenges of continuous temporal understanding, agent cooperation, and temporal context collapse. The framework significantly improves performance on understanding and reasoning tasks.
 
<p align="center">
  <img width="800" height="2000" alt="image" src="https://github.com/user-attachments/assets/963725dd-e630-4b01-8b51-cfb5156be00e" />
</p>


## Dependencies
* Python 3.12
* numpy==1.26.4
* numba==0.61.0
* pandas==2.3.1
* apache-flink==2.1.0

```bash
> conda env create -f env_linux.yaml
```

## Datasets
1. Gold datasets can be obtained from our **datasets directory**.
2. Understanding task dataset can be download from [TSQA](https://huggingface.co/datasets/ChengsenWang/TSQA).
3. Reasoning task datasets can be download from [AIOps](https://github.com/netmanaiops/kpi-anomaly-detection), [WeatherQA](https://www.bgc-jena.mpg.de/wetter), and [NAB](https://github.com/numenta/NAB), [Oracle](https://zenodo.org/records/6955909), and [MCQ2](https://github.com/behavioral-data/TSandLanguage)
4. Event forecasting task datasets can be download from [Finance](https://github.com/geon0325/TimeCAP/tree/main/dataset/finance), [Healthcare](https://github.com/geon0325/TimeCAP/tree/main/dataset/healthcare), and [Weather](https://github.com/geon0325/TimeCAP/tree/main/dataset/weather).

## Usages

* ### Obtain StreamTS-Agents

```bash
git clone https://github.com/Leeway-95/StreamTS-Agents.git
cd StreamTS-Agents
pip3 install -r requirements.txt
```

* ### Batch version

```bash
sh scripts/batch_run.sh
```

* ### Stream version
   
```bash
sh scripts/stream_run.sh
```

## Contact Us
For inquiries or further assistance, contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).
