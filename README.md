# Time-Aware Generative Adversarial and Temporal Convolutional Learning for Intelligent 5G Resource Orchestration
This repository contains an implementation summary and supporting code for my Masters' thesis, which was submitted in partial fulfilment of the requirements for the Master of Science in Computing in Big Data Analytics and Artificial Intelligence at Atlantic Technological University - Donegal, Ireland.

Author: Linus Ngatia

Contact: linusngatia434@gmail.com

## Overview
This research develops an intelligent resource orchestration framework that maximises Quality of Experience (QoE) for diverse 5G application types and services by accurately forecasting network demands. Leveraging an architecture that integrates a Time-series Generative Adversarial Network (TimeGAN) and a Temporal Convolutional Network (TCN)-based forecasting model, the work provides a robust framework for dynamic resource orchestration in 5G ecosystems.

## Abstract
<details>
  <summary><strong>Click to expand</strong></summary>
With rapid technological advancements and the digitisation of various sectors, the demand for connectivity has surged, driving exponential growth in mobile communication networks and a wide range of applications with diverse requirements. Fifth-generation (5G) mobile networks have shown strong potential in meeting these demands. However, the highly dynamic and heterogeneous nature of 5G environments makes resource management and orchestration particularly challenging, and traditional approaches often fail to meet the strict Quality of Service (QoS) requirements in 5G networks. Consequently, recent studies have explored the use of Artificial Intelligence (AI) algorithms to enable intelligent orchestration of network resources and ensure a high Quality of Experience (QoE) for users. Despite this progress, AI-driven approaches face critical limitations, primarily stemming from the scarcity and lack of diversity in training data. Issues such as temporal drift and privacy concerns restrict the availability of large, representative datasets. As a result, models trained on such limited data often fail to generalise effectively in real-world 5G environments, leading to inefficient orchestration and reduced QoE for end users. This thesis addresses these challenges by proposing a time-aware generative–adversarial and temporal convolutional learning framework for intelligent and accurate 5G resource orchestration. The framework incorporates a Time-series Generative Adversarial Network (TimeGAN) model for synthetic time-series data generation, effectively capturing temporal dependencies in network traffic to enhance data diversity and consistency. This synthetic data improves the forecasting accuracy and generalisation capability of a Temporal Convolutional Network (TCN) model, ultimately supporting higher QoE in dynamic 5G environments. Comparative experiments between models trained solely on real data and those trained on real data augmented with the synthetic data demonstrate that the proposed TimeGAN–TCN framework significantly improves forecasting performance, with Mean Square Error (MSE) reducing from 0.0412 to 0.0167, Mean Average Error (MAE) from 0.1638 to 0.1069, and Dynamic Time Warping distance (DTW) from 0.9528 to 0.6526.
</details>

--

## Proposed Framework

<p align="center">
  <img src="images/proposed_framework.png" alt="Time-Aware Generative Adversarial and Temporal Convolutional Learning Framework" width="900"/>
  <br>
  <em>Figure 1: Time-Aware Generative Adversarial and Temporal Convolutional Learning Framework. (S1) Collection of network traffic data and subsequent structuring using a sliding-window mechanism; (S2) Generation of synthetic data using TimeGAN; (S3) Labelling of the synthetic data; and (S4) TCN-based forecasting model.</em>
</p>


<details>
  <summary><strong>Click to expand</strong></summary>
The proposed framework enables intelligent 5G resource orchestration through a hybrid generative-discriminative architecture. It comprises three core modules: (1) a TimeGAN-based temporal data synthesizer, (2) a semi-supervised sequence labelling module, and (3) a forecasting model based on a Dilated TCN. As shown in Figure 1, the system uses real time-series data to train a TimeGAN model that captures complex temporal dependencies and generates realistic synthetic sequences. These sequences are then passed through a semi-supervised learning module to produce labelled training data, which are subsequently used to train a TCN model for robust forecasting under dynamic 5G environments.
</details>

--

## Code Implementation
The main implementation of this framework is provided in **[`implementation.ipynb`](codefiles/implementation.ipynb)**.
This notebook serves as a **step-by-step user guide**, showing how to use the other python modules to reproduce the experiments and generate results.

💡 **Tip:** Start with this notebook to understand the workflow, as it imports and orchestrates all other scripts.

### Python Modules

(1) [`data_loading.py`](codefiles/data_loading.py)

  - Transforms and preprocesses raw 5G network data into multivariate sequences suitable for TimeGAN and sequential models' training and evaluation.

(2) [`timegan.py`](codefiles/timegan.py)

  - Development and training of the TimeGAN model, and subsequent generation of synthetic network data.
    
(3) [`qualitative.py`](codefiles/qualitative.py)

  - Performs **qualitative analysis** of the synthetic data using:
    
    (i) t-Distributed Stochastic Neighbour Embedding
    
    (ii) Temporal Quality Comparison Plots

(4) [`quantitative.py`](codefiles/quantitative.py)

  - Performs **quantitative analysis** to evaluate the effectiveness of the synthetic data in enhancing downstream predictive tasks of:
    
    (i) Temporal Convolutional Network (TCN): Main sequential model used in this research
    
    (ii) Recurrent Neural Network architectures: BiLSTM, standard RNN, and GRU

---

## Contact

For questions or collaborations, feel free to reach out:
**Email:** linusngatia434@gmail.com












      
