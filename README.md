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
  <img src="images/proposed_framework.png" alt="Time-Aware Generative Adversarial and Temporal Convolutional Learning Framework" width="600"/>
  <br>
  <em>Figure 1: Time-Aware Generative Adversarial and Temporal Convolutional Learning Framework. (S1) Collection of network traffic data and subsequent structuring using a sliding-window mechanism; (S2) Generation of synthetic data using TimeGAN; (S3) Labelling of the synthetic data; and (S4) TCN-based forecasting model.</em>
</p>



## Code explanation
(1) implementation.ipynb
  - This is the main notebook and it provides a user-guide for the framework's implementation
    
(2) data_loading.py

  - Transforms and preprocesses raw 5G network data to multivariate sequences suitable for TimeGAN and sequential models' training and evaluation.

(3) timegan.py

  - Training and development of the TimeGAN model, and subsequent generation of synthetic network data.
    
(4) qualitative.py

  - Qualitative analysis of the synthetic data generated using:
    
    (i) t-Distributed Stochastic Neighbour Embedding
    
    (ii) Temporal Quality Comparison Plots

(5) quantitative.py

  - Quantitative analysis of the synthetic data generated through its effectiveness in enhancing downstream predictive tasks of:
    
    (i) Temporal Convolutional Network (TCN): Main sequential model used in this research
    
    (ii) Recurrent Neural Network architectures: BiLSTM, standard RNN, and GRU












      
