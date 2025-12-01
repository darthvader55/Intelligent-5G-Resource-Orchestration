# Codebase for "Time-Aware Generative Adversarial and Temporal Convolutional Learning for Intelligent 5G Resource Orchestration"
This repository contains an overview and implementation summary of my Masters' thesis, which was submitted in partial fulfilment of the requirements for the Master of Science in Computing in Big Data Analytics and Artificial Intelligence at Atlantic Technological University - Donegal, Ireland.


Author: Linus Ngatia

Contact: linusngatia434@gmail.com

## Purpose and Research Question
The purpose of this research is to develop an intelligent resource orchestration framework that maximises Quality of Experience (QoE) for diverse 5G application types and services by accurately forecasting network demands. Leveraging an architecture that integrates a Time-series Generative Adversarial Network (TimeGAN) and a Temporal Convolutional Network (TCN)-based forecasting model, this work seeks to provide a robust framework for dynamic resource orchestration in 5G ecosystems.

Accordingly, this dissertation aims to address the following research questions:

  - How can the components in TimeGAN be optimised to generate synthetic time-series data that effectively capture temporal dependencies in network traffic, thereby enhancing data diversity and consistency to enable accurate and intelligent 5G resource orchestration?

  - To what extent can a semi-supervised TimeGAN-TCN framework enhance forecasting accuracy and generalisation in dynamic 5G environments with sparse supervision and evolving traffic patterns, compared to traditional models such as BiLSTM, RNN, and GRU?

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












      
