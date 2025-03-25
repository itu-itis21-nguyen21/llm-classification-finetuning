# LLM Classification Finetuning

This repository contains a single Jupyter notebook created for participating in Kaggle's [LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning). The goal of the competition is to predict which of two LLM-generated responses a human user will prefer, given a prompt.

## Approach

To frame the problem as a classification task, I defined three possible categories:
- **Model A wins**: The human user prefers the response from Model A.
- **Model B wins**: The human user prefers the response from Model B.
- **Tie**: The human user has no clear preference between the two responses.

## Model and Implementation

- **Base Model**: I used **DeBERTaV3 extra small** (70.68 million parameters) as the foundation model.
- **Finetuning Strategy**: The model was finetuned with a **single dense network layer** for classification.
- **Framework**: The implementation was done using **KerasHub** with Jax as the backend.

## Repository Contents
- `model.ipynb`: A Jupyter notebook containing the complete training and evaluation pipeline.

