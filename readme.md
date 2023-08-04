# Car Evaluation

## Overview

This is a Deep Learning project for car evaluation using a custom Linear Model implemented with PyTorch. The goal of the project is to classify car evaluations into one of four categories: "unacceptable," "acceptable," "good," and "very good."

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Making Predictions](#making-predictions)
- [Results](#results)

## Dataset

[Source](https://code.datasciencedojo.com/datasciencedojo/datasets/tree/master/Car%20Evaluation)

Car Evaluation Data Set consists of 7 attributes. Buying, maint, doors, persons, lug_boot, safery are 6 features of cars, and class is the target variable.  
The goal is to predict car acceptability (class) of a car using these 6 features.  
The table below shows the detailed definition of each attribute.

| Attribute name | Definition                                                                                |
|----------------|-------------------------------------------------------------------------------------------|
| buying         | Buying price of the car (v-high, high, med, low)                                          |
| maint          | Price of the maintenance of car (v-high, high, med, low)                                  |
| doors          | Number of doors (2, 3, 4, 5-more)                                                         |
| persons        | Capacity in terms of persons to carry (2, 4, more)                                        |
| lug_boot       | The size of luggage boot (small, med, big)                                                |
| safety         | Estimated safety of the car (low, med, high)                                              |
| class          | Car acceptability (unacc: unacceptible, acc: acceptible, good: good,   v-good: very good) |

## Requirements

Before running the project, make sure you have the following prerequisites:

- Python (>= 3.6)
- PyTorch (>= 1.7.1)
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm
- imbalanced-learn

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

The project consists of several components:

- EDA.ipynb - jupyter notebook with Exploratory Data Analysis
- data_loader.py: contains the custom Dataset class for loading the car 
evaluation dataset.
- model.py: defines the custom Linear Model.
- engine.py: implements the training and evaluation functions.
- utils.py: contains utility functions for logging, checkpoint saving.
- early_stopping: defines class for early stopping. 
- run.py: the main script to run the training and evaluation process.
- made_pred.py - script for making prediction with user's data.

## Training and Evaluation
To train the model, run the following command:

```
python run.py
```
The progress will be displayed on the console, and the best model checkpoint will be saved in the experiments directory.

## Making Predictions
To make predictions using the trained model, you can use the make_pred.py script. Pass a CSV file containing the input data as an argument:

```
python make_pred.py input_data.csv
```
For now only encoded values are accepted. The predicted labels for each input 
sample will be printed on the console.

## Results

### Best Validation Score

The best validation score achieved during the training process is F1-score 
(macro) of **0.992** at epoch 117. The model checkpoint with this score is 
saved in the `experiments/best` directory.

### Model Parameters and Experiment Results

| Num  | learning_rate | batch_size | num_epochs | best f1-macro |
|------|---------------|------------|------------|---------------|
| 1    | 0.05          | 32         | 256        | 0.554         |
| 2    | 0.05          | 8          | 256        | 0.8125        |
| 3    | 0.05          | 128        | 256        | 0.844         |
| 4    | 0.02          | 32         | 256        | 0.856         |
| 5    | 0.2           | 32         | 256        | 0.476         |
| 6    | 0.9           | 4096       | 8192       | 0.602         |
| 7    | 0.99          | 4096       | 8192       | 0.546         |
| 8    | 1.0           | 32         | 256        | 0.313         |
 | best | 0.01          | 256        | 120        | 0.992         |

Also, there were some experiments with model layers, dropout, regularization, 
but they were not recorded.