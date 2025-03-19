# IAEmbarque_failure_prediction

This student project focuses on developing a **deep neural network (DNN)** for **predictive maintenance**, utilizing the **AI4I 2020 Predictive Maintenance Dataset**. The goal is to export and optimize this model for **execution on an STM32L4R9 microcontroller** using **STM32CubeIDE**. This project covers the complete **embedded machine learning development cycle**, from data preprocessing to final deployment.

---

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Model Development](#model-development)
   - [Neural Network Architecture](#model-development)
   - [Training and Optimization](#model-development)
   - [Performance Evaluation](#model-development)
3. [Deployment on STM32L4R9](#deployment-on-stm32l4r9)

   
## Data Preprocessing

The **AI4I 2020 Predictive Maintenance Dataset** contains sensor readings and operational conditions of industrial machinery. It is used to predict potential failures based on key metrics such as temperature, vibration, and pressure.

#### Step 1: Failure Detection Model
- The first model is a **binary classification model** that predicts whether a machine is functioning normally (`0`) or has failed (`1`).
- **Inputs (X):**
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
- **Output (y):**
  - Machine failure (Binary: `0` = No failure, `1` = Failure)

![Distribution of machine failures](screens/distribution_machine_failures.png)

We have a class imbalance issue, as we have 10,000 machines that are functioning correctly and fewer than 500 machines that are not functioning. We are going to have a bias toward the majority class. This may lead to a model that is very good at predicting functioning machines but poorly at identifying failures.

#### Step 2: Failure Type Classification Model
- If a failure is detected, a second model identifies the **type of failure**.
- **Output labels (y):**
  - **TWF** (Tool Wear Failure)
  - **HDF** (Heat Dissipation Failure)
  - **PWF** (Power Failure)
  - **OSF** (Overstrain Failure)
  - **RNF** (Random Failure)

![Distribution of types machine failures](screens/distribution_Type_failures.png)

We have some features like RNF and TWF classes that hasn't a lot of examples

## Model Development
### Neural Network Architecture

The first model is a **binary classification model** designed to predict whether a machine will fail. It consists of:

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(32, activation='relu'),  
    Dense(16, activation='relu'), 
    Dense(1, activation='sigmoid') # Output layer with a single neuron and sigmoid activation for binary classification (`0`: No failure, `1`: Failure).
])
```

### Performance Evaluation

<p align="center">
  <img src="screens/precision.png" width="45%" alt="Accuracy Curve">
  <img src="screens/loss.png" width="45%" alt="Loss Curve">
</p>

The **training accuracy** reaches **95%**, indicating strong generalization, while the **loss curve** shows proper convergence without significant overfitting.

The confusion matrix below evaluates the **performance of the failure detection model**:

![Confusion Matrix](screens/confusion_matrix.png)

- **True Negatives (1932 cases)**: The model correctly identified 1932 machines as **not failing**.
- **False Negatives (66 cases)**: The model **missed** 66 actual failures, predicting them as non-failures.
- **False Positives (0 cases)**: The model never predicted a failure when there was none (which is ideal).
- **True Positives (2 cases)**: The model correctly identified 2 machine failures.

```
               precision    recall  f1-score   support

           0       0.98      1.00      0.99      1932
           1       0.70      0.28      0.40        68

    accuracy                           0.97      2000
   macro avg       0.84      0.64      0.69      2000
weighted avg       0.97      0.97      0.97      2000
```

These results demonstrate **excellent performance**, with a high overall accuracy of **97%**.

## Deployment on STM32L4R9
- Model conversion for embedded execution
- Integration into **STM32CubeIDE**
- Testing and validation on the microcontroller


In a 2nd part, we implement a model which tries to find the failure among 5 different failures 
