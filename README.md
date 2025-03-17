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

#### Step 2: Failure Type Classification Model
- If a failure is detected, a second model identifies the **type of failure**.
- **Output labels (y):**
  - **TWF** (Tool Wear Failure)
  - **HDF** (Heat Dissipation Failure)
  - **PWF** (Power Failure)
  - **OSF** (Overstrain Failure)
  - **RNF** (Random Failure)

## Model Development
### Neural Network Architecture
- Description of the model structure

### Training and Optimization
- Model training process
- Optimization techniques used

### Performance Evaluation
- Metrics for model assessment


## Deployment on STM32L4R9
- Model conversion for embedded execution
- Integration into **STM32CubeIDE**
- Testing and validation on the microcontroller

