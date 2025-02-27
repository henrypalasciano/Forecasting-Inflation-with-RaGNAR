# RaGNAR: Forecasting UK Consumer Price Inflation with Random Generalised Network Autoregressive Processes

This repository contains the Python code used in our article:

**"Forecasting UK Consumer Price Inflation with RaGNAR: Random Generalised Network Autoregressive Processes"**  
*(Submitted to the **International Journal of Forecasting**).*

---

## Overview  
We forecast **monthly inflation in the United Kingdom** using **RaGNAR**, an ensemble of **Generalised Network Autoregressive (GNAR) processes** fitted to a set of **random networks** generated according to the **Erdős–Rényi–Gilbert model**. Nodes represent the **Consumer Price Index (CPI)** and its sub-components (*divisions, classes, and groups*).

---

## Data Source  
The CPI data is publicly available from the UK **Office for National Statistics (ONS)**:  
🔗 [ONS Consumer Price Indices Dataset](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceindices)  

For convenience, the script **`download_data.py`** (located in the **`methodology/`** folder) automatically **downloads and prepares the dataset**.

---

## Methodology  
1. Each month, the set of graphs is ranked according to the forecasting performance at the CPI node.  
2. The best-performing graphs are selected and used to forecast inflation.  
3. Multiple forecasts from different GNAR processes** and graphs are averaged to produce robust inflation predictions.  

---

