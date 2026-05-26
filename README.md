# 🚀 MLOps: OCR Model Deployment & Load Testing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)
![Locust](https://img.shields.io/badge/Locust-Load%20Testing-red)
![Status](https://img.shields.io/badge/Status-Deployment%20&%20Testing-brightgreen)

Welcome to the **OCR Model Deployment & Load Testing** repository! This project showcases the deployment of a Machine Learning model (OCR) via a FastAPI server and the rigorous load testing conducted using Locust to evaluate its scalability and performance bottlenecks.

## 📖 Overview

The goal of this project is to take a trained OCR model, wrap it in a robust API using FastAPI, and subject it to varying levels of concurrent user traffic. By doing this, we identify real-world performance limitations such as CPU saturation and inference latency.

## 📂 Repository Contents

*   **`app.py`**: The FastAPI server script exposing the `/predict` endpoint.
*   **`model.py` & `inference.py`**: The underlying neural network architecture and inference scripts for the OCR task.
*   **`locustfile.py`**: The configuration for simulating concurrent users sending POST requests.
*   **`benchmark.py` & `API_test.py`**: Scripts for local endpoint testing and benchmarking.
*   **`mapping.json`**: Character mapping for the OCR output.
*   **`requirements.txt`**: Python dependencies for deployment.
*   **`BSAI23034_lt-mlops.txt`**: The final output report generated from the load tests.
*   *(Note: The 470MB `best.ckpt` model weights file is excluded from this repository due to size limits).*

## 📊 Load Testing Outputs & Analysis

We conducted multiple load tests using Locust to evaluate server stability under stress. The tests revealed significant bottlenecks in CPU saturation due to OCR inference latency. 

Here is a summary of the test outputs:

### Test 1: 2 Users (Moderate Load)
*   **Total Requests**: 10
*   **Failures**: 0
*   **Average Response Time**: 7.47s
*   **Conclusion**: **MODERATELY STABLE**. The system handles minimal concurrent users, though inference time is high.

### Test 2: 5 Users (High Load, 120s)
*   **Total Requests**: 22
*   **Failures**: 15
*   **Average Response Time**: 19.36s
*   **Conclusion**: **UNSTABLE**. Significant failures (Error 500) observed due to server overload and concurrent OCR request piling.

### Test 3: 10 Users (Stress Test)
*   **Total Requests**: 48
*   **Failures**: 48 (100% Failure Rate)
*   **Average Response Time**: 17.80s
*   **Conclusion**: **UNSTABLE**. The CPU completely saturates, leading to massive bottlenecks and request timeouts.

## 🛠️ Setup & Usage

To run the API locally:
1. Clone this repository: `git clone https://github.com/valiant-034/MLOps-OCR-Deployment-LoadTesting.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn app:app --host 0.00.0.0 --port 8000`
4. Run load tests: `locust -f locustfile.py`
