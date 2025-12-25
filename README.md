# Federated Learning for Customer Churn Prediction
Privacy-Preserving Machine Learning with Apache Kafka and Streamlit

## Project Overview
This project implements a Federated Learning (FL) system for customer churn prediction. Multiple clients collaboratively train a machine learning model without sharing raw customer data. Each client trains a local model on its own data and sends only model parameters to a central aggregator, which computes a global model using Federated Averaging (FedAvg).

The system demonstrates how privacy-preserving machine learning can achieve performance close to centralized training while respecting data confidentiality constraints.

## Objectives
- Predict customer churn using a federated learning approach
- Avoid centralization of sensitive customer data
- Implement the FedAvg aggregation algorithm
- Build a distributed, scalable ML architecture
- Provide real-time monitoring through a dashboard

## Key Concepts
- Federated Learning
- Federated Averaging (FedAvg)
- Privacy-preserving machine learning
- Distributed systems with Apache Kafka
- End-to-end ML pipelines

## Technologies Used
- Python
- Scikit-learn
- Apache Kafka
- Docker and Docker Compose
- Streamlit
- NumPy and Pandas

## Machine Learning Pipeline
- Model: Logistic Regression
- Preprocessing:
  - One-Hot Encoding for categorical features
  - Standard Scaling for numerical features
- Evaluation Metric: ROC-AUC

Each client trains the model locally using an identical preprocessing pipeline to ensure feature space consistency.

## Federated Learning Workflow
1. The server initializes or waits for local model updates
2. Each client:
   - Loads its local data partition
   - Trains a local model
   - Sends model weights, intercept, and sample count to Kafka
3. The aggregator:
   - Collects updates from all clients
   - Computes a weighted average using FedAvg
4. The global model is published back to clients
5. The process repeats for multiple rounds

## Monitoring and Visualization
The Streamlit dashboard provides:
- ROC-AUC comparison between local and federated models
- Logs of federated learning rounds
- Kafka message flow visualization

## How to Run the Project
### Prerequisites
- Docker
- Docker Compose

### Start the system
docker-compose up --build

### Access the dashboard
http://localhost:8501

## Privacy Considerations
- Raw customer data never leaves client containers
- Only model parameters are exchanged
- Kafka messages contain no sensitive information
- The system is designed to be compliant with data protection regulations

## Results
The federated model achieves performance close to centralized training while significantly improving data privacy.

## Limitations and Future Work
- Secure aggregation to hide individual client updates
- Differential privacy to prevent information leakage
- Support for non-IID data distributions
- Extension to deep learning models
- Improved fault tolerance and asynchronous training

## Author
AI / Machine Learning Engineer

## License
This project is intended for educational and research purposes.

