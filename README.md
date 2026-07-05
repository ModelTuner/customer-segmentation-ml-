## Customer Segmentation using Unsupervised Learning

This project focuses on customer segmentation using unsupervised machine learning techniques. The objective is to identify groups of customers with similar behavior and convert those groups into meaningful business personas that can support marketing and business decisions.

The project follows a complete machine learning workflow, including data preprocessing, dimensionality reduction, clustering, visualization, model saving, and report generation.

---

## Project Objectives

- Load and clean customer data
- Handle missing values and duplicate records
- Prepare features for clustering
- Reduce feature dimensions using PCA
- Find the optimal number of clusters using the Elbow Method and Silhouette Score
- Train a K-Means clustering model
- Visualize clustering results
- Generate business insights and customer personas
- Save trained models and reports

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## Project Structure

```
Customer_Segmentation/
│
├── data/
│   └── customer_data.csv
│
├── outputs/
│   ├── logs/
│   ├── models/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── clustering.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── utils.py
│   └── __init__.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Workflow

```
Customer Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Preparation
      │
      ▼
Feature Scaling
      │
      ▼
Principal Component Analysis (PCA)
      │
      ▼
Optimal Cluster Selection
(Elbow Method + Silhouette Score)
      │
      ▼
K-Means Clustering
      │
      ▼
Customer Personas
      │
      ▼
Reports & Visualizations
```

---

## How to Run

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Run the project.

```bash
python main.py
```

---

## Outputs

After the pipeline finishes, the following files are generated automatically.

### Plots

- Missing Value Heatmap
- Correlation Heatmap
- Feature Distribution Plots
- Elbow Method
- Silhouette Score
- PCA Explained Variance
- PCA Cluster Visualization
- Cluster Distribution

### Reports

- Customer Persona Report
- Business Insights Report
- Clustering Summary (JSON)

### Models

- Trained K-Means Model
- PCA Model
- Feature Scaler

---

## Business Insights

The clustering results can be used to identify different types of customers based on their purchasing behavior.

Some possible business actions include:

- Reward loyal customers with exclusive offers.
- Re-engage inactive customers through targeted campaigns.
- Personalize promotions for different customer segments.
- Improve retention by understanding customer behavior.

---

## Features

- Automatic data cleaning
- Missing value handling
- Duplicate removal
- One-hot encoding for categorical features
- Feature scaling
- PCA for dimensionality reduction
- K-Means clustering
- Elbow Method analysis
- Silhouette Score evaluation
- Automatic report generation
- Model serialization using Joblib
- Professional logging

---

## Notes

If no dataset is available in the **data** folder, the project automatically generates a sample retail dataset so the complete pipeline can still be executed.

---

## Future Improvements

Some ideas for extending this project include:

- Streamlit dashboard for interactive visualization
- Testing additional clustering algorithms
- Hyperparameter tuning
- Support for larger real-world datasets

---

## Author

**Kush Sharma**