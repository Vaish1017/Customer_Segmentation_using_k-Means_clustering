# Customer Segmentation Using K-Means Clustering

## 📌 Project Overview
Customer segmentation is the process of dividing customers into distinct groups 
based on their purchasing behavior and demographic information.

In this project, K-Means clustering is applied to identify different customer 
segments using income and spending behavior. This helps businesses understand 
their customers better and design targeted marketing strategies.

---

## 🎯 Objective
- Identify distinct customer groups
- Understand purchasing behavior
- Help businesses improve marketing decisions

---

## 📊 Dataset
- Dataset: Mall Customers Dataset
- Source: Kaggle

Features used:
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

Dataset link:  
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🔍 Exploratory Data Analysis
- Gender distribution analysis
- Income vs spending visualization
- Identification of customer behavior patterns

---

## ⚙️ Methodology
1. Data loading and preprocessing  
2. Feature scaling using StandardScaler  
3. Determining optimal clusters using Elbow Method  
4. Applying K-Means clustering  
5. Visualizing customer segments  

---

## 📈 Model Used
**K-Means Clustering (Unsupervised Learning)**

- Optimal clusters selected: **5**
- Distance metric: Euclidean
- Initialization: k-means++

---

## 📌 Cluster Interpretation

| Cluster | Characteristics | Business Insight |
|--------|----------------|------------------|
| 0 | High income, high spending | Premium customers |
| 1 | Low income, low spending | Budget customers |
| 2 | High income, low spending | Potential customers |
| 3 | Low income, high spending | Impulsive buyers |
| 4 | Moderate income and spending | Regular customers |

---

## 📊 Results & Insights
- Customers can be clearly segmented into 5 distinct groups
- Spending behavior does not always increase with income
- Marketing strategies can be customized per cluster

---

## 🧪 Evaluation Metric
- Silhouette Score used to validate clustering performance

---

## 🚀 How to Run the Project
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
---
## ✅ Conclusion

This project demonstrates how unsupervised machine learning can be used to
identify meaningful customer segments. The insights gained can help businesses
improve targeting, personalization, and revenue optimization.

---

## 👤 Author

P Vaishnavi |
Aspiring Data Analyst | Machine Learning Enthusiast
---
