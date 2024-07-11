# Industrial Copper Modeling

## 📜 Project Overview

**Industrial Copper Modeling** is a machine learning project focused on addressing challenges in the copper industry related to sales and pricing. The project involves developing a regression model to predict copper item selling prices and a classification model to classify leads as "WON" (Success) or "LOST" (Failure). Additionally, an interactive web application is created using Streamlit for users to input data and get predictions.

---

## 🎯 Objectives

- **Predict Selling Price:** Create a regression model to predict the selling price of copper items.
- **Classify Lead Status:** Develop a classification model to determine if leads are successful ("WON") or unsuccessful ("LOST").
- **Build Streamlit App:** Design a web application that allows users to input data and receive predictions for selling prices and lead status.

---

## 🛠️ Skills and Tools

### Skills

- **Python Scripting:** Writing Python code for data analysis and machine learning.
- **Data Preprocessing:** Techniques for handling missing values, outlier detection, and data normalization.
- **Exploratory Data Analysis (EDA):** Visualizing data to uncover patterns and insights.
- **Streamlit:** Building interactive web applications for machine learning models.

### Tools

- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit
- **Environment:** Python 3.x

---

## 📊 Domain

**Manufacturing:** The project addresses challenges in the copper industry, focusing on sales and pricing issues.

---

## 📝 Problem Statement

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. This project aims to solve these problems using machine learning techniques:

1. **Explore skewness and outliers** in the dataset.
2. **Transform the data** into a suitable format and perform necessary cleaning and pre-processing.
3. **Build a regression model** to predict the continuous variable `Selling_Price`.
4. **Create a classification model** to predict the lead status as either "WON" or "LOST".
5. **Develop a Streamlit application** for users to input data and receive predictions.

---

## 📂 Data

### Data Description

- **id:** Unique identifier for each transaction or item.
- **item_date:** Date of the transaction or item record.
- **quantity_tons:** Quantity of the item in tons.
- **customer:** Identifier of the customer.
- **country:** Country associated with each customer.
- **status:** Current status of the transaction (e.g., Draft, WON, LOST).
- **item_type:** Type or category of the item.
- **application:** Specific use or application of the items.
- **thickness:** Thickness of the items.
- **width:** Width of the items.
- **material_ref:** Reference or identifier for the material used.
- **product_ref:** Reference or identifier for the specific product.
- **delivery_date:** Expected or actual delivery date.
- **selling_price:** Price at which the items are sold.

---

## 🛠️ Approach

### 1. Data Understanding

- **Identify variable types** and distributions.
- **Clean data** by converting invalid `Material_Reference` values to null.

### 2. Data Preprocessing

- **Handle missing values** using mean/median/mode.
- **Treat outliers** using IQR or Isolation Forest.
- **Address skewness** with transformations like log or Box-Cox.
- **Encode categorical variables** using one-hot or label encoding.

### 3. Exploratory Data Analysis (EDA)

- **Visualize data** using Seaborn’s boxplots, histograms, and violin plots.

### 4. Feature Engineering

- **Engineer new features** and drop highly correlated columns using a heatmap.

### 5. Model Building and Evaluation

- **Split data** into training and testing sets.
- **Train and evaluate models**: Regression for `Selling_Price` and Classification for `Status`.
- **Optimize models** using cross-validation and grid search.

### 6. Model GUI

- **Develop a Streamlit app** for user inputs and predictions.

### 7. Tips

- **Use the `pickle` module** to save and load models.

---

## 🧩 Learning Outcomes

1. **Proficiency in Python** and data analysis libraries.
2. **Experience in data preprocessing** and handling missing values.
3. **Skills in EDA** for visualizing and understanding data.
4. **Application of machine learning techniques** for regression and classification.
5. **Model building and optimization** using evaluation metrics and hyperparameter tuning.
6. **Feature engineering** to improve model performance.
7. **Web application development** using Streamlit for machine learning models.
8. **Understanding manufacturing domain challenges** and machine learning solutions.

---

## 🏆 Project Evaluation Metrics

- **Modular Code:** Code organized into functional blocks.
- **Maintainability:** Code should be maintainable as the codebase grows.
- **Portability:** Code should work across different environments.
- **GitHub Repository:** Public repository with a proper README.
- **Readme File:** Comprehensive documentation including project details and instructions.
- **Coding Standards:** Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards.
- **Demo Video:** Create and post a demo video on LinkedIn.

---

## 🚀 Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/SETHU0010/Industrial_Copper_Modeling.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd Industrial_Copper_Modeling
    ```

3. **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

5. **Access the web application at** [http://localhost:8501](http://localhost:8501)

---

## 📄 Requirements

Ensure the following Python libraries are installed:

```plaintext
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
