Sales Prediction Using Python

Overview

This project aims to predict sales based on advertising data using machine learning techniques. The dataset explores the correlation between advertising expenditure on TV and sales. The notebook includes data visualization, preprocessing, model training, and evaluation.

Dataset

The dataset includes variables such as:

TV (Advertising expenditure in TV)

Sales (Total sales revenue)

The dataset indicates a strong correlation between TV advertising and sales, making it suitable for linear regression.

Technologies & Libraries Used

The following Python libraries were used:

numpy (Numerical computing)

pandas (Data manipulation and analysis)

matplotlib, seaborn (Data visualization)

sklearn.model_selection (Train-test split)

sklearn.linear_model.LinearRegression (Linear Regression Model)

sklearn.metrics (Model Evaluation: MSE, R² Score)

Usage Instructions

Open Google Colab.

Upload the SalesDataset.ipynb file.

Run the notebook cell by cell.

Ensure you have the required libraries installed using:

!pip install numpy pandas matplotlib seaborn scikit-learn

Follow the steps in the notebook to preprocess data, train the model, and evaluate performance.

Model Training & Evaluation

The dataset is split into training and testing sets.

A Linear Regression model is used for sales prediction.

The model's performance is evaluated using Mean Squared Error (MSE) and R² Score.

Results & Findings

Sales are highly correlated with TV advertising expenditure.

Linear Regression effectively models this relationship.

The trained model achieves a certain level of accuracy in predicting sales based on TV spending.

Author

This project was implemented on Google Colab as part of a sales prediction initiative.

License

This project is open-source and available for further improvements and modifications.
