# Study Notes

This file is for taking notes on the supervised learning lessons and addendums.

# Chapter 1: Linear Regression

Linear Regression is a fundamental machine learning technique used for predicting continuous values, such as prices, temperatures, or quantities. It works by finding the best straight line that fits the relationship between input features (independent variables) and the output (dependent variable).

## Simple Idea
Imagine you have data points on a graph, like house prices vs. house sizes. Linear regression draws a straight line through these points that best represents the overall trend. This line can then predict the price of a new house based on its size.

## Key Components
- **Equation**: The basic form is `y = β₀ + β₁x + ε`, where:
  - `y` is the predicted output
  - `β₀` is the intercept (starting point)
  - `β₁` is the slope (how much y changes with x)
  - `x` is the input feature
  - `ε` is the error term (difference between prediction and reality)

- **Training**: The model learns by minimizing the Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values. Methods include Ordinary Least Squares (OLS) and Gradient Descent.

- **Evaluation**: 
  - R-squared (R²) measures how well the model explains the data (closer to 1 is better).
  - Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) quantify prediction accuracy.

## Assumptions
For linear regression to work well, the data should meet certain assumptions:
- Linear relationship between inputs and output
- No strong correlations between input features (no multicollinearity)
- Constant variance of errors (homoscedasticity)
- Normally distributed errors

## When to Use
- Predicting continuous outcomes
- Understanding relationships between variables
- Simple, interpretable models

## Example from Notebook
In the diamonds dataset, we used linear regression to predict diamond prices based on carat weight. The model found a line where price increases as carat increases, but we also explored improvements like adding polynomial terms or log transformations to handle non-linear relationships and issues like heteroscedasticity.

## Libraries and Tools Used
The notebook demonstrates linear regression using several Python libraries:

- **pandas**: For data manipulation and creating DataFrames (`import pandas as pd`)
- **numpy**: For numerical operations and array handling (`import numpy as np`)
- **matplotlib.pyplot**: For plotting and visualization (`import matplotlib.pyplot as plt`)
- **seaborn**: For loading datasets like diamonds (`import seaborn as sns`)
- **scikit-learn (sklearn)**: 
  - `LinearRegression` for fitting the model
  - `train_test_split` for splitting data into training and testing sets
  - `mean_squared_error` and `r2_score` for evaluating model performance
- **statsmodels**: For statistical modeling and detailed diagnostics (`import statsmodels.api as sm`), including `sm.OLS` for Ordinary Least Squares regression

## Scikit-Learn vs Statsmodels
Scikit-Learn and Statsmodels are both Python libraries for data analysis and modeling, but they serve different primary purposes:

### Scikit-Learn (sklearn)
- **Focus**: Machine learning and predictive modeling
- **Strengths**: 
  - Easy-to-use API for training models and making predictions
  - Emphasis on model performance metrics (accuracy, MSE, R²)
  - Supports a wide range of ML algorithms (classification, regression, clustering, etc.)
  - Optimized for large datasets and production use
- **Use Case**: When you want to build and deploy predictive models quickly
- **Example in Notebook**: Used for `LinearRegression().fit()` and evaluating with `mean_squared_error()` and `r2_score()`

### Statsmodels
- **Focus**: Statistical analysis and inference
- **Strengths**:
  - Detailed statistical diagnostics (p-values, confidence intervals, hypothesis tests)
  - Comprehensive model summaries with statistical significance
  - Tools for time series, econometrics, and advanced statistical tests
  - Better for understanding relationships and causality
- **Use Case**: When you need to interpret model coefficients, test assumptions, or perform statistical inference
- **Example in Notebook**: Used for `sm.OLS().fit()` which provides extensive summary tables with p-values, R², and diagnostic tests

### Key Differences
- **Output**: Scikit-Learn gives predictions and basic metrics; Statsmodels gives detailed statistical reports
- **Philosophy**: Scikit-Learn is about prediction; Statsmodels is about statistical understanding
- **Performance**: Scikit-Learn is faster for large-scale ML; Statsmodels excels at statistical depth

In practice, you might use both: Scikit-Learn for initial modeling and Statsmodels for deeper statistical analysis. For linear regression in the notebook, Scikit-Learn handles the basic fitting and prediction, while Statsmodels provides richer diagnostics like residual analysis and heteroscedasticity tests.

## Key Concepts

- 

## Models Covered

- 

## Code Examples

### Scikit-Learn Tutorial for Linear Regression (Step-by-Step)

The notebook demonstrates implementing linear regression using scikit-learn with a synthetic dataset. Here's a step-by-step breakdown:

1. **Import Required Libraries**:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error, r2_score
   ```
   - `pandas` and `numpy`: For data manipulation
   - `matplotlib.pyplot`: For visualization
   - `sklearn` modules: For the ML model, data splitting, and evaluation

2. **Create Synthetic Dataset**:
   ```python
   np.random.seed(0)
   X = 2.5 * np.random.randn(100) + 1.5
   res = 0.5 * np.random.randn(100)
   y = 2 + 0.3 * X + res
   ```
   - Generates 100 random X values with mean 1.5 and std 2.5
   - Creates residuals (noise) and calculates y = 2 + 0.3*X + noise

3. **Prepare Data as DataFrame**:
   ```python
   df = pd.DataFrame({'X': X, 'y': y})
   ```
   - Converts arrays to a pandas DataFrame for easier handling

4. **Visualize the Data**:
   ```python
   plt.scatter(df['X'], df['y'])
   plt.title('Dataset')
   plt.xlabel('X')
   plt.ylabel('y')
   plt.show()
   ```
   - Scatter plot to understand the relationship between X and y

5. **Split Data into Training and Testing Sets**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'], test_size=0.2, random_state=0)
   X_train = X_train.values.reshape(-1, 1)
   X_test = X_test.values.reshape(-1, 1)
   ```
   - 80% for training, 20% for testing
   - Reshape X to 2D array (required by sklearn)

6. **Train the Linear Regression Model**:
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```
   - Creates and fits the model using Ordinary Least Squares

7. **Make Predictions on Test Set**:
   ```python
   y_pred = model.predict(X_test)
   ```
   - Uses the trained model to predict y values for test data

8. **Evaluate Model Performance**:
   ```python
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   print("Mean Squared Error:", mse)
   print("R^2 Score:", r2)
   ```
   - MSE measures average squared prediction error
   - R² measures proportion of variance explained (0-1 scale)

9. **Visualize Results**:
   ```python
   plt.scatter(X_test, y_test)
   plt.plot(X_test, y_pred, color='red')
   plt.title('Linear Regression Model')
   plt.xlabel('X')
   plt.ylabel('y')
   plt.show()
   ```
   - Scatter plot of actual vs predicted values with regression line

This tutorial shows the complete workflow: data preparation, model training, prediction, evaluation, and visualization using scikit-learn's straightforward API.

## Questions and Answers

- 