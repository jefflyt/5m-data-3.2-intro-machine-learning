# Data Science Intersection
How do we learn Machine Learning?

<img src="assets/dsintersect.jpg" width="50%" alt="Data Science Intersection">

# Linear Regression
- A statistical / machine learning model for regression.
- Model the relationship between a dependent variable and one or more independent variables.
- Equation: $$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_nX_n + \epsilon $$
  - Dependent variable: $$ Y $$
  - Independent variables: $$ X_1, X_2, \ldots, X_n $$
  - Coefficients: $$ \beta_0, \beta_1, \beta_2, \ldots, \beta_n $$
  - Error term: $$ \epsilon $$
  ![linear_reg](assets/linear_reg.jpeg)
 
- Assumptions: 
  - Linear relationship between dependent and independent variables
  - No multicollinearity among independent variables
  - Homoscedasticity: Constant variance of errors
  - Normal distribution of errors

# Let's get an intuition of the linear regression process
[Click here for external link](https://developers.google.com/machine-learning/crash-course/linear-regression/parameters-exercise)

# Linear Regression Implementation Example
Below is a complete example of implementing linear regression with scikit-learn:

```python
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('data.csv')
df = df.dropna()  # Handle missing values

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model (minimizes MSE loss function)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Visualize results
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

# Display model coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

## Training / Fitting the Model

### Cost function

- Definition: A measure of how far off our predictions are from the actual values.

- Mean Squared Error (MSE): $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 $$
  - Actual value: $$y_i$$
  - Predicted value: $$ \hat{y_i} $$
  - Number of observations: $$ n $$

### Objective

- The goal is to find the values of the coefficients that minimize the sum of the squared differences between the observed values and the values predicted by the linear equation.
    - **Ordinary Least Squares (OLS)**: OLS is the most common method used in linear regression to estimate the coefficients. Often involves solving a set of linear equations or using optimization algorithms.
    - **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively moving towards the minimum.

## Evaluating the Model

### R-Squared (R²) in Linear Regression

R-squared, often denoted as R², is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables in a regression model.

#### Definition and Formula

R² is defined as the ratio of the variance explained by the model to the total variance. It is calculated as:

$$ R^2 = 1 - \frac{\text{Sum of Squares of Residuals (SSR)}}{\text{Total Sum of Squares (SST)}} $$

where,
- SSR (Sum of Squares of Residuals) is the sum of the squares of the model residuals.
- SST (Total Sum of Squares) is the total sum of the squares of the difference from the mean.

#### Interpretation

- **Value Range**: R² ranges from 0 to 1.
- **Higher R²**: A higher R² indicates a better fit of the model. An R² of 1 means that the regression predictions perfectly fit the data.
- **Limitation**: While a high R² indicates a good fit, it doesn't guarantee that the model is appropriate. Overfitting, where the model is too complex, can lead to misleadingly high R² values.

![sst_ssr](assets/sst_ssr.png)

### Sum of Squares in Linear Regression

Sum of Squares is a measure used in statistical analysis to quantify variations in data points.

#### 1. Total Sum of Squares (SST)

SST measures the total variation in the dependent variable. It is calculated as:

$$ \text{SST} = \sum_{i=1}^{n} (y_i - \bar{y})^2 $$

where $ y_i $ is the actual value, $ \bar{y} $ is the mean of the dependent variable, and $ n $ is the number of observations.

#### 2. Sum of Squares of Residuals (SSR)

SSR, also known as the sum of squared errors of the model, measures the unexplained variation by the model. It is calculated as:

$$ \text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

where $ \hat{y}_i $ is the predicted value.

#### Importance in Regression Analysis

- **SST**: Reflects the total variation in the data.
- **SSR**: Indicates the amount of variation not explained by the model.
- **Balance**: A good model minimizes SSR and explains most of the SST.

Sum of Squares helps in understanding the effectiveness of the regression model in explaining the variance of the data. The lower the SSR in comparison to SST, the better the model's predictive capabilities.

## Scikit-Learn Tutorial

Let's implement linear regression in Python using the Scikit-Learn (`sklearn`) library.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

Let's create a synthetic dataset using NumPy.

```python
np.random.seed(0)
# Array of 100 values with mean = 1.5, stddev = 2.5
X = 2.5 * np.random.randn(100) + 1.5
# Generate 100 residual terms
res = 0.5 * np.random.randn(100)
# Actual values of Y (equation)
y = 2 + 0.3 * X + res
```

```python
# Convert X and y into a pandas DataFrame
df = pd.DataFrame({
    'X': X,
    'y': y
})
```

```python
# It's often useful to visualize the data before building models.
plt.scatter(df['X'], df['y'])
plt.title('Dataset')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Before we train (fit) the model, we'll need to split the dataset into `training` and `testing` sets. The split is an `80:20` (which can be arbitrary).

The reason for splitting the dataset is to avoid overfitting. Overfitting occurs when the model learns the training data too well, including the noise, and is unable to generalize to new  / unseen data. We will fit the model on the training set and evaluate it on the testing set (unseen data).

```python
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'], test_size=0.2, random_state=0)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
```

```python
X_train.shape
```

```python
X_test.shape
```

Now, let's train (fit) the model.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

```python
# Use the model to make predictions on the test set.
y_pred = model.predict(X_test)
```

```python
# Evaluate the model's performance on the test set using metrics like Mean Squared Error (MSE) and R-squared.

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
```

```python
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

## Statsmodels Tutorial

We can also implement linear regression using the `statsmodels` library.

```python
import statsmodels.api as sm
```

Prepare the predictor and response variables, for statsmodels we need to add a constant term to the predictor.

```python
X = df['X']
y = df['y']
X = sm.add_constant(X)  # Adds a constant term to the predictor
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

```python
model = sm.OLS(y_train, X_train).fit()
```

Statsmodels provides extensive statistical diagnostics and tests about the model, such as _p-values, confidence intervals, and in-depth analysis of residuals._

```python
model.summary()

# Regular R² always increases when adding more predictors, even if they're irrelevant
# Adjusted R² penalizes adding unnecessary predictors
```

```python
# Use the model to make predictions on the test set.
y_pred = model.predict(X_test)
```

```python
# Evaluate the model's performance on the test set using metrics like Mean Squared Error (MSE) and R-squared.

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
```

The results are the same as sklearn's model above.

## Exercise

Now, use the statsmodels library to perform linear regression on the "diamonds" dataset, which contains the prices and attributes of almost 54,000 diamonds. We'll try to predict the price of a diamond based on its carat weight.

```python
import seaborn as sns
```

```python
diamonds = sns.load_dataset('diamonds')
diamonds.head()
```

```python
X = diamonds['carat']
y = diamonds['price']
# Adds a constant term to the predictor
X = sm.add_constant(X)
```

```python
# let's use 30% for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python
model = sm.OLS(y_train, X_train).fit()
```

```python
model.summary()
```

```python
y_pred = model.predict(X_test)
```

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
```

```python
plt.scatter(X_test['carat'], y_test, alpha=0.3, label='Actual Price')
plt.plot(X_test['carat'], y_pred, color='red', label='Predicted Price')
plt.title('Diamond Price Prediction')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.legend()
plt.show()
```

# Logistic Regression
- A statistical / machine learning model for classification, specifically binary (`positive or 1` class vs `negative or 0` class) classification.
- Model the probability that a given input point belongs to the positive class (negative probability is just `1 - positive probability`)
- Equation: $$ P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}} $$
  - Uses the `logistic function (Sigmoid function)` to model the linear predicted output (from a linear regression) into a binary output variable.
  - The logistic function converts any input to a value between 0 and 1, which is interpreted as a probability of the instance belonging to the positive class.
  ![log_reg](assets/log_reg.png)

- Concepts:
  - Odds: $$ \frac{P(Y=1)}{1 - P(Y=1)} $$
  - In the context of probability, "odds" is a way of expressing the likelihood that a particular event will occur. Odds are calculated as the ratio of the probability of the event occurring to the probability of the event not occurring.
  - Log-Odds: $$ \log\left(\frac{P(Y=1)}{1 - P(Y=1)}\right) = \beta_0 + \beta_1X $$
  - Log odds, or the logit function, is the logarithm of the odds. In logistic regression, the logit function is used to create a linear combination of the independent variables.
  - The logistic function is the inverse of the logit function. It takes the log odds and transforms them back into a probability.

- Example: Binary classification tasks such as email spam detection, image recognition, etc.

## Training / Fitting the Model

### Maximum Likelihood Estimation

- Objective: Find the values of coefficients that maximize the `likelihood (function)` of observing the given sample.
- Likelihood Function: Measures how well our model explains the observed data.
- Procedure:
  1. Define the likelihood function for the logistic regression model.
  2. Use optimization techniques (e.g., gradient descent) to maximize the likelihood function.
  3. The values of coefficients that maximize the likelihood function are considered as the estimates.

## Scikit-Learn Tutorial
For this tutorial, we'll use the Iris dataset, which is a classic dataset for classification. The dataset contains 150 observations of iris flowers. 

![Three species of iris: setosa, versicolor, and virginica](assets/Three-species-of-IRIS-flower.jpg)

In this exercise, we will build a classifier to distinguish between two species of iris flowers: **Iris sentosa and Iris versicolor**. Using the measurements of sepal length, sepal width, petal length, and petal width, we'll train a model to accurately identify which species a flower belongs to.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```

```python
iris = load_iris()
X = iris.data
y = iris.target
```

We'll need to convert the problem to a binary classification for logistic regression. Drop the additional class.

```python
X = X[y != 2]
y = y[y != 2]
```

```python
X.shape
```

```python
y.shape
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

We'll use `Accuracy Score` to evaluate the performance of the model.

$$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Observations}} $$

> There are many more metrics to evaluate classification models, which we'll cover in the next unit.

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

```python
iris.feature_names
```

```python
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.xlabel(f'{iris.feature_names[0]}')
plt.ylabel(f'{iris.feature_names[1]}')
plt.title('Logistic Regression Model Predictions')
plt.show()
```

```python
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred)
plt.xlabel(f'{iris.feature_names[2]}')
plt.ylabel(f'{iris.feature_names[3]}')
plt.title('Logistic Regression Model Predictions')
plt.show()
```

## Statsmodels Tutorial

We can also implement logistic regression using the `statsmodels` library.

Again, the benefits of statsmodels is it provides extensive statistical diagnostics and tests about the model, such as _p-values, confidence intervals, and in-depth analysis of residuals._

Let's use another more realistic dataset such as the _Pima Indians Diabetes_ dataset.

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
pima = pd.read_csv(url, names=column_names)
```

```python
pima.head()
```

Let's try to predict the diabetes outcome using glucose, BMI and age.

```python
features = ['Glucose', 'BMI', 'Age']
X = pima[features]
y = pima['Outcome']
```

```python
# Adds a constant term to the predictor
X = sm.add_constant(X)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python
model = sm.Logit(y_train, X_train).fit()
```

```python
model.summary()
```

```python
y_prob = model.predict(X_test)
```

```python
# y_prob is an array of probabilities, we need to convert it to predicted class
y_pred = np.array([1 if x > 0.5 else 0 for x in y_prob])
```

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

# K-Nearest Neighbors (K-NN)
- A machine learning model for regression and classification.
- Algorithm:
  1. Compute the distance between the test data point and each of the training data points.
  2. Identify the 'k' nearest data points to the test data point.
  3. Perform a majority vote (for classification) or average (for regression) of these 'k' nearest points.
- Distance Metrics:
  - Euclidean Distance
  - Manhattan Distance
  - Minkowski Distance
 
**Euclidean Distance in coordinate notation**
![Euclidean Distance (in coordinate space)](assets/distance.png)

**Euclidean distance in vector notation**
The Euclidean distance between two points in Euclidean space is a measure of the length of a straight line between these two points. In a space with $ p $ dimensions, the Euclidean distance between two points, $ P $ and $ Q $, with coordinates $ P = (p_1, p_2, \ldots, p_p) $ and $ Q = (q_1, q_2, \ldots, q_p) $, is given by the formula:

$$ d(P, Q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \cdots + (p_p - q_p)^2} $$

In a more generalized and compact form, it can be written as:

$$ d(P, Q) = \sqrt{\sum_{i=1}^{p} (p_i - q_i)^2} $$

where $ \sum $ denotes the summation and $ \sqrt{} $ denotes the square root.

> We'll dive deeper into `Manhattan` and `Minkowski` Distance in Unit 3.6: Unsupervised Learning.
- Example: If k=3, and the three closest points to a test instance are of classes A, A, and B, the algorithm would classify the test instance as class A.

![knn](assets/knn.png)

Choosing the Right 'k' in K-NN
- Small 'k' value:
  - Pros: Captures the nuances in the data.
  - Cons: Sensitive to noise in the data; higher variance.
- Large 'k' value:
  - Pros: More stable and robust to noise.
  - Cons: May oversimplify the model; higher bias.

> We'll dive deeper into this in Unit 3.5: Model Evaluation

## Scikit-Learn Tutorial

Let's apply KNN on the `diamonds` regression problem.

```python
from sklearn.neighbors import KNeighborsRegressor
```

```python
X = diamonds['carat']
y = diamonds['price']
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
```

```python
k = 5
knn_reg = KNeighborsRegressor(n_neighbors=k)
knn_reg.fit(X_train, y_train)
```

```python
y_pred = knn_reg.predict(X_test)
```

```python
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

```python
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
```

We can experiment with different `k` values and see how they affect the performance.

```python
errors = []
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred))

# Plotting
plt.plot(range(1, 21), errors)
plt.title('k-NN Varying number of neighbors (k)')
plt.xlabel('k')
plt.ylabel('Mean Squared Error')
plt.show()
```

## Exercise

Now, let's practice on the `iris` dataset for classification, without dropping class `2`. Hence, this will be a multi-class (>2) classification

```python
from sklearn.neighbors import KNeighborsClassifier
```

```python
iris = load_iris()
X = iris.data
y = iris.target
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

```python
print("Accuracy Score:", accuracy_score(y_test, y_pred))
```

# Let's revisit Linear regression for the diamond dataset
Addressing OLS assumptions and potential issues (non-normal residuals, heteroscedasticity, and model specification)

```python
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

#diamonds = sns.load_dataset('diamonds')
#diamonds.head()
X = diamonds['carat']
y = diamonds['price']
# Adds a constant term to the predictor
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = sm.OLS(y_train, X_train).fit()
```

### 1. Residual normality diagnostics

```python
residuals = model.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax[0])
ax[0].set_title('Residual Histogram')
qqplot(residuals, line='s', ax=ax[1])
ax[1].set_title('Q-Q Plot')

jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals)
print(f"Jarque–Bera: stat={jb_stat:.2f}, p-value={jb_pvalue:.5f}, skew={skew:.3f}, kurtosis={kurt:.3f}")
```

### 2. Heteroscedasticity test & robust standard errors

```python
_, bp_pvalue, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"Breusch–Pagan p-value: {bp_pvalue:.5f}")

robust_model = model.get_robustcov_results(cov_type='HC3')
print(robust_model.summary())

plt.figure(figsize=(6, 4))
plt.scatter(model.fittedvalues, residuals, alpha=0.3)
plt.axhline(0, color='red', lw=1)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
```

### 3. Improved model: add quadratic term for carat

```python
if 'carat2' not in diamonds.columns:
    diamonds['carat2'] = diamonds['carat'] ** 2
X_poly = diamonds[['carat', 'carat2']]
X_poly = sm.add_constant(X_poly)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.3, random_state=0)
```

```python
# Fit the polynomial model
poly_model = sm.OLS(y_train_poly, X_train_poly).fit()
```

```python
# a. Residual Normality Diagnostics
residuals_poly = poly_model.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals_poly, kde=True, ax=ax[0])
ax[0].set_title('Residual Histogram (Polynomial Model)')
qqplot(residuals_poly, line='s', ax=ax[1])
ax[1].set_title('Q-Q Plot (Polynomial Model)')
plt.tight_layout()
plt.show()
```

```python
# b. Jarque-Bera test for normality
jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals_poly)
print(f"\nJarque–Bera Test (Polynomial Model):")
print(f"  Statistic: {jb_stat:.2f}, p-value: {jb_pvalue:.5f}")
print(f"  Skewness: {skew:.3f}, Kurtosis: {kurt:.3f}")
```

```python
# c. Heteroscedasticity Test
_, bp_pvalue, _, _ = het_breuschpagan(residuals_poly, poly_model.model.exog)
print(f"\nBreusch–Pagan Test (Polynomial Model): p-value = {bp_pvalue:.5f}")
```

```python
# d. Robust Standard Errors
robust_poly = poly_model.get_robustcov_results(cov_type='HC3')
print("\nPolynomial Model with Robust Standard Errors (HC3):")
print(robust_poly.summary())
```

```python
# e. Model Performance
y_pred_poly = poly_model.predict(X_test_poly)
print('\nModel Performance:')
print(f"MSE: {mean_squared_error(y_test_poly, y_pred_poly):.2f}")
print(f"R²: {r2_score(y_test_poly, y_pred_poly):.4f}")
```

```python
# f. Residuals vs Fitted Plot
plt.figure(figsize=(8, 6))
plt.scatter(poly_model.fittedvalues, residuals_poly, alpha=0.3)
plt.axhline(0, color='red', lw=1)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (Polynomial Model)')
plt.show()
```

### Model Assessment

Comparing our linear and polynomial models:

- The polynomial model better captures the non-linear relationship between carat and price
- Both models still show non-normal residuals and heteroscedasticity
- The quadratic term improves the model fit as seen in the R-squared values

### 4. Log-Transformed Model to Address Non-Normality and Heteroscedasticity

```python
# Let's examine the relationship between carat and price in the log-transformed space.
# Create log-transformed variables
diamonds['log_price'] = np.log1p(diamonds['price'])
diamonds['log_carat'] = np.log(diamonds['carat'])
plt.figure(figsize=(12, 5))

# Original scale
plt.subplot(1, 2, 1)
plt.scatter(diamonds['carat'], diamonds['price'], alpha=0.3)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Price vs Carat (Original Scale)')

# Log-transformed scale
plt.subplot(1, 2, 2)
plt.scatter(diamonds['log_carat'], diamonds['log_price'], alpha=0.3, color='orange')
plt.xlabel('log(Carat)')
plt.ylabel('log(Price)')
plt.title('log(Price) vs log(Carat)')

plt.tight_layout()
plt.show()
```

```python
# Let's examine the relationship between carat and price in the log-transformed space.
# Create log-transformed variables
diamonds['log_price'] = np.log1p(diamonds['price'])
diamonds['log_carat'] = np.log(diamonds['carat'])
plt.figure(figsize=(12, 5))

# Original scale
plt.subplot(1, 2, 1)
plt.scatter(diamonds['carat'], diamonds['price'], alpha=0.3)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Price vs Carat (Original Scale)')

# Log-transformed scale
plt.subplot(1, 2, 2)
plt.scatter(diamonds['log_carat'], diamonds['log_price'], alpha=0.3, color='orange')
plt.xlabel('log(Carat)')
plt.ylabel('log(Price)')
plt.title('log(Price) vs log(Carat)')

plt.tight_layout()
plt.show()
```

```python
# Prepare data for log model
X_log = diamonds[['log_carat']]
X_log = sm.add_constant(X_log)
y_log = diamonds['log_price']

# Split the data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.3, random_state=0
)

# Fit the log model
log_model = sm.OLS(y_train_log, X_train_log).fit()
```

```python
# a. Residual Normality Diagnostics
residuals_log = log_model.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals_log, kde=True, ax=ax[0])
ax[0].set_title('Residual Histogram (Log Model)')
qqplot(residuals_log, line='s', ax=ax[1])
ax[1].set_title('Q-Q Plot (Log Model)')
plt.tight_layout()
plt.show()
```

```python
# b. Heteroscedasticity Test
_, bp_pvalue_log, _, _ = het_breuschpagan(residuals_log, log_model.model.exog)
print(f"\nBreusch–Pagan Test (Log Model): p-value = {bp_pvalue_log:.5f}")
```

```python
# c. Model Performance
y_pred_log = log_model.predict(X_test_log)

# Original metrics in log scale
# log_mse_logscale = mean_squared_error(y_test_log, y_pred_log)
# log_r2_logscale = r2_score(y_test_log, y_pred_log)

# Metrics in original price scale for fair comparison
y_test_original = np.expm1(y_test_log)
y_pred_original = np.expm1(y_pred_log)
log_mse = mean_squared_error(y_test_original, y_pred_original)
log_r2 = r2_score(y_test_original, y_pred_original)
```

```python
# d. Residuals vs Fitted Plot
plt.figure(figsize=(8, 6))
plt.scatter(log_model.fittedvalues, residuals_log, alpha=0.3)
plt.axhline(0, color='red', lw=1)
plt.xlabel('Fitted Values (log scale)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (Log Model)')
plt.show()
```

```python
# Final Model Comparison Table
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
# Calculate RMSE for each model
rmse_simple = mean_squared_error(y_test, model.predict(X_test), squared=False)
rmse_quad = mean_squared_error(y_test_poly, y_pred_poly, squared=False)
rmse_log = mean_squared_error(y_test_original, y_pred_original, squared=False)

print(f"{'Model':<15} {'R²':>10} {'MSE':>15} {'RMSE':>12} {'BP p-value':>15} {'Skewness':>10}")
print("-" * 85)

# Simple Linear
print(f"{'Simple Linear':<15} {r2_score(y_test, model.predict(X_test)):10.4f} "
      f"{mean_squared_error(y_test, model.predict(X_test)):15,.2f} "
      f"{rmse_simple:12,.2f} "
      f"{bp_pvalue:15.5f} {skew:10.4f}")

# Quadratic
print(f"{'Quadratic':<15} {r2_score(y_test_poly, y_pred_poly):10.4f} "
      f"{mean_squared_error(y_test_poly, y_pred_poly):15,.2f} "
      f"{rmse_quad:12,.2f} "
      f"{bp_pvalue:15.5f} {skew:10.4f}")

# Log-Transformed (Original Scale)
print(f"{'Log-Transformed':<15} {log_r2:10.4f} "
      f"{log_mse:15,.2f} "
      f"{rmse_log:12,.2f} "
      f"{bp_pvalue_log:15.5f} {stats.skew(residuals_log):10.4f}")

# Uncomment to see log-scale metrics for reference
# print(f"{'Log-Transformed*':<15} {log_r2_logscale:10.4f} "
#       f"{log_mse_logscale:15,.2f} "
#       f"{bp_pvalue_log:15.5f} {stats.skew(residuals_log):10.4f}")
# print("*Metrics in log scale for reference (not directly comparable)")

print("\nNote: Lower BP p-value indicates more evidence for heteroscedasticity")
print("      Skewness closer to 0 indicates more normal residuals")
```