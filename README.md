
<h1 align="center">Machine learning algorithm: Linear regression</h1>  

### Introduction  
Linear regression is a fundamental statistical method used for modeling the relationship between a dependent variable (target) and one or more independent variables (features). It is widely used for predictive analysis and can be applied to various fields including finance, biology, engineering, and social sciences.

### Key Concepts
Simple linear regression involves a single independent variable. The relationship between the dependent variable 
𝑦 and the independent variable x

### Linear Regression Equation

The equation for simple linear regression is:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

- \( y \): Dependent variable
- \( x \): Independent variable
- \( \beta_0 \): Intercept (the value of \( y \) when \( x = 0 \))
- \( \beta_1 \): Slope (the change in \( y \) for a one-unit change in \( x \))
- \( \epsilon \): Error term (residuals, representing the difference between the observed and predicted values)

The relationship between the dependent variable \( y \) and the independent variable \( x \) is modeled by the equation 

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

### Assumptions
Linear regression relies on several key assumptions:

1. Linearity: The relationship between the independent and dependent variables is linear.  
2. Independence: Observations are independent of each other.  
3. Homoscedasticity: The variance of residuals is constant across all levels of the independent variables.  
4. Normality: Residuals are normally distributed

### Metrics for model evaluation
**R-squared (\( R^2 \))**  
The R-squared value measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

**Mean Squared Error (MSE)**  
The MSE is the average of the squared differences between the observed and predicted values. It provides a measure of the model's accuracy.

**Adjusted R-squared** 
Adjusted R-squared adjusts the R-squared value based on the number of predictors in the model. It is useful for comparing models with different numbers of independent variables.

### Implementation in Python

#### Linear Regression Example in Python

Here is an example of how to perform linear regression using the `scikit-learn` library in Python:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')  




