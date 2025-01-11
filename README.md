# **`Australian_Car_Price_Prediction`**
# **`GOAL`** 
The aim of this project is to use the given dataset to find insights about the data using Exploratory Data Analysis and create a model to predict prices of the cars. This is a beginners project that I have done to merely illustrate how the supervised machine learning models- Linear Regression, Decision Tree Regressor, Random Forest Regressor nad Gradient Boosting Regressor are implemented in a project. I have used several metrics like R2 Score,adjusted R-squared, root mean squared error, and mean absolute error to evaluate each model and compare them to each other. The theory of linear regression is based on certain statistical assumptions. Before model fitting the models, I have checked these regression assumptions using the linear regression approach.

# **`ABOUT THE DATA`**
This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.
Source - https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices

## Attributes-
- Brand: Name of the car manufacturer
- Year: Year of manufacture or release
- Model: Name or code of the car model
- Car/Suv: Type of the car (car or suv)
- Title: Title or description of the car
- UsedOrNew: Condition of the car (used or new)
- Transmission: Type of transmission (manual or automatic)
- Engine: Engine capacity or power (in litres or kilowatts)
- DriveType: Type of drive (front-wheel, rear-wheel, or all-wheel)
- FuelType: Type of fuel (petrol, diesel, hybrid, or electric)
- FuelConsumption: Fuel consumption rate (in litres per 100 km)
- Kilometres: Distance travelled by the car (in kilometres)
- ColourExtInt: Colour of the car (exterior and interior)
- Location: Location of the car (city and state)
- CylindersinEngine: Number of cylinders in the engine
- BodyType: Shape or style of the car body (sedan, hatchback, coupe, etc.)
- Doors: Number of doors in the car
- Seats: Number of seats in the car
- Price: Price of the car (in Australian dollars)
# **`Tools Used`**
1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. Scikit-learn
# **`What we will do:`**
1. Reading the data
2. Data Cleaning
   - Dealing with irrelevant columns
   - Handling the missing data 
3. Exploratory Data Analysis
4. Checking for the assumptions of Linear Regression
   - Checking for the outliers
   - Checking for the non-linearity of the response-predictor relationship.
   - Checking for the presence of correlation between the error terms in the model.
   - Checking for heteroscedasticity.
   - Checking for the multicollinearity in the data.
   - Checking for any influential points in the datset.
   - Checking for the normality of the response variable.
5. Data Preparation
   - Splitting the data into train and test data
   - Rescaling the data
6. Model building
7. Feature Importance
8. Model Evaluation
9. Conclusion

## **`Steps Involved`**
### EDA & Pre-processing 
After dropping the redundant columns like model of the car, Colour, etc. I have calculated the percentage of missing values in each column. For the columns that had less null values I have removed those data points. For columns having higher percentage of missing values I have filled them with the mode of that respective column. This is because most of the columns like Engine used, Doors, Seats, etc. had categorical data. So use of mode for filling the missing values is the best suitable. 
I have made use of several univariate and bivariate plots to reveal hidden insights present in the data.
For univariate plots, I have used count plots that revealed that
-	The highest sales in the Australian market in the year 2023 is Toyota.  
-	Automatic transmission cars were sold in great numbers. 
-	Unleaded fuel was used by maximum number of cars.
  
For bivariate plots I have made use of scatterplots and boxplots which revealed that 
-	Cars with low prices cover short distances.
-	As the year increases the price of car also increases.
-	The brand Lamborghini has the highest price range while Hyundai has the lowest price range.

I have used heatmap to learn about the correlation between each features with the other and it summarized that some features have really high correlation with the others.
After these findings I have tried to check for the key assumptions of the regression model before going for the model building.
-	First I have eliminated outliers from the dataset using the IQR method. 
-	Then using the library statsmodels I have fitted a linear regression model with price as the response and the remaining variables as predictor.
-	In order to check the non-linearity relationship between the response and the predictors, I have made use of the scatter plot between residuals and fitted values, which revealed that there was some aspect of non-linearity in the data.
-	Durbin Watson test is then used to check for the presence of autocorrelation between the error terms in the model, which revealed that there was no autocorrelation since the DW test fetches a value of 2.
-	For checking the heteroscedasticity I have used Breush Pagan test which resulted in a p value that is very small, and hence the null hypothesis that heteroscedasticity is absent was rejected. 
-	For checking the multicollinearity in the data, I have used VIF, which showed that there are high values for certain features. 
-	A Q-Q plot has been drawn to investigate the presence of normality in the data. However the plot showed a deflection from the usual straight line concluding that there is some sense of non-normality of the response variable.

### Eliminating the problems before model building.
-	Box-Cox transformation has been used to resolve the problem concerning the non-normality of the response variable. 
-	Then I proceeded with this transformed Y variable and the original predictor variables.

### Model building 
For modelling I have gone for the Ridge Regression, Decision Trees, Random Forest and Gradient Boosting Regressor.
-	Since multicollinearity is present in the data, Ridge is a good choice because it helps handle that issue while maintaining the model’s interpretability. After the Box-Cox transformation, the normality and heteroscedasticity issues are mitigated, making Ridge suitable for predicting car prices.
-	Given that non-linearity was present in the data, Decision Trees are a good fit because they are capable of modelling complex relationships between the features and the target variable without requiring transformations like linear models.
-	Since the dataset is relatively large (having over 15,000 rows), Random Forest is suitable as it can effectively capture complex interactions and non-linear relationships. It’s also less prone to overfitting compared to a single Decision Tree, making it a strong candidate for improving predictive performance.
-	Gradient Boosting is particularly effective for datasets with complex patterns and non-linear relationships. It often achieves higher accuracy than Random Forest by iteratively improving the model’s performance. After applying the Box-Cox transformation to stabilize variance and normalize the target variable, Gradient Boosting can efficiently capture the remaining patterns in the data and boost predictive accuracy.

### Model Evaluation
Then I have evaluated and compared the models using the performance metrics such as Adjusted R2, RMSE and MAE. 
-	Adjusted R-squared penalizes the inclusion of irrelevant features that don't contribute to the model's performance. This ensures that you're only keeping predictors that significantly improve the model’s ability to predict car prices.
-	There might arise a situation where there are large deviations due to mispricing of a car significantly. In such cases RMSE can serve as a good metric for evaluation.
-	MAE helps to understand the average prediction error without heavily penalizing outliers, giving a clearer picture of overall model accuracy.
I achieved the best performance with Gradient Boosting having an Adjusted R2 of 0.91, RMSE OF 6.8 and MAE of 4.99.

# **`Conclusion`**
In conclusion my model can help determine the most accurate selling price for each car based on factors like brands, models, types, and other features of cars. 
-	This allows the dealership to offer competitive pricing while maximizing profit margins.
-	The model allows the dealership to offer data-backed, fair prices, increasing customer trust and satisfaction, which could result in repeat business and customer loyalty.
-	By leveraging the model, the business can gain insights into current market trends and competitor pricing strategies, allowing them to stay ahead in the competitive landscape of used car sales.

