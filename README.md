# **`Australian_Car_Price_Prediction`**
# **`GOAL`** 
The aim of this project is to use the given dataset to find insights about the data using Exploratory Data Analysis and create a model to predict prices of the cars. This is a beginners project that I have done to merely illustrate how the three supervised machine learning models- Linear Regression, Decision Tree Regressor and Random Forest Regressor are implemented in a project. I have used several metrics like R2 Score, root mean squared error, and mean absolute eror to evaluate each model and compare them to each other. The theory of linear regression is based on certain statistical assumptions. Before model fitting the models, I have checked these regression assumptions using the linear regression approach.

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
7. Model Evaluation

