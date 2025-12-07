# NSB-Amazon-and-bot
## project Structure 
```
├── __pycache__/                           # Compiled Python cache
├── .devcontainer/                         # Dev container setup
│   ├── devcontainer.json
│   └── Dockerfile
├── .github/                               # (Optional) GitHub workflows
├── .pytest_cache/                         # Pytest cache
├── .vscode/                               # VSCode settings
│   └── launch.json
├── path/                                  # (Custom folder, reserved for future use)
├── .gitignore                             # Git ignore rules
├── amazon_cleaned.csv                     # Cleaned dataset
├── amazon_products_sales_data_uncleaned.csv # Raw dataset
├── main.ipynb                             # Jupyter notebook for experimentation
├── main.py                                # Main data cleaning + ML pipeline
├── Makefile                               # Automation commands (optional)
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
└── test_main.py                           # Unit & system test cases
 ```    

## Import the Dataset
"amazon_products_sales_data_uncleaned.csv"- 38.6MB


## Inspect the Data

### Coding Part
##### Pandas and Polars respectively
 1. Display the first 5 rows using .head() to get a quick overview.
 2. Use .info() and .describe() to understand data types and summary statistics.
 3. Check for missing values and duplicates

### Analysis part: Data overview 
there are 42657 rows and 16 columns. All of the entires are string. The headers(the first row) are title, rating, number_of_reviews, bought_in_last_month, current/discounted_price, price_on_variant, listed_price, is_best_seller, is_sponsored, is_couponed, buy_box_availability, delivery_details, sustainability_badges, image_url, product_url, collected_at.

1. Title: name of the product; 0 null value 
2. Rating: Amazon ratings in the format of "X.X out of 5 stars"; 1024 null values 
3. Number_of_reviews: strings with "," as character separator; 1024 null values corresponidng to rating
4. Bought_in_last_month：in the format of "XX+ bought in past month"; 3217 null values + some unmeaningful strings such as "Typical:", "List Price, " etc.
5. Current/discounted_price: contain "," as separator; in the format of "XX.XX"; 12941 null values 
6. Price_on_variant: in the format of "basic variant price: $XX.XX / XX Count(Pack of X) / XX GHz / nan / ..." ; 0 null values
7. Listed_price: contain "," as separator; in the format of "$XX.XX" or "No Discount" (listed_price = current/discounted_price in this case); 0 null values 
8. Is_best_seller: Binary variable: "No Badge" or "Best Seller"; 0 null value
9. Is_sponsored: Binary variable: "Sponsored" or "Organic"; 0 null value
10. Is_couponed: "Save XX% with coupon" or "No Coupon"l 0 null value
11. buy_box_availability: null means unavailabile; "Add to cart"; 14653 null values 
12. delivery_details: estimated delivery date as of the day collected the data; format example "Delivery Mon, Sep 1"; 11720 null values
13. sustainability_badges: several categories, need further inspection; 39267 null values 
14. image_url: format example "https://m.media-amazon.com/images/I/71pAqiVEs3L._AC_UL320_.jpg"; 0 null value
15. product_url:  2069 null values 
16. collected_at: format example "2025-08-21 11:14:29"; 0 null value 



## Clean the data 
### Coding Part
1. Clean "rating": 
    remove " out of 5 stars" in rating with str.replace(), and change the string data type into float using pd.to_numeric()
2. Clean "number_of_review":
    remove "," separator in number_of_reviews with str.replace()
3. Clean "current/discounted_price":
    fill the null current/discounted_price with price_on_variant
    1. replace cells without "$" to NA, and remove "basic variant price: $" part in price_on_variant
    2. apply cleaned price_on_variant to missing current_discounted_price
    3. remove all "," separator in current/discounted_price and convert to numeric
    4. check again the missing value in current/discounted_price after cleaning
4. Clean "listed_price":
    remove"$" in listed_price and replace "No Discount" with current/discounted_price

### Analysis Part
I chose four main columns to clean, mainly dealing with converting data type for future manipulation, and clean null values. There are still 2062 missing values in current/discounted_price and listed_price, which are aligned with each other after cleaning.


## Basic Filtering and Grouping
 1. keep only title, rating, number_of_reviews, current/discounted_price, listed_price columnsApply filters to extract meaningful subsets of the data.
 2. Use groupby() to group the rating and get the corresponding average number_of_review, current/discounted_price and listed_price.



## Explore a Machine Learning Algorithm
### Coding part
I chose to generate a Random Forest model and a XGBoost model to compare with each other in terms of their RSME and R^2

1. Feature Engineering: create a new colunm called "disctounted_rate" calculated by (listed_price-current/discounted_price)/listed_price, combine with other columns from the cleaned dataframe to be evaluated as the features: rating, number_of_reviews, listed_price, discount_rate. Leave current/discounted_price as target of study.
2. split the data set, and using only 20% on testing with train_test_split()
3. train the Random Forest model using RandomForestRegressor(), setting 100 decision trees and the seed to be 42
4. train the XGBoost model using XGBRegressor(), setting 100 decision trees with equal contribution to the result, and the seed to be 42.
evaluate the model by calculating RMSE(the lower the better) and R^2(the closer to 1 the better)

### Analysis part: Model Selection
the result of evaluation shows that the RandomForest regression is better than the XGBoost regression, given by a lower RMSE(15.2 comparing to 24.87), as well as a higher R^2(0.999 comparing to 0.9972). Therefore, RandomForest model is used for future visualization.


## Visualization (show in main.ipynb)
### Coding part
 1. use plot_feature_importance(model, model_name) to visualize how important each feature is in predicting the target variable respectively from random forest and xgboost
 2. using matplotlib plot a scatterplot comparing actual vs. predicted prices from the Random Forest model

 ### Analysis part
From the feature importance grapg, we can see that both Random Forest and XGBoost shows taht the listed price is the primary signal for predicting the final price, even though disocunt rate plays a small part of the role.There’s minimal contribution from customer-centric features like rating or number_of_reviews.

The Random Forest model demonstrates a good predictive performance, suggesting that the current feature set (mainly listed_price) captures most of the variability in the pricing.

## Test Cases
1. Data Loading: Ensures CSV loads correctly
2. Filtering: No null values after cleaning
3. Grouping: Ratings grouped properly
4. Preprocessing: Valid discount rate range
5. Model Training: Models train & achieve R² > 0
6. A/B Testing: Compares RandomForest vs XGBoost using RMSE
```
run by  pytest -v test_main.py
sample output: 
================================================ test session starts ================================================
platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0 -- /usr/local/py-utils/venvs/pytest/bin/python
cachedir: .pytest_cache
rootdir: /workspaces/IDS706_week2-3-assignment
plugins: cov-7.0.0
collected 6 items                                                                                                   
test_main.py::test_csv_loads_correctly PASSED                                                                 [ 16%]
test_main.py::test_no_missing_values_after_cleaning PASSED                                                    [ 33%]
test_main.py::test_grouping_by_rating PASSED                                                                  [ 50%]
test_main.py::test_discount_rate_calculation PASSED                                                           [ 66%]
test_main.py::test_models_train_and_predict PASSED                                                            [ 83%]
test_main.py::test_ab_testing_models PASSED                                                                   [100%]

================================================= 6 passed in 4.18s =================================================
```


## Dev and Docker Container
1. verify installation
```
docker --version
```

2. build image
```
docker build -t welcome-to-docker .
```
Once the build is complete, an image will appear in the Images tab. Select the image name to see its details. Select Run to run it as a container. In the Optional settings remember to specify a port number (something like 8089).

3. run your container
You can click "Run" in action column for image you just created from the Image list. If you don't have a name for your container, Docker provides one. View your container live by selecting the link below the container's name.

4. search Docker images and view the container
You can search images by selecting the bar at the top, or by using the ⌘ + K shortcut. Search for welcome-to-docker to find the image used in this guide.

Go to the Containers tab in Docker Desktop to view the container.

5. run the application
```
cd multi-container-app
docker compose up -d
```



## CI flow

```

name: IDS706_week2-3-assignment

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Lint with flake8
      run: flake8 main.py

    - name: Run tests
      run: pytest -v

```


### update changes to github

see what has been modified
```
git status
```

stage changes
```
git add xxx
```

commit changes
```
git commit -m"name"
```

push changes to git
```
git push
```
or 
```
git push original main --force
```


## Documentation
 Explain your steps and findings in this README.md file.(shown in the above)



