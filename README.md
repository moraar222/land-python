

# House Price Prediction

This project provides a simple Python script to predict house prices based on various features such as the number of bedrooms, bathrooms, square footage, and location. The model is built using a linear regression algorithm from the `scikit-learn` library.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

House price prediction is a common task in the real estate industry. By analyzing past data, we can build models to predict the price of a house based on its features. This project aims to create a simple linear regression model that can be trained on a dataset of houses and then used to predict the prices of new houses.

## Features

- **Data Preprocessing**: Handles missing values and encodes categorical variables.
- **Model Training**: Trains a linear regression model on the provided dataset.
- **Model Evaluation**: Evaluates the model using the Root Mean Squared Error (RMSE) metric.
- **Prediction**: Provides an example to predict the price of a house based on input features.

## Installation

### Requirements

- Python 3.6+
- `pandas` library
- `scikit-learn` library

You can install the required Python libraries using pip:

```bash
pip install pandas scikit-learn
```

### Files

- `house_price_prediction.py`: The main script for training the model and making predictions.
- `README.md`: Project documentation (this file).
- `house_data.csv`: Example dataset (you need to provide your own dataset).

## Usage

1. **Prepare your dataset**: Ensure you have a CSV file named `house_data.csv` in the same directory as the script. The CSV file should include columns like `bedrooms`, `bathrooms`, `square_footage`, `location`, and `price`.

2. **Run the script**: Execute the Python script to train the model and evaluate its performance.

```bash
python house_price_prediction.py
```

3. **Evaluate the Model**: The script will output the Root Mean Squared Error (RMSE) of the model's predictions on the test data.

4. **Predict House Prices**: You can modify the script to predict the price of a new house by providing its features in the `new_data` section.

## Dataset

The dataset should be in CSV format with the following columns:

- `bedrooms`: Number of bedrooms in the house.
- `bathrooms`: Number of bathrooms in the house.
- `square_footage`: Total square footage of the house.
- `location`: Location of the house (categorical variable).
- `price`: Price of the house (target variable).

Example:

| bedrooms | bathrooms | square_footage | location   | price     |
|----------|-----------|----------------|------------|-----------|
| 3        | 2         | 1500           | New York   | 500000    |
| 4        | 3         | 2000           | San Francisco | 1000000  |

## Model Training

The model is trained using the `LinearRegression` class from the `scikit-learn` library. The data is split into training and testing sets using the `train_test_split` function. The model learns the relationship between the features and the target variable (price) during training.

## Evaluation

The model's performance is evaluated using the Root Mean Squared Error (RMSE) metric, which gives an indication of how well the model's predictions match the actual prices.

## Prediction

To predict the price of a new house, you can modify the `new_data` variable in the script with the features of the house you want to predict.

Example:

```python
new_data = [[3, 2, 1500, 1, 0, 0]]  # Adjust based on your dataset's encoding
predicted_price = model.predict(new_data)
print(f'Predicted House Price: ${predicted_price[0]:,.2f}')
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

