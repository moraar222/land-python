import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    :param file_path: str - Path to the CSV file.
    :return: DataFrame
    """
    return pd.read_csv(file_path)

# Preprocess the data
def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values and encode categorical variables.
    :param df: DataFrame - Raw dataset.
    :return: DataFrame - Processed dataset.
    """
    # Handling missing values (simple approach: drop them)
    df = df.dropna()

    # Encoding categorical features (e.g., location)
    df = pd.get_dummies(df, columns=['location'], drop_first=True)
    
    return df

# Train the model
def train_model(X_train, y_train):
    """
    Train a linear regression model.
    :param X_train: DataFrame - Features for training.
    :param y_train: Series - Target variable (house prices) for training.
    :return: Model - Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test dataset.
    :param model: Model - Trained model.
    :param X_test: DataFrame - Features for testing.
    :param y_test: Series - Actual house prices for testing.
    :return: float - Root mean squared error (RMSE) of the model predictions.
    """
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse

# Main function to run the script
def main():
    # File path to your dataset
    file_path = 'house_data.csv'
    
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Features (X) and target (y)
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    rmse = evaluate_model(model, X_test, y_test)
    print(f'Root Mean Squared Error: {rmse:.2f}')
    
    # Predict prices for new data
    # Example new data point: [bedrooms, bathrooms, square_footage, location_encoded_features]
    # new_data = [[3, 2, 1500, 1, 0, 0]]  # Adjust based on your dataset's encoding
    # predicted_price = model.predict(new_data)
    # print(f'Predicted House Price: ${predicted_price[0]:,.2f}')

if __name__ == "__main__":
    main()
