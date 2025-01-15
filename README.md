# Spam Email Classifier

This project demonstrates how to build a spam email classifier using machine learning, deploy it as a FastAPI endpoint, and create a frontend interface using Tailwind CSS.

## Table of Contents
1. [Setup Environment](#install-repository)
1. [Data Preprocessing](#data-preprocessing)
2. [Model Training](#model-training)
3. [API Deployment](#api-deployment)
4. [Frontend Integration](#frontend-integration)


## Setup Environment

1. **Install Dependencies**: Install the required dependencies using `pip`:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing

1. **Install the dataset from the UCI Machine Learning Repository**: The dataset used in this project is the SMS Spam Collection Dataset from the UCI Machine Learning Repository. You can download the dataset from the following link: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Then put it in the same directory as the notebook under the data folder.

2. **Load the Dataset**: Load the dataset from a file named `SMSSpamCollection`. This dataset contains SMS messages labeled as spam or ham (not spam).

3. **Check for Missing Values**: Ensure there are no missing values in the dataset to avoid errors during model training.

4. **Label Distribution**: Check the distribution of spam and ham labels to understand the dataset's balance.

5. **Text Vectorization**: Convert the text data into numerical data using `CountVectorizer`. This step transforms the text into a format suitable for machine learning algorithms.

6. **Scaling and PCA**: Scale the data using `StandardScaler` and apply PCA to reduce dimensionality. This step helps in improving the model's performance and reducing overfitting.

## Model Training

1. **Train-Test Split**: Split the data into training and testing sets to evaluate the model's performance on unseen data.

2. **Train the Model**: Train a `RandomForestClassifier` on the training data. This classifier will learn to distinguish between spam and ham messages.

3. **Evaluate the Model**: Evaluate the model's performance on the test data using metrics like accuracy and classification report.

4. **Save the Model and Vectorizer**: Save the trained model and vectorizer using `joblib` for later use in the API.


## API Deployment

1. **Create FastAPI Application**: Create a FastAPI application and load the saved model and vectorizer. Add CORS middleware to allow cross-origin requests.

2. **Run the API**: Run the FastAPI server using the command:

```bash
python app.py
```

## Frontend Integration

1. **Create HTML File**: Create an HTML file with Tailwind CSS and a spam checker interface. This file includes a text area for entering email text, a button for checking spam, and a gauge to display the spam probability.


2. **Run the Frontend**: Open the `index.html` file in a web browser or serve it using a simple HTTP server.

```bash
python -m http.server
```

3. **Test the Application**: Enter some text in the textarea and click the "Check for Spam" button to see the result.

## Conclusion

This project demonstrates the complete workflow of building a spam email classifier, deploying it as an API, and integrating it with a frontend interface. You can further enhance the model and the frontend as needed.

## Frontend Interface

![Frontend Image1](https://github.com/user-attachments/assets/60657855-352c-4510-ba00-1db3e0af8ff1)
![Frontend Image2](https://github.com/user-attachments/assets/107a1a55-d9ea-46b6-b2e2-1ee1db30e6f6)

## References
Used dataset: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)