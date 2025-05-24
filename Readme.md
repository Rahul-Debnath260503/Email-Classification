# Spam Email Classification Project

This project focuses on building and deploying a machine learning model to classify emails as either "spam" or "not spam" (ham).

## Project Steps

The project follows these main steps, covering the process from data loading to deployment:

1.  **Importing Libraries:** Essential libraries like pandas, numpy, and matplotlib were imported for data handling, numerical operations, and visualization.
2.  **Loading the Dataset:** The `spam.csv` dataset was loaded into a pandas DataFrame for analysis.
3.  **Initial Data Exploration:** The head of the DataFrame was viewed to get an initial understanding of the data structure and content.
4.  **Category Distribution Visualization:** A horizontal bar plot was generated using matplotlib and seaborn to visualize the distribution of 'ham' and 'spam' categories in the dataset.
5.  **Counting Data Entries:** The `.count()` method was used to check the number of non-null entries in each column.
6.  **Categorical to Numerical Conversion:** The 'Category' column was mapped to numerical values (0 for 'ham' and 1 for 'spam') for model training.
7.  **Text Lowercasing:** The 'Message' column was converted to lowercase to ensure consistency in text processing.
8.  **Text Cleaning with NLTK:**
    *   NLTK stopwords were downloaded.
    *   A `clean_text` function was defined to remove special characters, extra spaces, and stopwords from the email messages.
    *   The `clean_text` function was applied to the 'Message' column to create a new 'Cleaned_Message' column.
    *   A basic list of common English stopwords was also defined.
9.  **TF-IDF Feature Extraction:**
    *   `TfidfVectorizer` from scikit-learn was used to convert the cleaned text data into TF-IDF features.
    *   The resulting feature matrix (`X`) and target labels (`y`) were checked for their shapes.
10. **Data Splitting and Balancing:**
    *   The data was split into training (80%) and testing (20%) sets using `train_test_split` with stratification based on the target variable.
    *   SMOTE (Synthetic Minority Over-sampling Technique) from imblearn was applied to the training data to handle class imbalance by oversampling the minority class ('spam').
11. **Model Training:**
    *   Multinomial Naive Bayes, Logistic Regression, and Decision Tree models were trained on the SMOTE-balanced training data.
12. **Model Evaluation:**
    *   Predictions were made on the test data using each of the trained models.
    *   Accuracy scores and classification reports were printed for each model to evaluate their performance.
13. **Model Accuracy Comparison Visualization:**
    *   A bar plot was created using matplotlib to visually compare the accuracy scores of the three trained models.
14. **Ensemble Modeling (Voting Classifier):**
    *   A Voting Classifier was created to combine the predictions of the Naive Bayes, Logistic Regression, and Decision Tree models using a 'hard' voting strategy.
    *   The ensemble model was trained on the SMOTE-balanced training data and evaluated on the test data. Accuracy and a classification report were printed.
15. **Model and Vectorizer Saving:**
    *   The trained Voting Classifier model and the TF-IDF vectorizer were saved using pickle for later use in the Streamlit application.
16. **Streamlit Application (`app.py`):**
    *   A Streamlit script (`app.py`) was written to create a simple web application.
    *   This script loads the saved model and vectorizer.
    *   It includes a function to preprocess user input text similar to the training data.
    *   The app provides a text area for users to enter email text and a button to get a spam/ham prediction.
17. **Requirements File:**
    *   A `requirements.txt` file was created to list all the necessary Python libraries for running the project.
18. **Readme File (This File):**
    *   A `Readme.md` file was written to provide an overview of the project, the steps involved, and setup instructions.

## Setup and Installation

1.  **Open in Google Colab:** This notebook is designed to be run in Google Colab.
2.  **Install Libraries:** Ensure you have the necessary libraries installed. The `requirements.txt` file lists all required libraries. You can install them using pip in your Colab notebook:

## How to Run

1.  Run all the code cells in the Colab notebook sequentially from top to bottom. This will load and preprocess data, train the models, and save the necessary files.
2.  Once the `app.py` file is created in your Colab environment (using `%%writefile app.py`), you can run the Streamlit application directly from Colab.
