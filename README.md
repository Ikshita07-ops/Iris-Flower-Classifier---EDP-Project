# Iris Flower Classification

## Week 1 - Data Handling and Preprocessing

* Loaded the Iris dataset using Pandas.
* Removed the unnecessary `Id` column.
* Checked for missing values.
* Removed duplicate records.
* Checked the different flower species and their counts.
* Selected the flower measurements as features.
* Selected the flower species as the target label.

## Week 2 - Analysis, Visualization and Classification

* Visualized the feature distributions using Matplotlib histograms.
* Created a scatter plot to compare petal length and petal width.
* Split the dataset into training and testing data.
* Built a Logistic Regression classification model.
* Trained the model using the training data.
* Predicted flower species using the test data.
* Evaluated the model using:

  * Accuracy
  * Confusion Matrix
  * Classification Report
* Tested the model with a new flower sample.

## Week 3 - Text Processing and Text Classification

* Learned the basics of **Text Preprocessing** and Natural Language Processing (NLP).
* Learned how to clean text by:

  * Converting text to lowercase
  * Removing URLs and email addresses
  * Handling numbers and punctuation
  * Removing unnecessary spaces
* Learned how to convert text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* Learned the basics of **Naive Bayes Classification**.
* Built an **SMS Spam Classifier** using TF-IDF and Multinomial Naive Bayes.
* Split the SMS dataset into training and testing data.
* Trained the Naive Bayes model using the processed text data.
* Tested the trained model with new SMS messages and classified them as **HAM or SPAM**.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Natural Language Processing (NLP)

## Datasets

### Iris Flower Dataset

* **Name:** Iris Flower Dataset
* **Source:** Kaggle
* **Samples:** 150
* **Features:**

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* **Target:**

  * Iris-setosa
  * Iris-versicolor
  * Iris-virginica

### SMS Spam Dataset

* **Purpose:** SMS Spam Classification
* **Features:** SMS message text
* **Target:**

  * HAM
  * SPAM
