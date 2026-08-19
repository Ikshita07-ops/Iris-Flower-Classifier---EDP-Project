# Iris Flower Classification

A Machine Learning project that focuses on data preprocessing, visualization, classification, and text classification concepts using Python and Scikit-learn.

## Week 1 - Data Handling and Preprocessing

* Loaded the Iris dataset using Pandas.
* Removed the unnecessary `Id` column.
* Checked for missing values.
* Removed duplicate records.
* Checked the different flower species and their counts.
* Selected flower measurements as features.
* Selected flower species as the target label.

## Week 2 - Analysis, Visualization and Classification

* Visualized feature distributions using Matplotlib histograms.
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

* Learned the basics of Text Preprocessing and Natural Language Processing (NLP).
* Learned text cleaning techniques:

  * Converting text to lowercase
  * Removing URLs and email addresses
  * Handling numbers and punctuation
  * Removing unnecessary spaces
* Learned how to convert text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* Learned the basics of **Naive Bayes Classification**.
* Built an **SMS Spam Classifier** using TF-IDF and Multinomial Naive Bayes.
* Split the SMS dataset into training and testing data.
* Trained the Naive Bayes model using processed text data.
* Tested new SMS messages and classified them as **HAM or SPAM**.

## Week 4 - Data Visualization and Feature Relationships

* Learned the basics of **Data Visualization** for Machine Learning.
* Created a **Pair Plot** to visualize relationships between all Iris features.
* Created a **Correlation Heatmap** to identify relationships between numerical features.
* Created **Box Plots** to compare feature distributions across Iris species.
* Created scatter plots to compare different feature pairs.
* Compared Sepal Length vs. Sepal Width and Petal Length vs. Petal Width.
* Analyzed the visualizations to identify useful features for classification.
* Observed that **Petal Length and Petal Width provide better separation between Iris species**.
* Observed that **Iris-setosa is clearly separated**, while Iris-versicolor and Iris-virginica show some overlap.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
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

