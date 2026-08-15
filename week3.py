import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Loading dataset
df = pd.read_csv(
    "spam_dataset.csv",
    encoding="latin-1"
)

# Keep only required columns
df = df[["v1", "v2"]]

df.columns = ["label", "message"]

# Remove empty rows
df.dropna(inplace=True)

# Check for duplicates
df.drop_duplicates(inplace=True)


print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


#Text Preprocessing 

def preprocess_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Replacing URL with word 
    text = re.sub(
        r"http\S+|www\S+",
        " url ",
        text
    )

    # Replacing email with word
    text = re.sub(
        r"\S+@\S+",
        " email ",
        text
    )

    # Replacing numbers with a word
    text = re.sub(
        r"\d+",
        " number ",
        text
    )

    # Removing punctuations
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Removing extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

# Apply preprocessing

df["clean_message"] = df["message"].apply(
    preprocess_text
)

# Converting labels
# ham = 0
# spam = 1

df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


#features and labels
X = df["clean_message"]
y = df["label_num"]



# Train-test split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

#TF-IDF Vectorization
vectorizer = TfidfVectorizer(

    # Single words + two-word combinations
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)


# Learning vocabulary from training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transforming test data 
X_test_tfidf = vectorizer.transform(
    X_test
)
print("\nTF-IDF shape:")
print(X_train_tfidf.shape)

# NAIVE BAYES 
model = MultinomialNB(
    alpha=0.1
)

# Train
model.fit(
    X_train_tfidf,
    y_train
)

#Model Prediction
y_pred = model.predict(
    X_test_tfidf
)

#Evaluation metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("MODEL EVALUATION")

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(classification_report(
        y_test,
        y_pred,
        target_names=[
            "HAM",
            "SPAM"
        ]
    )
)

#Testing with new message
def predict_sms(message):

    # Preprocess
    cleaned_message = preprocess_text(
        message
    )

    # TF-IDF
    message_tfidf = vectorizer.transform(
        [cleaned_message]
    )

    # Predict
    prediction = model.predict(
        message_tfidf
    )[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    return result


print("SMS SPAM CLASSIFIER")
print("\nType an SMS to classify.")
print("Type 'exit' to stop.")


while True:

    message = input("\nEnter SMS: ")

    if message.lower() == "exit":
        print("Program stopped.")
        break

    result = (predict_sms(message))

    print(
        "Prediction:",
        result
    )
