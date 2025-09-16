import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])

# Split into features and labels
X = df["message"]
y = df["label"]

# Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Loop for user input
while True:
    msg = input("\nEnter a message (or type 'exit' to quit): ")
    if msg.lower() == "exit":
        print("Goodbye! 👋")
        break

    prediction = model.predict(cv.transform([msg]))[0]
    print(f"Message: {msg}")
    print(f"Prediction: {'✅ Ham (Not Spam)' if prediction == 'ham' else '🚨 Spam'}")
