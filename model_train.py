import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

# Load dataset
df = pd.read_csv("tweet_sentiment.csv", encoding="latin-1", header=None)

# Assign correct column names for Sentiment140 dataset
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Keep only needed columns
df = df[["text", "sentiment"]].copy()

# Remove missing values
df.dropna(inplace=True)

# Keep rows that contain English letters
df = df[df["text"].astype(str).str.contains(r"[a-zA-Z]", na=False)]

# Convert labels
df["sentiment"] = df["sentiment"].replace({
    0: "negative",
    4: "positive"
})

# Clean text
df["cleaned_text"] = df["text"].apply(clean_text)

# Remove empty cleaned rows
df = df[df["cleaned_text"].str.strip() != ""]

# Features and labels
X = df["cleaned_text"]
y = df["sentiment"]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")