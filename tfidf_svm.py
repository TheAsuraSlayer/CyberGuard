import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('train_edit_processed.csv')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['crimeaditionalinfo'], df['sub_category'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train_tfidf, y_train)

# Predict
y_pred = svm.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(report)

# Save to file
with open('SVM_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(report)
