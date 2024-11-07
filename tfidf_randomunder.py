import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('train_edit_processed_nodupli.csv')
df.dropna(inplace=True)

# Encode the labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['sub_category'])

# Filter extremely rare classes
threshold = 5
class_counts = df['labels'].value_counts()
rare_classes = class_counts[class_counts < threshold].index
df = df[~df['labels'].isin(rare_classes)]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['crimeaditionalinfo'], df['labels'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Apply RandomOverSampler and RandomUnderSampler
#over_sampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)

# Oversample minority classes
#X_train_res, y_train_res = over_sampler.fit_resample(X_train_tfidf, y_train)

# Undersample majority classes
#X_train_res, y_train_res = under_sampler.fit_resample(X_train_res, y_train_res)
X_train_res, y_train_res = under_sampler.fit_resample(X_train_tfidf, y_train)

# Train Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_res, y_train_res)

# Predict
y_pred = nb.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(report)

# Save to file
with open('TFIDF_NaiveBayes_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(report)
