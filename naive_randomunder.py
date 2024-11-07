import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load the dataset
df = pd.read_csv('train_edit_processed_nodupli.csv')
df.dropna(inplace=True)


df_test = pd.read_csv('test_edit_processed.csv')
df_test.dropna(inplace=True)

df_combined = pd.concat([df, df_test], ignore_index=True)



# Encode the labels
label_encoder = LabelEncoder()
df_combined['labels'] = label_encoder.fit_transform(df_combined['sub_category'])

df_train_encode = df_combined[:len(df)]
df_test_encode = df_combined[len(df):]

# Filter extremely rare classes
threshold = 5
class_counts = df_train_encode['labels'].value_counts()
rare_classes = class_counts[class_counts < threshold].index
df_train_encode = df_train_encode[~df_train_encode['labels'].isin(rare_classes)]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df_train_encode['crimeaditionalinfo'], df_train_encode['labels'], test_size=0.2, random_state=42)

# Vectorize text using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Apply RandomOverSampler and RandomUnderSampler
over_sampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)

# Oversample minority classes
X_train_res, y_train_res = over_sampler.fit_resample(X_train_counts, y_train)

# Undersample majority classes
X_train_res, y_train_res = under_sampler.fit_resample(X_train_res, y_train_res)

# Train Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_res, y_train_res)

# Predict
y_pred = nb.predict(X_test_counts)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(report)

# Save to file
with open('CountVectorizer_NaiveBayes_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(report)

# Predict test dataset
x_testdataset = df_test_encode['crimeaditionalinfo']
y_testdataset = df_test_encode['labels']

x_testdataset_vect = count_vectorizer.transform(x_testdataset)

y_pred_test = nb.predict(x_testdataset_vect)

accuracy_test = accuracy_score(y_pred_test, y_testdataset)
testReport = classification_report(y_pred_test, y_testdataset)

f1_test = f1_score(y_pred=y_pred_test, y_true=y_testdataset, average='weighted')

# Print results
print(f"Accuracy: {accuracy_test}")
print(testReport)
print(f1_test)