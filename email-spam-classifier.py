from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re

print ("="*10+"The first 5 rows of the dataframe"+"="*10)
df = pd.read_csv('emails.csv')
print (df.head()) # Print the first 5 rows of the dataframe
print("\n")

print ("="*10+"The shape of the dataframe"+"="*10)
print (df.shape) # Print the shape of the dataframe
print("\n")

print ("="*10+"The columns of the dataframe"+"="*10)
print (df.columns) # Print the columns of the dataframe
print("\n")

print ("="*10+"The spam vs non-spam emails count"+"="*10)
print (df['Prediction'].value_counts()) # Print how many spam vs non-spam emails exist
print("\n")

print ("="*10 +"Reconstructed Text"+"="*10)
word_columns = [col for col in df.columns if col not in ['Email No.', 'Prediction']] # Create a list of all column names except the metadata/target columns.
df['text'] = df[word_columns].apply(lambda row: ' '.join(word for col in word_columns for word in [col] * row[col]), axis=1) # Reconstruct the original email text from the binary word presence columns.
df['clean_text'] = df['text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True) # Clean the reconstructed text by lowercasing and removing non-alphabetic characters.
print(df['clean_text'].head())
print ('\n')

print ("="*10 +"X_train, X_test, y_train, y_test shapes"+"="*10)
X_train_text, X_test_text, y_train, y_test = train_test_split(df['clean_text'], df['Prediction'], test_size=0.2, random_state=42)
print (X_train_text.shape, X_test_text.shape, y_train.shape, y_test.shape) # Print the shapes of all 4 variables to verify the split
print("\n")

print ("="*10 +"TF-IDF Vectorization"+"="*10)
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english') # Create a TfidfVectorizer instance with 5000 features and English stop words.
X_train_tfidf = vectorizer.fit_transform(X_train_text) # Fit the vectorizer on the training text and transform it into a TF-IDF matrix.
X_test_tfidf = vectorizer.transform(X_test_text) # Transform the test text into a TF-IDF matrix using the fitted vectorizer.
print(X_train_tfidf.shape, X_test_tfidf.shape) # Print the shapes of the TF-IDF matrices to verify the transformation.
print ('\n')

print ("="*10 +"X and y shapes"+"="*10)
X = df.drop(columns=['Email No.', 'Prediction']) # Drop the non-useful column and separate features from the label
y = df["Prediction"]
print (X.shape, y.shape) # Print the shapes of X and y
print("\n")

print ("="*10 +"Model score"+"="*10)
model = LogisticRegression(max_iter=1000) # Create a LogisticRegression model instance
model.fit (X_train_tfidf, y_train) # Train it on X_train and y_train
y_pred = model.predict(X_test_tfidf) # Generate predictions on X_test, and store in y_pred
print (model.score(X_test_tfidf, y_test))
print("\n")

print ("="*10 +"Classification Report"+"="*10)
print (classification_report(y_test, y_pred))
print ('\n')

print ("="*10 +"Confusion Matrix"+"="*10)
print (confusion_matrix(y_test, y_pred))
print ('\n')
