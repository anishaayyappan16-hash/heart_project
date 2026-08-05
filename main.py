import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('heart.csv') # Load Data
print("Original Data Loaded:", df.shape)

df = df.drop_duplicates() # Data Cleaning
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])
print("After Cleaning:",df.shape)
# pre-processing
for c in df.select_dtypes('object'): # Convert String → Numbers
    df[c] = LabelEncoder().fit_transform(df[c])
print("After Encoding:",df.head())

corr = df.corr()['target'].abs() # Feature Selection (keep only useful features)
features = corr[corr>0.1].index.drop('target')
print("co-relations:\n",corr,"\nSelected Features:",list(features))

x = df[features] # Split Data for training
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear') #  Train & Evaluate
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

joblib.dump(model, 'heart_model.pkl')  # Save the model as .pkl automatically
print("heart_model.pkl file created successfully!")
