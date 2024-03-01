import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#  load data
data = pd.read_csv('creditcard.csv')

# seperate legitimate and fraudulent transactions
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

legit_new = legit.sample(n=len(fraud), random_state=2)
new_df = pd.concat([legit_new, fraud], axis=0)

# spliting the data in  x and y
X = new_df.drop('Class', axis=1)
Y = new_df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# checking model performance
train_data_acc = accuracy_score(model.predict(X_train), Y_train)
test_data_acc = accuracy_score(model.predict(X_test), Y_test)

# web app
st.title("Credit Card Fraud Detection Model")
input_data = st.text_input("Enter all required features values")
input_splited = input_data.split(',')
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_splited, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")


