import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

from catboost import Pool, CatBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.layers import Dense

train_data = pd.read_csv('preped_data2.csv')
labels = np.array(train_data['accepted'])

print(train_data.shape)

cols = ['loan_type', 'property_type', 'loan_purpose', 'occupancy', 'loan_amount', 'preapproval', 
                'msa_md', 'state_code', 'county_code', 'applicant_income',
                'tract_to_msa_md_income_pct', 'number_of_owner-occupied_units', 'lender']  

features = np.array(train_data[cols])

sc = StandardScaler()
X = sc.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, y_train)
print(eval_model)

y_pred=classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)