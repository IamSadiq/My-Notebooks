from keras import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np

# import sklearn.metrics as sklm

X_train = np.array(pd.read_csv("scaled/x_train.csv"))
X_test = np.array(pd.read_csv("scaled/x_test.csv"))

y_train = np.array(pd.read_csv("scaled/y_train.csv"))
y_test = np.array(pd.read_csv("scaled/y_test.csv"))

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

probabilities=classifier.predict(X_test)
# y_pred =(y_pred>0.5)

def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

# def print_metrics(labels, scores):
#     metrics = sklm.precision_recall_fscore_support(labels, scores)
#     conf = sklm.confusion_matrix(labels, scores)
#     print('                 Confusion matrix')
#     print('                 Score positive    Score negative')
#     print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
#     print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
#     print('')
#     print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
#     print(' ')
#     print('           Positive      Negative')
#     print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
#     print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
#     print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
#     print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


# print_metrics(y_test, scores)