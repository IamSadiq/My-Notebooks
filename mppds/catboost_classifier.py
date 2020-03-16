import pandas as pd
import numpy as np

from catboost import Pool, CatBoostClassifier
# import sklearn.metrics as sklm

# 4th submission train features
X_train = np.array(pd.read_csv("scaled/x_train.csv"))
X_test = np.array(pd.read_csv("scaled/x_test.csv"))

y_train = np.array(pd.read_csv("scaled/y_train.csv"))
y_test = np.array(pd.read_csv("scaled/y_test.csv"))

y_train = y_train[:,1]
y_test = y_test[:,1]

train_dataset = Pool(X_train, y_train)
eval_dataset = Pool(X_test, y_test)

model = CatBoostClassifier(iterations=700, 
                        learning_rate=0.03,
                        custom_metric=['Logloss', 'AUC:hints=skip_train~false'])

model.fit(train_dataset, 
        eval_set=eval_dataset, 
        use_best_model=True, 
        verbose=False)

print(model.get_best_score())
print(model.get_params())

probabilities = model.predict(eval_dataset)

pd.DataFrame(probabilities).to_csv('predictions/pred1.csv')

# TEST VALUES
# test_values = pd.read_csv("test_values.csv")
# preped_test_values = np.array(pd.read_csv("scaled/actual_test_values.csv"))

preped_test_values = np.array(pd.read_csv('scaled/test_values_preped_1_feature_selection.csv'))

# eval_dataset = Pool(test_values)
test_prediction = model.predict(preped_test_values)

pd.DataFrame(test_prediction).to_csv('predictions/test_pred4.csv')