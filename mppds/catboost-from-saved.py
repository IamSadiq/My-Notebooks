import pandas as pd
import numpy as np

from catboost import Pool, CatBoostClassifier

X_train = np.array(pd.read_csv("preped/x_train_3.csv"))
X_test = np.array(pd.read_csv("preped/y_train_3.csv"))

y_train = np.array(pd.read_csv("preped/x_val_test_3.csv"))
y_test = np.array(pd.read_csv("preped/y_val_test_3.csv"))


y_train = y_train[:,1]
y_test = y_test[:,1]

train_pool = Pool(X_train, y_train)
eval_pool = Pool(X_test, y_test)


# load model
model = CatBoostClassifier()
model.load_model('models/catboost_model_4.dump')


# Feature Importance: Know which feature contributed the most
feature_importances = model.get_feature_importance(train_pool)
feature_names = pd.DataFrame(X_train).columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

print('\n\n\n')
print(model.get_best_score())
print(model.get_params())


# Validation Prediction
probabilities = model.predict(eval_pool)
# print(probabilities)
pd.DataFrame(probabilities).to_csv('validation-scores/val-scores-3.csv')


# TEST VALUES

# preped_test_values = np.array(pd.read_csv('preped/preped_test_&_featured.csv'))

# eval_dataset = Pool(test_values)
# test_prediction = model.predict(preped_test_values)

# pd.DataFrame(test_prediction).to_csv('predictions/10th-submission-catboost.csv')