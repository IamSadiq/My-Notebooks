import pandas as pd
import numpy as np

from catboost import Pool, CatBoostClassifier

# New CatBoost submission
X_train = np.array(pd.read_csv("preped/x_train_new_catboost-10.csv"))
X_test = np.array(pd.read_csv("preped/x_test_new_catboost-10.csv"))

y_train = np.array(pd.read_csv("preped/y_train_new_catboost-10.csv"))
y_test = np.array(pd.read_csv("preped/y_test_new_catboost-10.csv"))

# X_train = np.array(pd.read_csv("scaled/x_train.csv"))
# X_test = np.array(pd.read_csv("scaled/x_test.csv"))

# y_train = np.array(pd.read_csv("scaled/y_train.csv"))
# y_test = np.array(pd.read_csv("scaled/y_test.csv"))


y_train = y_train[:,1]
y_test = y_test[:,1]

train_pool = Pool(X_train, y_train)
eval_pool = Pool(X_test, y_test)

model = CatBoostClassifier(iterations=700, 
                        learning_rate=0.03,
                        custom_metric=['Logloss', 'AUC:hints=skip_train~false'])

# model = CatBoostClassifier(iterations=7000,
#                         depth=8, 
#                         learning_rate=0.02,
#                         loss_function='Logloss',
#                         eval_metric='AUC',
#                         random_seed=42,
#                         rsm=0.2,
#                         od_type='Iter',
#                         od_wait=100,
#                         verbose=100,
#                         l2_leaf_reg=20)

model.fit(train_pool, 
        eval_set=eval_pool, 
        use_best_model=True, 
        verbose=True)

# Feature Importance: Know which feature contributed the most
feature_importances = model.get_feature_importance(train_pool)
feature_names = pd.DataFrame(X_train).columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

print('\n\n\n')
print(model.get_best_score())
print(model.get_params())

# save model
model.save_model('models/new_catboost_classifier_model-10.dump')

# Validation Prediction
probabilities = model.predict(eval_pool)
pd.DataFrame(probabilities).to_csv('validation-scores/val-scores-new-catboost-10.csv')

# TEST VALUES
preped_test_values = np.array(pd.read_csv('preped/test_new_catboost-10.csv'))
eval_test_pool = Pool(preped_test_values)
test_prediction = model.predict(eval_test_pool)

# save predictions for test_values
pd.DataFrame(test_prediction).to_csv('predictions/test_new_catbost_pred-10.csv', index=None)