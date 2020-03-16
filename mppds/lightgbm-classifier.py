# !pip install lightgbm
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (roc_curve, auc, accuracy_score)
import sklearn.metrics as sklm

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq': 1,
    'is_training_metric': True,
    'max_bin': 255,
    'learning_rate': 0.1,
    'num_leaves': 100,
    'num_iterations': 1000,
    'tree_learner': 'serial',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 5,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_save_binary_file': False,
    'output_model': 'LightGBM_model.txt',
    'num_machines': 1,
    'local_listen_port': 12400,
    'machine_list_file': 'mlist.txt',
    'verbose': 0,
    'subsample_for_bin': 200000,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'max_depth': 9
}

df_train = pd.read_csv("cleaned_train_data.csv")

labels = pd.read_csv('train_labels.csv')
df_train['accepted'] = labels['accepted']

# # feature engineering
mean_encoding = df_train.groupby(['lender']).agg({'accepted':['mean']}).reset_index()
df_train = df_train.merge(mean_encoding, on='lender', how='left')

last_col = df_train.columns[df_train.shape[1]-1]

cols = ['loan_type', 'property_type', 'loan_purpose', 'occupancy', 'loan_amount', 'preapproval', 
                'msa_md', 'state_code', 'county_code', 'applicant_income', 'applicant_ethnicity', 'applicant_sex',
        'ffiecmedian_family_income', 'tract_to_msa_md_income_pct', 'number_of_owner-occupied_units', 
        'number_of_1_to_4_family_units', 'lender', last_col]

X = df_train[cols]
Y = df_train['accepted']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1000)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# train
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval)


# gridParams = {
#     'learning_rate': [ 0.1],
#     'num_leaves': [63],
#     'boosting_type' : ['gbdt'],
#     'objective' : ['binary']
# }

# mdl = lgb.LGBMClassifier(
#     task = params['task'],
#     metric = params['metric'],
#     metric_freq = params['metric_freq'],
#     is_training_metric = params['is_training_metric'],
#     max_bin = params['max_bin'],
#     tree_learner = params['tree_learner'],
#     feature_fraction = params['feature_fraction'],
#     bagging_fraction = params['bagging_fraction'],
#     bagging_freq = params['bagging_freq'],
#     min_data_in_leaf = params['min_data_in_leaf'],
#     min_sum_hessian_in_leaf = params['min_sum_hessian_in_leaf'],
#     is_enable_sparse = params['is_enable_sparse'],
#     use_two_round_loading = params['use_two_round_loading'],
#     is_save_binary_file = params['is_save_binary_file'],
#     n_jobs = -1
# )

# scoring = {'AUC': 'roc_auc'}

# # Create the grid
# grid = GridSearchCV(mdl, gridParams, verbose=2, cv=5, scoring=scoring, n_jobs=-1, refit='AUC')

# # Run the grid
# grid.fit(X_train, y_train)

# print('Best parameters found by grid search are:', grid.best_params_)
# print('Best score found by grid search is:', grid.best_score_)


# making validation prediciton for one column
val_pred = gbm.predict(X_test)


preds = []
for i in val_pred:
  if i >= 0.5:
    preds.append(1)
  else:
    preds.append(0)
    
print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, preds)
))
print('')


def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


    
print_metrics(y_test, preds)    


def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs)
    auc = sklm.auc(fpr, tpr)
    
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
# probabilities = gbm.predict_proba(X_test)
plot_auc(y_test, val_pred)