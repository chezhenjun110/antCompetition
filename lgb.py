import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")



def evaluate(y_true,y_proba,pos_label):
    fpr,tpr,thresholds=roc_curve(y_true,y_proba[:,1],pos_label=1)
    return 0.4*tpr[np.where(fpr>=0.001)[0][0]]+\
        0.3*tpr[np.where(fpr>=0.005)[0][0]]+\
        0.3*tpr[np.where(fpr>=0.01)[0][0]]


def get_cate_features(data):
    cat_col=[]
    cols = [c for c in data.columns if 'f' in c]
    for col in cols:
        if train[col].dtype =='int64':
            cat_col.append(col)
    return cat_col
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)

y=train["label"].values
X=train.drop(["label","date"],axis=1,inplace=True)
train_x,train_y,valid_x,valid_y=train_test_split(X,y,test_size=0.2,random_state=2018)
categorical_feature=get_cate_features(train)
test=test.drop(["label","date"],axis=1,inplace=True)
features=train.columns.tolist()

clf=lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=1000,
    max_bin=255,
    subsample_for_bin=200000,
    objective=None,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=1,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None, n_jobs=-1,
    silent=True
)
clf.fit(train_x, train_y,
        init_score=None, eval_set=([valid_x,valid_y]), eval_metric='auc',
        early_stopping_rounds=100, verbose=True, feature_name='auto',
        categorical_feature=categorical_feature, callbacks=None
        )
proba=clf.predict_proba(X)
submission=pd.read_csv("./sub_sample/submission.csv")
submission["score"]=proba
submission.to_csv("submission.csv",index=False)
