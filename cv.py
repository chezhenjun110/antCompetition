import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,roc_curve
import lightgbm as lgb
import gc
import feature_engineering as fe

def score(y_true, y_score):
    """ Evaluation metric
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    score = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.01)[0][0]]

    return score


def evaluate(y_true, y_pred, y_prob):
    """ 估计结果: precision, recall, f1, auc, mayi_score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mayi = score(y_true, y_prob)
    
    return [p,r,f1,auc,mayi] 

def cv(clf_fit_params,clf,X,y,n_splits=5):
    models=[]
    i=0
    eval_train = pd.DataFrame(index=range(n_splits), columns=['P','R','F1','AUC','mayi'])
    eval_test  = pd.DataFrame(index=range(n_splits), columns=['P','R','F1','AUC','mayi'])
    kf=KFold(n_splits=5,shuffle=True,random_state=2018)
    for train_index,test_index in kf.split(X):
        X_,X_test=X[train_index],X[test_index]
        y_,y_test=y[train_index],y[test_index]
        X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=2018)
        del X_,y_
        gc.collect()
        
        
        clf.fit(X_train, y_train,eval_set=[(X_valid,y_valid)], eval_metric='auc',
        early_stopping_rounds=100, verbose=True, **clf_fit_params)
        
        ## Model Testing
        # On training set
        y_prob_train = clf.predict_proba(X_train)[:,1]
        y_pred_train = clf.predict(X_train)
        eval_train.iloc[i,:] = evaluate(y_train, y_pred_train, y_prob_train)
        
        # On testing set
        y_prob_test = clf.predict_proba(X_test)[:,1]
        y_pred_test = clf.predict(X_test)
        eval_test.iloc[i,:] = evaluate(y_test, y_pred_test, y_prob_test)
        models.append(clf)
        
        i+=1
        
    return models,eval_train,eval_test

def dataprocess(data):
    data=fe.transfer_date2_week(data)
    data["f22_f21"]=data["f22"]-data["f21"]
    data["f24_f23"]=data["f24"]-data["f23"]
    data["f163_f162"]=data["f163"]-data["f162"]
    data["f164_f163"]=data["f164"]-data["f163"]
    data["f164_f162"]=data["f164"]-data["f162"]
    data.fillna(-999,inplace=True)
    return data


def loaddata():

    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")
    return train,test

if __name__=='__main__':
    train,test=loaddata()
    train=dataprocess(train)
    test=dataprocess(test)

    y=train["label"].values
    train.drop(["label","date"],axis=1,inplace=True)
    test.drop(["label","date"],axis=1,inplace=True)

    categorical_features=[]
    for item in train.columns:
        if train[item].nunique()<=10:
            categorical_features.append(item)
    features=train.columns.tolist()

    
    X=train[features].values

    clf=lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=1000,
    max_bin=255,
    subsample_for_bin=200000,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=1,
    colsample_bytree=1.0,
    random_state=None, n_jobs=-1,
    silent=True
    )


    clf_fit_params = {'feature_name': features,
                          'categorical_feature': categorical_features}
    models,eval_train,eval_test=cv(clf_fit_params,clf,X,y,n_splits=5)

    print(eval_test)
    print(eval_train)
    test_prob_final = np.zeros((len(test),))
    for model in models:
        test_prob = model.predict_proba(test)[:,1]
        test_prob_final += (test_prob*0.2)



    submission=pd.read_csv("./sub_sample/submission.csv")
    submission["score"]=test_prob_final
    submission.to_csv("submission.csv",index=False)
















