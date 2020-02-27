'''
Implementing F1 score in Keras 
'''
import pandas as pd
import numpy as np
import os
from random import sample, seed
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

#### if use tensorflow=2.0.0, then import tensorflow.keras.model_selection 
from tensorflow.keras import backend as K
from tensorflow import random_normal_initializer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, Reshape, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint 

pd.options.display.max_columns = 100
np.set_printoptions(suppress=True) 


credit_dat = pd.read_csv('creditcard.csv')

counts = credit_dat.Class.value_counts()
class0, class1 = round(counts[0]/sum(counts)*100, 2), round(counts[1]/sum(counts)*100, 2)
print(f'Class 0 = {class0}% and Class 1 = {class1}%')

import seaborn as sns
sns.set(style="whitegrid")
ax = sns.countplot(x="Class", data=credit_dat)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()/len(credit_dat)*100), (p.get_x()+0.15, p.get_height()+1000))
ax.set(ylabel='Count', 
       title='Credit Card Fraud Class Distribution')

dat = credit_dat

##### comparing the variable means:
# for c in dat.columns[:dat.shape[1]-1]:
#     print(dat[c].groupby(dat.Class).mean())


### Preprocess the training and testing data 
### save 20% for final testing 
def Pre_proc(dat, current_test_size=0.2, current_seed=42):    
    x_train, x_test, y_train, y_test = train_test_split(dat.iloc[:, 0:dat.shape[1]-1], 
                                                        dat['Class'], 
                                                        test_size=current_test_size, 
                                                        random_state=current_seed)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    y_train, y_test = np.array(y_train), np.array(y_test)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = Pre_proc(dat)


### Defining the custom metric function F1
def custom_f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

### Defining the Callback Metrics
class Metrics(Callback):
    def __init__(self, validation):   
        super(Metrics, self).__init__()
        self.validation = validation    
            
        print('validation shape', len(self.validation[0]))
        
    def on_train_begin(self, logs={}):        
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]   
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()        
    
        val_f1 = round(f1_score(val_targ, val_predict), 6)
        val_recall = round(recall_score(val_targ, val_predict), 6)     
        val_precision = round(precision_score(val_targ, val_predict), 6)
        
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
 
        print(f' — val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}')


### Building a neural nets 
def runModel(x_tr, y_tr, x_val, y_val, epos=20, my_batch_size=112):  
    ## weight_init = random_normal_initializer(mean=0.0, stddev=0.05, seed=9125)
    inp = Input(shape = (x_tr.shape[1],))
    
    x = Dense(1024, activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
        
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    
    return model 
 
### CV for the model 
models = []
f1_cv, precision_cv, recall_cv = [], [], []

current_folds = 3
current_epochs = 5
current_batch_size = 112

## macro_f1 = True for Callback 
macro_f1 = True

kfold = StratifiedKFold(current_folds, random_state=42, shuffle=True)
for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X=x_train, y=y_train)):

    print('---- Starting fold %d ----'%(k_fold+1))
    
    x_tr, y_tr = x_train[tr_inds], y_train[tr_inds]
    x_val, y_val = x_train[val_inds], y_train[val_inds]
    
    model = runModel(x_tr, y_tr, x_val, y_val, epos=current_epochs)
    
    if macro_f1:        
        model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[])  
        model.fit(x_tr, 
                  y_tr, 
                  callbacks=[Metrics(validation=(x_val, y_val))], 
                  epochs=current_epochs, 
                  batch_size=current_batch_size,   
                  verbose=1)
    else:
        model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[custom_f1, 'accuracy'])
        model.fit(x_tr, 
                  y_tr,                  
                  epochs=current_epochs, 
                  batch_size=current_batch_size,   
                  verbose=1)
    
    models.append(model)
    
    y_val_pred = model.predict(x_val)
    y_val_pred_cat = (np.asarray(y_val_pred)).round() 

    ### Get performance metrics 
    f1, precision, recall = f1_score(y_val, y_val_pred_cat), precision_score(y_val, y_val_pred_cat), recall_score(y_val, y_val_pred_cat)
    
    print("the fold %d f1 score is %f"%((k_fold+1), f1))
   
    f1_cv.append(round(f1, 6))
    precision_cv.append(round(precision, 6))
    recall_cv.append(round(recall, 6))        

print('mean f1 score = %f'% (np.mean(f1_cv)))    
 

### Predicting the hold-out testing data        
def predict(x_test):
    model_num = len(models)
    for k, m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_test, batch_size=current_batch_size)
        else:
            y_pred += m.predict(x_test, batch_size=current_batch_size)
            
    y_pred = y_pred / model_num    
    
    return y_pred

y_test_pred_cat = predict(x_test).round()

cm = confusion_matrix(y_test, y_test_pred_cat)
f1_final = round(f1_score(y_test, y_test_pred_cat), 6)

print(cm)


### print the confusion matrix 
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax=ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['business', 'health']); ax.yaxis.set_ticklabels(['health', 'business']);


def printCM(confusion_matrix=cm, title=None):    
    sns.set(style="white")
    colorway = plt.cm.gist_earth
    
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=colorway, alpha=0.15)  
    classNames = ['0','1']
    
    if title:
        plt.title('Confusion Matrix')
        
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)  
    plt.yticks(tick_marks, classNames)
    
    s = [['TN','FP'], ['FN', 'TP']]
    
    [plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j])) for j in range(2) for i in range(2)]
            
    plt.show()


plt.clf()
printCM(cm)
print(f'Testing F1 score = {f1_final}')













