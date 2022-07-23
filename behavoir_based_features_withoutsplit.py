import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from matplotlib import pyplot as pt
import gensim
import scikitplot.plotters as skplt
import nltk
#from xgboost import XGBClassifier
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import requests
import psycopg2
from sklearn.datasets import fetch_20newsgroups
from keras.layers import  Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras import layers
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing
import time
import itertools
from matplotlib import pyplot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns


def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks =np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
  
    if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
            1#print('Confusion matrix, without normalization')

            #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur3 = conn.cursor()
#cur1.execute("select * from urls_with_handcraft2")
#cur2.execute("select url,typ from bench_mark2")

##cur3.execute("select  "
##+"Min_Val_Sent ,"
###+"Avg_Val_Sent  ,"
##+"Max_Value_Received  ,"
###+"Min_Value_Received  ,"
##+"Avg_Value_Received  ,"
##+"Txn_fee_in ,"
##+"Txn_fee_out ,"
###+"Total_Txn_fee ,"
###+"gasUsed_in ,"
###+"gasUsed_out ,"
###+"Failed_Txn_in ,"
###+"Failed_Txn_out ,"
##+"Total_Failed_Txn ,"
###+"Success_Txn_in ,"
###+"Success_Txn_out ,"
###+"Total_Success_Txn ,"
##+"Std_value_in  ,"
###+"mean_in  ,"
###+"Std_gasPrice_in  ,"
###+"AP_gasUsed_in  ,"
###+"AP_gasUsed_out  ,"
###+"Avg_value_out  ,"
###+"Avg_value_in  ,"
###+"gasPrice_in ,"
###+"gasPrice_out ,"
###+"gas_used_out ,"
###+"gas_used_in ,"
###+"Avg_gasPrice_in   ,"
##+"Avg_gasPrice_out   , " 
###+"TxnSent ,"
###+"TxnReceived ,"
##+"Total_Txn  ,"
###+"Valueout  ,"
##+"Valuein  ,"
###+"Value_difference   ,"
###+"Per_TxnSent  ,"
###+"Per_TxnReceived  ,"
###+"distinct_address   ,"
###+"Unique_TxnReceived  ,"
###+"Unique_TxnSent  ,"
###+"distinct_recived_address ,"
###+"distinct_sent_address ,"
###+"difftimelast  ,"
###+"Avg_time  ,"
###+"lstvalue  ,"
###+"First_Txn_Value  ,"
###+"Last_Txn_Bit ,"
###+"First_Txn_Bit ,"
###+"mean_in_time  ,"
###+"Avg_time_in  ,"
##+"Txn_fee_contract_create ,"
##+"Per_gasUsed_contract_create  ," 
##+"gasPrice_contract_create ,"
##+"Gini_amt_out   ,"
##+"Gini_amt_in   ,"
##+"label  from behaivour_features_10 ")
###rows1 = cur1.fetchall()
###rows2 = cur2.fetchall()
##rows3 = cur3.fetchall()


cur3.execute("select   "
+"Min_Val_Sent ,"
+"Avg_Val_Sent  ,"
+"Max_Value_Received  ,"
#+"Min_Value_Received  ,"
+"Avg_Value_Received  ,"
+"Txn_fee_in ,"
#+"Txn_fee_out ,"
+"Total_Txn_fee ,"
+"gasUsed_in ,"
#+"gasUsed_out ,"
+"Failed_Txn_in ,"
+"Failed_Txn_out ,"
+"Total_Failed_Txn ,"
+"Success_Txn_in ,"
#+"Success_Txn_out ,"
+"Total_Success_Txn ,"
+"Std_value_in  ,"
#+"mean_in  ,"
+"Std_gasPrice_in  ,"
+"AP_gasUsed_in  ,"
#+"AP_gasUsed_out  ,"
#+"Avg_value_out  ,"
#+"Avg_value_in  ,"
+"gasPrice_in ,"
+"gasPrice_out ,"
+"gas_used_out ,"
+"gas_used_in ,"
+"Avg_gasPrice_in   ,"
+"Avg_gasPrice_out   , " 
#+"TxnSent ,"
+"TxnReceived ,"
+"Total_Txn  ,"
#+"Valueout  ,"
+"Valuein  ,"
#+"Value_difference   ,"
#+"Per_TxnSent  ,"
#+"Per_TxnReceived  ,"
#+"distinct_address   ,"
+"Unique_TxnReceived  ,"
#+"Unique_TxnSent  ,"
+"distinct_recived_address ,"
#+"distinct_sent_address ,"
+"difftimelast,"
#+"Avg_time  ,"
#+"lstvalue  ,"
#+"First_Txn_Value  ,"
#+"Last_Txn_Bit ,"
#+"First_Txn_Bit ,"
#+"mean_in_time  ,"
#+"Avg_time_in  ,"
+"Txn_fee_contract_create ,"
+"Per_gasUsed_contract_create  ," 
+"gasPrice_contract_create ,"
+"Gini_amt_out   ,"
+"Gini_amt_in   ,"
+"label  from behaivour_features_10")

rows3 = cur3.fetchall()




cols = int(32)
####
x = []
y = []
for i in rows3:
  row1 = []
  row2 = []
  for j in range(cols):
        if i[j]!='NaN':
            row1.append(float(i[j]))
        else:
            row1.append(int('0'))
    
  #row2.append(int(i[59]))
  y.append(int(i[32]))
  x.append(row1)
##
##for i in x:
## print(i)
####
##for i in y:
##    print(i)


from sklearn import preprocessing
# separate the data from the target attributes
# normalize the data attributes
#x = preprocessing.normalize(x)
##for i in x:
##    print(i)
x = preprocessing.scale(x)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

##sum_ifram=0
##sum_login=0
#ext_inter_ratio=0

   



##for i in y:
##   print(i)
from sklearn.model_selection import train_test_split 
X_train2, X_test2, train_y2, valid_y2 = train_test_split(x, y, test_size=0.2, random_state=0)


##trainDF = pandas.DataFrame()
##trainDF['label'] =train_y2
##pd.value_counts(trainDF['label']).plot.bar()
##pt.title('Fraud class histogram')
##pt.xlabel('Class')
##pt.ylabel('Frequency')
##trainDF['label'].value_counts()


#for j in x:
 #   ext_inter_ratio=ext_inter_ratio+j[6]
##    sum_login=sum_login+j[13]
##    sum_total=sum_total+j[12]
##
##
#print("\n",ext_inter_ratio/len(x))    
##print("\n",sum_login/len(x))    
##print("\n",sum_total/len(x))

encoder = preprocessing.LabelEncoder()
train_y2 = encoder.fit_transform(train_y2)
valid_y2 = encoder.fit_transform(valid_y2)
X_train2=np.array(X_train2, dtype='float32')
X_test2=np.array(X_test2, dtype='float32')

#for i in X_test2:
 # print(i)


#print(len(X_train2))
##print(len(train_y2))
##print(len(X_test2))
##print(len(valid_y2))



def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    t0=time.time()
    classifier.fit(feature_vector_train, label)
    t1=time.time()
    t2=t1-t0
    print("\n traing time is:",t2)
    # predict the labels on validation dataset
    t00=time.time()
    predictions = classifier.predict(feature_vector_valid)
    t11=time.time()
    t22=t11-t00
    print("\n testing time is:",t22)

##
##    for i,j in valid_y2,predictions:
##      print(i," ",j)
##  for i in 
##    cur3.execute("INSERT INTO result_classifier2 VALUES(%s,%s)",(valid_y2,predictions))
##    conn.commit()

##    for i in valid_y2:
##         cur3.execute("INSERT INTO valid_y(valid_y) VALUES(%s)",str(i))
##    for i in predictions:
##         cur3.execute("INSERT INTO predictions(predictions) VALUES(%s)",str(i))
##         
##    conn.commit()

    j=0
    tp=0
    tn=0
    fp=0
    fn=0
    p=0
    l=0
    for i in valid_y2:
        if i==1 and predictions[j]==1:
            tp=tp+1
        if i==0 and predictions[j]==0:
            tn=tn+1
        if i==0 and predictions[j]==1:
            fp=fp+1
        if i==1 and predictions[j]==0:
            fn=fn+1
        if i==1:
            p=p+1
        if i==0:
            l=l+1
            
        j=j+1

    tpr=float(tp/p)
    tnr=float(tn/l)
    fpr=float(fp/l)
    fnr=float(fn/p)

    print("\n","tpr=",tpr)
    print("\n","fpr=",fpr)
    print("\n","fnr=",fnr)
    print("\n","tnr=",tnr)
    print("\n","number of legit= ",l)
    print("\n","number of phishing= ",p)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
        
    precision_score=(tpr/(tpr+fpr))*100
    recall_score=(tpr/(tpr+fnr))*100
    f1_score=(2*precision_score*recall_score)/(precision_score+recall_score)
    accuracy=((tpr+tnr)/(tpr+tnr+fpr+fnr))*100
    
        #print("precision_score: ",metrics.precision_score(predictions,valid_y)*100)
    print("precision_score: ",precision_score)
    #print("f1_score: ",metrics.f1_score(predictions,valid_y)*100)
    print("f1_score: ",f1_score)
    #print("roc_auc_score: ",metrics.roc_auc_score(predictions,valid_y)*100)
    #print("roc_auc_score: ",metrics.roc_auc_score(valid_y2,predictions)*100)
    #print("recall_score: ",metrics.recall_score(predictions,valid_y)*100)
    print("recall_score: ",recall_score)
    print("accuracy: ",accuracy)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 14

    from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
    cnf_matrix_tra = confusion_matrix(valid_y2, predictions)
    print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix_tra , classes=class_names)
    plt.show()
    fpr, tpr, thresholds = roc_curve(valid_y2, predictions)
    roc_auc = auc(fpr,tpr)
    #Plot ROC
    #plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    sns.set(font_scale=1.4)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate',fontname='Times New Roman')
    plt.xlabel('False Positive Rate',fontname='Times New Roman')
    plt.show()
    

    conf_mat = confusion_matrix(y_true=valid_y2, y_pred=predictions)
    print('Confusion matrix:\n', conf_mat)

    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()
    
    return accuracy



##
#####Naive Bayes on Character Level TF IDF Vectors
##accuracy = train_model(naive_bayes.MultinomialNB(), train_data, train_y, test_data)
##accuracy = train_model(naive_bayes.MultinomialNB(), train_x, train_y, valid_x)
##print ("\nNB, MultinomialNB accuracy: ", accuracy)

##gnb = GaussianNB() 
##gnb.fit(xtrain_count.toarray(), train_y)
## making predictions on the testing set 
##y_pred = gnb.predict(xvalid_count.toarray())
## comparing actual response values (y_test) with predicted response values (y_pred) 
##from sklearn import metrics 
##print("\nGaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(valid_y, y_pred)*100)
##print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_pred, valid_y)*100)
##print("Gaussian Naive Bayes model accuracy(in %):", metrics.confusion_matrix(valid_y, y_pred))
##print("Gaussian Naive Bayes model precision_score(in %):", metrics.precision_score(valid_y,y_pred)*100)
##print("Gaussian Naive Bayes model precision_score(in %):", metrics.precision_score(y_pred, valid_y)*100)
##print("Gaussian Naive Bayes model f1_score(in %):", metrics.f1_score(valid_y, y_pred)*100)
##print("Gaussian Naive Bayes model f1_score(in %):", metrics.f1_score(y_pred, valid_y)*100)
##print("Gaussian Naive Bayes model roc_auc_score(in %):", metrics.roc_auc_score(valid_y,y_pred)*100)
##print("Gaussian Naive Bayes model roc_auc_score(in %):", metrics.roc_auc_score(y_pred,valid_y)*100)
##print("Gaussian Naive Bayes model recall_score(in %):", metrics.recall_score(valid_y,y_pred)*100)
##print("Gaussian Naive Bayes model recall_score(in %):", metrics.recall_score(y_pred,valid_y)*100)




## #Linear Classifier on Character Level TF IDF Vectors
##accuracy = train_model(linear_model.LogisticRegression(), X_train2, train_y2, X_test2)
##print ("\nLR, hyperlink features: ", accuracy)
##
### RF on Word Level TF IDF Vectors
####print("RF TF_IDF:")
####print('\n')
##
##accuracy = train_model(ensemble.RandomForestClassifier(), X_train2, train_y2, X_test2)
##print ("\nRF,hyperlink features: ", accuracy)
##print('\n')
##
##accuracy = train_model(xgboost.XGBClassifier(),X_train2, train_y2,X_test2)
##print ("Xgb, hyperlink features: ", accuracy)
####print('\n')
##
# SVM on Ngram Level TF IDF Vectors
#accuracy = train_model(svm.SVC(),X_train2 , train_y2,X_test2 )
#print ("SVM, CharLevel Vectors: ", accuracy)
##
##
##

#####Naive Bayes on Character Level TF IDF Vectors
##accuracy = train_model(naive_bayes.MultinomialNB(), X_train2, train_y2, X_test2)
###accuracy = train_model(naive_bayes.MultinomialNB(), train_x, train_y, valid_x)
##print ("\nNB, hyperlink features: ", accuracy)
##
##gnb = GaussianNB() 
##gnb.fit(X_train2, train_y2)
### making predictions on the testing set 
##y_pred = gnb.predict(X_test2)
## #comparing actual response values (y_test) with predicted response values (y_pred) 
##from sklearn import metrics 
##print("\nGaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(valid_y2, y_pred)*100)
###print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_pred, valid_y)*100)
###print("Gaussian Naive Bayes model accuracy(in %):", metrics.confusion_matrix(valid_y, y_pred))
##print("Gaussian Naive Bayes model precision_score(in %):", metrics.precision_score(valid_y2,y_pred)*100)
####print("Gaussian Naive Bayes model precision_score(in %):", metrics.precision_score(y_pred, valid_y)*100)
##print("Gaussian Naive Bayes model f1_score(in %):", metrics.f1_score(valid_y2, y_pred)*100)
####print("Gaussian Naive Bayes model f1_score(in %):", metrics.f1_score(y_pred, valid_y)*100)
##print("Gaussian Naive Bayes model roc_auc_score(in %):", metrics.roc_auc_score(valid_y2,y_pred)*100)
####print("Gaussian Naive Bayes model roc_auc_score(in %):", metrics.roc_auc_score(y_pred,valid_y)*100)
##print("Gaussian Naive Bayes model recall_score(in %):", metrics.recall_score(valid_y2,y_pred)*100)
####print("Gaussian Naive Bayes model recall_score(in %):", metrics.recall_score(y_pred,valid_y)*100)
####

##
##def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
##    """
##    buildModel_DNN_Tex(shape, nClasses,dropout)
##    Build Deep neural networks Model for text classification
##    Shape is input feature space
##    nClasses is number of classes
##    """
##    model = Sequential()
##    node = 512 # number of nodes
##    nLayers = 4 # number of  hidden layer
##   # model.add(layers.Embedding(vocab_size+1, output_dim=95, weights=[embedding_weights], input_length=1014))
##    model.add(Dense(node,input_dim=shape,activation='relu'))
##    model.add(Dropout(dropout))
##    for i in range(0,nLayers):
##        model.add(Dense(node,input_dim=node,activation='relu'))
##        model.add(Dropout(dropout))
##    model.add(Dense(nClasses, activation='softmax'))
##    model.compile(loss='sparse_categorical_crossentropy',
##                  optimizer='adam',
##                  metrics=['accuracy'])
##    return model
##
##
##ckpt_callback = ModelCheckpoint('keras_model', 
##                                 monitor='val_accuracy', 
##                                 verbose=1, 
##                                 save_best_only=True, 
##                                 mode='auto')
##
##model_DNN = Build_Model_DNN_Text(X_train2.shape[1], 2)
##history1=model_DNN.fit(np.array(X_train2),np.array(train_y2),
##                              validation_data=(np.array(X_test2), np.array(valid_y2)),
##                              epochs=20,
##                              batch_size=128,
##                              verbose=2, callbacks =[ckpt_callback])
###model = load_model('keras_model')
###predicted = model.predict(X_test2)
##predicted = model_DNN.predict(X_test2)
##predicted = np.argmax(predicted, axis=1)
##print("accuracy_score: ",metrics.accuracy_score(valid_y2, predicted)*100)
##print("recall_score: ",metrics.recall_score(valid_y2, predicted)*100)
##print("precision_score: ",metrics.precision_score(valid_y2,predicted)*100)
##print("roc_auc_score: ",metrics.roc_auc_score(valid_y2, predicted)*100)
##print("f1_score: ",metrics.f1_score(valid_y2, predicted)*100)
##
##
##

##
##

from sklearn.datasets import make_classification
##trainDF['feature'] =x_train
X=X_train2
y=train_y2
##from imblearn.over_sampling import ADASYN
##oversample = ADASYN()
##X, y = oversample.fit_resample(X, y)
trainDF = pandas.DataFrame()
trainDF['label'] =y

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 14
sns.set(font_scale=1.4)
target_count = trainDF.label.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar' )
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        pt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    pt.title(label)
    pt.legend(loc='upper right')
    pt.show()

from collections import Counter
from numpy import where
# define dataset
##X, y = make_classification(n_samples=10000, n_features=59, n_redundant=0,
##	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
#summarize class distribution
counter = Counter(y)
print(counter)
#scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
from imblearn.over_sampling import ADASYN

oversample = ADASYN()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

##accuracy1 = train_model(KNeighborsClassifier(),X , y,X_test2)
##print ("KNeighborsClassifier, behavior based features: ", accuracy1)
##
##accuracy2 = train_model(DecisionTreeClassifier(),X , y,X_test2)
##print ("DecisionTreeClassifier, behavior based features: ", accuracy2)
##
##accuracy3 = train_model(AdaBoostClassifier(),X , y,X_test2)
##print ("AdaBoostClassifier, behavior based features: ", accuracy3)
##
####
##accuracy4 = train_model(RandomForestClassifier(),X , y,X_test2)
##print ("RandomForestClassifier, behavior based features: ", accuracy4)
####
##accuracy5 = train_model(ExtraTreesClassifier(),X , y,X_test2)
##print ("ExtraTreesClassifier, FS5: ", accuracy5)
##
##accuracy6 = train_model(GradientBoostingClassifier(),X , y,X_test2)
##print ("GradientBoostingClassifier, behavior based features: ", accuracy6)
##accuracy7 = train_model(BaggingClassifier(),X , y,X_test2)
##print ("BaggingClassifier, behavior based features: ", accuracy7)
##
##accuracy8 = train_model(xgboost.XGBClassifier(),X , y,X_test2)
##print ("XGBClassifier, behavior based features: ", accuracy8)


from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier

est_XGB = xgboost.XGBClassifier()
est_EX = ExtraTreesClassifier()
est_GB = GradientBoostingClassifier()

est_BA = RandomForestClassifier()
##
##est_DT = DecisionTreeClassifier()
##est_KN = KNeighborsClassifier()
##est_ET = ExtraTreesClassifier()

est_Ensemble = VotingClassifier(estimators=[('est_EX', est_EX), ('est_GB', est_GB)],
                        voting='soft',
                        weights=[1,1])
####
accuracy9 = train_model(est_Ensemble, X, y, X_test2)
print ("\nEnsemble, behavior based features", accuracy9)
##
##score_Ensemble=est_Ensemble.fit(X_train2,train_y2).score(X_test2,valid_y2)
##
##accuracy=train_model(est_AB, X_train2, train_y2, X_test2)
##print ("\naddnost: ", accuracy)
##print('\n')
##accuracy = train_model(est_Ensemble, X_train2, train_y2, X_test2)
##print ("\nensemble hyperlink: ", accuracy)
##print('\n')

#print(score_Ensemble)

##from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier
##
##est_AB = xgboost.XGBClassifier()
####score_AB=est_AB.fit(X_train2,train_y2).score(X_test2,valid_y2)
####
##est_RF = RandomForestClassifier()
####score_RF=est_RF.fit(X_train2,train_y2).score(X_test2,valid_y2)
####
##est_Ensemble = VotingClassifier(estimators=[('AB', est_AB), ('RF', est_RF)],
##                        voting='soft',
##                        weights=[1, 1])
##
##score_Ensemble=est_Ensemble.fit(X_train2,train_y2).score(X_test2,valid_y2)
##
##
##
##accuracy=train_model(est_Ensemble, X_train2, train_y2, X_test2)
##print ("\nensemble2: ", accuracy)


model = xgboost.XGBClassifier()
#model=est_Ensemble
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
#pyplot.plot(['KNeighbors', 'DecisionTree', 'AdaBoost', 'Random Forest','Extra-Tree', 'Gradient Boost','Bagging','XGBoost','Ensemble'])

#plt.figure(figsize=(9, 4))
#plt.plot(['KNeighbors', 'DecisionTree', 'AdaBoost', 'Random Forest','Extra-Tree', 'Gradient Boost','Bagging','XGBoost','Ensemble'], [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8,accuracy9], 'ro')
#plt.plot(['KNeighbors', 'DecisionTree', 'AdaBoost', 'Random Forest','Extra-Tree', 'Gradient Boost','Bagging','XGBoost','Ensemble'], [43, 77, 55, 44, 99, 43, 90, 88,33], 'bo')

##plt.figure(figsize=(9, 4))
##plt.plot(['KNeighbors', 'DecisionTree', 'AdaBoost', 'Random Forest','Extra-Tree', 'Gradient Boost','Bagging','XGBoost','Ensemble'], [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8,accuracy9], 'ro')



pyplot.show()












