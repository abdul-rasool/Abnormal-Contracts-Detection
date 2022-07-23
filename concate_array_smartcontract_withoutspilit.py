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
import gensim
import scikitplot.plotters as skplt
import nltk
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from autocorrect import spell
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from scipy import sparse
import time
import scikitplot.plotters as skplt
from matplotlib import pyplot
from pyevmasm import instruction_tables, disassemble_hex, disassemble_all, assemble_hex
instruction_table = instruction_tables['byzantium']
instruction_table[20]
instruction_table['EQ']
import itertools
from matplotlib import pyplot as plt
from matplotlib import pyplot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier


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

def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
    return text.lower()

def listToString(s):  
    # initialize an empty string 
    str1  = " "
    
    # traverse in the string   
    for ele in s:  
        str1 +=" "+ ele 
    
    # return string   
    return str1 

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')
##
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
            token=re.sub(r'[^\x00-\x7f]',r'', token)
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
    return result

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

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    j=0
    tp=0
    tn=0
    fp=0
    fn=0
    p=0
    l=0
    for i in valid_y:
        if i==1 and predictions[j]==1:
            tp=tp+1
        if i==0 and predictions[j]==0:
            tn=tn+1
        if i==0 and predictions[j]==1:
            fp=fp+1
            cur3.execute("INSERT INTO fp_contracts_new(id,valid_y) VALUES(%s,%s)",(int(id_test[j]),int(i)))
            #conn.commit()
        if i==1 and predictions[j]==0:
            fn=fn+1
            cur3.execute("INSERT INTO fn_contracts_new(id,valid_y) VALUES(%s,%s)",(int(id_test[j]),int(i)))
            #conn.commit()
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



    precision_score=(tpr/(tpr+fpr))*100
    recall_score=(tpr/(tpr+fnr))*100
    f1_score=(2*precision_score*recall_score)/(precision_score+recall_score)
    accuracy=((tpr+tnr)/(tpr+tnr+fpr+fnr))*100
    
        #print("precision_score: ",metrics.precision_score(predictions,valid_y)*100)
    print("precision_score: ",precision_score)
    #print("f1_score: ",metrics.f1_score(predictions,valid_y)*100)
    print("f1_score: ",f1_score)
    #print("roc_auc_score: ",metrics.roc_auc_score(predictions,valid_y)*100)
    print("roc_auc_score: ",metrics.roc_auc_score(valid_y,predictions)*100)
    #print("recall_score: ",metrics.recall_score(predictions,valid_y)*100)
    print("recall_score: ",recall_score)
    print("accuracy: ",accuracy)
    
    
    from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
    cnf_matrix_tra = confusion_matrix(valid_y, predictions)
    print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
    plt.show()
    fpr, tpr, thresholds = roc_curve(valid_y, predictions)
    roc_auc = auc(fpr,tpr)
    #Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    return accuracy




conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur3 = conn.cursor()
cur3.execute("select id,opcode,label from training_code_chen order by id")
rows2 = cur3.fetchall()
##
##
train_x, valid_x, train_y, valid_y,id_train,id_test = [],[],[],[],[],[]

##for row in rows2:
####    #count=0;
##    train_y.append(int(row[2]))
##    id_train.append(row[0])
##    #for i in row:
####    #    if count<=62:
####     #       value=int(row[count])
####      #      count=count+1
##    processeddata=preprocess(row[1])
##    train_x.append(listToString(processeddata))

for row in rows2:
    train_y.append(row[2])
    lines=""
    smart_row=""
    lines=str(row[1]).split('\n')
    for line in lines:
        opcode=""
        #print(line)
    #print("new smart \n");
        for element in line:
            if 48 <= ord(element) <= 57:    
                break
            else:
                opcode+=element
        smart_row+=" "+opcode
        #print(opcode)
    #text_features=text_cleaner(smart_row)
    #processeddata=preprocess(text_features)
    #processeddata=preprocess(smart_row)
    train_x.append(smart_row)
##
##

cur3.execute("select id,opcode,label from test_code_chen order by id")
rows3 = cur3.fetchall()

for row in rows3:
    valid_y.append(row[2])
    id_test.append(row[0])
    lines=""
    smart_row=""
    lines=str(row[1]).split('\n')
    for line in lines:
        opcode=""
        #print(line)
    #print("new smart \n");
        for element in line:
            if 48 <= ord(element) <= 57:    
                break
            else:
                opcode=opcode+element
        smart_row=smart_row+" "+opcode
    #text_features=text_cleaner(smart_row)
    #processeddata=preprocess(text_features)    
    #processeddata=preprocess(smart_row)
    #valid_x.append(listToString(processeddata))    
    valid_x.append(smart_row)
    
##cur3.execute("")
###rows1 = cur1.fetchall()
###rows2 = cur2.fetchall()
##rows3 = cur3.fetchall()
##for row in rows3:
####    #count=0;
##    valid_y.append(int(row[2]))
##    id_test.append(row[0])
##    #for i in row:
####    #    if count<=62:
####     #       value=int(row[count])
####      #      count=count+1
##    processeddata=preprocess(row[1])
##    valid_x.append(listToString(processeddata))
##
##tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
#####tfidf_vect_ngram_chars.fit(text)
##xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.fit_transform(train_x).toarray() 
##xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x).toarray()
##
#xtrain_tfidf_ngram_chars=sparse.csr_matrix(xtrain_tfidf_ngram_chars).toarray() 
#xvalid_tfidf_ngram_chars=sparse.csr_matrix(xvalid_tfidf_ngram_chars).toarray() 
##


####
#ngram word level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
tfidf_vect_ngram.fit(train_x)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x).toarray()
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x).toarray()
####
# 
    
tfidf_vect_ngram=np.array(xtrain_tfidf_ngram, dtype='float32')
xvalid_tfidf_ngram=np.array(xvalid_tfidf_ngram, dtype='float32')
#print(tfidf_vect_ngram.shape[0])
print(tfidf_vect_ngram.shape[1])
#print(xvalid_tfidf_ngram.shape[0])
print(xvalid_tfidf_ngram.shape[1])


count_vect = CountVectorizer(analyzer='char', token_pattern=r'\w{1,}',max_features=30000)
count_vect.fit(train_x)
#transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x).toarray()
xvalid_count =  count_vect.transform(valid_x).toarray()
xtrain_count=np.array(xtrain_count, dtype='float32')
xvalid_count=np.array(xvalid_count, dtype='float32')
#print(xtrain_count.shape[0])
print("dim:",xtrain_count.shape[1])
#print(xvalid_count.shape[0])
print("dim:",xvalid_count.shape[1])

##
##cur3.execute("select txt_feauters_t3_train.id,webpage_code33_train.url,webpage_code33_train.typ from webpage_code33_train, txt_feauters_t3_train where txt_feauters_t3_train.id= webpage_code33_train.id order by 1")
##rows2 = cur3.fetchall()
##train_x1, valid_x1, train_y1, valid_y1,id_train1,id_test1 = [],[],[],[],[],[]
##
##for row in rows2:
####    #count=0;
##    train_y1.append(int(row[2]))
##    id_train1.append(int(row[0]))
##    #for i in row:
####    #    if count<=62:
####     #       value=int(row[count])
####      #      count=count+1
##    train_x1.append(row[1])
##
##cur3.execute("select txt_feauters_t3_test.id,webpage_code33_test.url,webpage_code33_test.typ from webpage_code33_test,txt_feauters_t3_test where webpage_code33_test.id=txt_feauters_t3_test.id order by 1 ")
##rows2 = cur3.fetchall()
##for row in rows2:
####    #count=0;
##    valid_y1.append(int(row[2]))
##    id_test1.append(int(row[0]))
##    #for i in row:
####    #    if count<=62:
####     #       value=int(row[count])
####      #      count=count+1
##    
##    valid_x1.append(row[1])
##    
##tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
##tk.fit_on_texts(train_x1)
##alphabet ="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
##
##char_dict = {}
##for i, char in enumerate(alphabet):
##    char_dict[char] = i + 1
##    tk.word_index = char_dict.copy()
### Add 'UNK' to the vocabulary
##tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
##
##train_sequences = tk.texts_to_sequences(train_x1)
##test_texts = tk.texts_to_sequences(valid_x1)
##
### Padding
##train_data = pad_sequences(train_sequences, maxlen=500, padding='post')
##test_data = pad_sequences(test_texts, maxlen=500, padding='post')
### Convert to numpy array
##train_data = np.array(train_data, dtype='float32')
##test_data = np.array(test_data, dtype='float32')
##print(train_data.shape[1])
##print(train_data.shape[0])
##print(test_data.shape[1])
##print(test_data.shape[0])


##conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
##cur3 = conn.cursor()
#cur1.execute("select * from urls_with_handcraft2")
#cur2.execute("select url,typ from bench_mark2")
cur3.execute("select id,STOP,ADD,MUL,SUB,DIV,"
    +"SDIV,MOD,SMOD,ADDMOD,MULMOD,EXP,SIGNEXTEND,LT,GT,SLT,SGT,EQ,ISZERO,AND1,OR1,XOR1,NOT1,BYTE,"
    +"SHA,BALANCE,ORIGIN,CALLVALUE,CALLDATALOAD,CALLDATASIZE,CALLDATACOPY,CODESIZE,CODECOPY,GASPRICE,"
    +"EXTCODESIZE,EXTCODECOPY,BLOCKHASH,COINBASE,TIMESTAMP,NUMBER,"
    +"DIFFICULTY,GASLIMIT,POP,MLOAD,MSTORE,SLOAD,SSTORE,JUMP,JUMPI,MSIZE,GAS,"
    +"JUMPDEST,DUP,SWAP,LOG,CREATE1,CALL,CALLCODE,RETURN,ADDRESS2,label from training_code2 order by id")
#rows1 = cur1.fetchall()
#rows2 = cur2.fetchall()
rows3 = cur3.fetchall()

cols = int(59)
####
x_train2 = []
y_train2 = []

for i in rows3:
  row1 = []
  #row2 = []
  for j in range(cols):
    row1.append((i[j+1]))
    
  #row2.append(int(i[60]))
  y_train2.append(int(i[60]))
  x_train2.append(row1)

from sklearn import preprocessing
# separate the data from the target attributes
# normalize the data attributes
#x_train2 = preprocessing.normalize(x_train2)

cur3.execute("select id,STOP,ADD,MUL,SUB,DIV,"
    +"SDIV,MOD,SMOD,ADDMOD,MULMOD,EXP,SIGNEXTEND,LT,GT,SLT,SGT,EQ,ISZERO,AND1,OR1,XOR1,NOT1,BYTE,"
    +"SHA,BALANCE,ORIGIN,CALLVALUE,CALLDATALOAD,CALLDATASIZE,CALLDATACOPY,CODESIZE,CODECOPY,GASPRICE,"
    +"EXTCODESIZE,EXTCODECOPY,BLOCKHASH,COINBASE,TIMESTAMP,NUMBER,"
    +"DIFFICULTY,GASLIMIT,POP,MLOAD,MSTORE,SLOAD,SSTORE,JUMP,JUMPI,MSIZE,GAS,"
    +"JUMPDEST,DUP,SWAP,LOG,CREATE1,CALL,CALLCODE,RETURN,ADDRESS2,label from test_code2 order by id")   
#rows1 = cur1.fetchall()
#rows2 = cur2.fetchall()
rows3 = cur3.fetchall()
cols = int(59)
####
x_test2 = []
y_test2 = []

for i in rows3:
  row1 = []
  #row2 = []
  for j in range(cols):
    row1.append((i[j+1]))
    
  #row2.append(int(i[60]))
  y_test2.append(int(i[60]))
  x_test2.append(row1)

#x_train2 = preprocessing.normalize(x_train2)
#x_test2 = preprocessing.normalize(x_test2)

x_train2=np.array(x_train2, dtype='float32')
x_test2=np.array(x_test2, dtype='float32')
#print(x_train2.shape[0])
print(x_train2.shape[1])
#print(x_test2.shape[0])
print(x_test2.shape[1])

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
y_train2 = encoder.fit_transform(y_train2)
y_test2 = encoder.fit_transform(y_test2)


#sequenc features
cur3 = conn.cursor()
cur3.execute("select id,opcode,label from training_code2 order by 1 ")
rows2 = cur3.fetchall()

#labels, texts,texts2 = [], [],[]
x_train3, y_train3, x_test3,y_test3=[],[],[],[]
time11=time.time()

for row in rows2:
    y_train3.append(row[2])
    lines=""
    smart_row=""
    row1 = []
    le=0
    lines=str(row[1]).split('\n')
    for line in lines:
        opcode=""
        p=0
        le+=1
        #print(line)
    #print("new smart \n");
        for element in line:
            if element == " ":
                p=1
                break
            else:
               opcode+=element
        if le>1500:
                break;      
        try:
            if p!=1:
            #smart_row=smart_row+" "+opcode
                row1.append(int(assemble_hex(opcode),16))
        except Exception: 
            cur3.execute("insert into t1 values(%s,%s)",(row[0],opcode))
            #conn.commit()
    le2=1500-len(row1)
    for i in range(le2):
        row1.append(int(0))
    x_train3.append(row1)

cur3.execute("select id,opcode,label from test_code2 order by 1")
rows3 = cur3.fetchall()


for row in rows3:
    y_test3.append(row[2])
    lines=""
    smart_row=""
    row1 = []
    lines=str(row[1]).split('\n')
    le=0
    for line in lines:
        le+=1
        opcode=""
        p=0
        #print(line)
    #print("new smart \n");
        for element in line:
            if element == " ":
                p=1
                break
            else:
                opcode=opcode+element
        try:
            if p!=1:
                row1.append(int(assemble_hex(opcode),16))
            if le>1500:
                break;
        except Exception: 
            cur3.execute("insert into t1 values(%s,%s)",(row[0],opcode))
            
            #smart_row=smart_row+" \n"+opcode
    #text_features=text_cleaner(smart_row)
    #processeddata=preprocess(text_features)    
    #processeddata=preprocess(smart_row)
    #valid_x.append(listToString(processeddata))
    #print(assemble_hex(smart_row))
    #print(int(assemble_hex(smart_row),16))
    #valid_x.append(int(assemble_hex(smart_row),16))
    
    le2=1500-len(row1)
    for i in range(le2):
        row1.append(int(0))
    x_test3.append(row1)

x_train3=np.array(x_train3, dtype='float32')
x_test3=np.array(x_test3, dtype='float32')
print(x_train3.shape[0])
y_train3 = encoder.fit_transform(y_train3)
y_test3 = encoder.fit_transform(y_test3)


cur4 = conn.cursor()
cur5 = conn.cursor()


#behavoiur features
cur4.execute("select id,   "
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
+"difftimelast  ,"
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
+"label  from tarining_behaviour_shuhui_fan order by id")

####
##cur4.execute("select id, "
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
##+"label  from tarining_behaviour order by id")
rows3 = cur4.fetchall()
cols = int(32)
####
x_train4 = []
y_train4 = []
for i in rows3:
  row1 = []
  row2 = []
  for j in range(cols):
        if i[j+1]!='NaN':
            row1.append(float(i[j+1]))
        else:
            row1.append(int('0'))
    
  #row2.append(int(i[59]))
  y_train4.append(int(i[33]))
  x_train4.append(row1)

##for i in y_train4:
##    print(i)

cur5.execute("select id,   "
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
+"difftimelast  ,"
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
+"label  from test_behaviour_shuhui_fan order by id")

##cur5.execute("select id,  "
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
##+"Std_value_in,"
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
##+"label  from test_behaviour order by id")
rows4 = cur5.fetchall()
cols = int(32)
####
x_test4 = []
y_test4 = []
for i in rows4:
  row1 = []
  row2 = []
  for j in range(cols):
        if i[j+1]!='NaN':
            row1.append(float(i[j+1]))
        else:
            row1.append(int('0'))
    
  #row2.append(int(i[59]))
  y_test4.append(int(i[33]))
  x_test4.append(row1)


from sklearn import preprocessing
# separate the data from the target attributes
# normalize the data attributes
#x = preprocessing.normalize(x)
##for i in x:
##    print(i)
##x = preprocessing.scale(x)
##min_max_scaler = preprocessing.MinMaxScaler()
##x = min_max_scaler.fit_transform(x)

from sklearn import preprocessing
# separate the data from the target attributes
# normalize the data attributes
#x = preprocessing.normalize(x)
##for i in x:
##    print(i)
##x_train4 = preprocessing.scale(x_train4)
##min_max_scaler = preprocessing.MinMaxScaler()
##x_train4 = min_max_scaler.fit_transform(x_train4)
####x_test4 = preprocessing.scale(x_test4)
####min_max_scaler = preprocessing.MinMaxScaler()
##x_test4 = min_max_scaler.fit_transform(x_test4)

##x_train4 = preprocessing.normalize(x_train4)
##x_test4 = preprocessing.normalize(x_test4)



encoder = preprocessing.LabelEncoder()
y_train4 = encoder.fit_transform(y_train4)
y_test4 = encoder.fit_transform(y_test4)
x_train4=np.array(x_train4, dtype='float32')
x_test4=np.array(x_test4, dtype='float32')


#sourc_code fetaures

cur6 = conn.cursor()
cur7 = conn.cursor()
cur6.execute("select tarining_source_chen.source_code,label from tarining_source_chen order by id")
#rows1 = cur1.fetchall()
#rows2 = cur2.fetchall()
rows6 = cur6.fetchall()

x_train5, x_test5, y_train5, y_test5,id_test = [],[],[],[],[]
bad_chars={';','0','1','2','3','4','5',
                    '6','7','8','9','\n',':','!',"*",
                    '[',']','{','(',')',",",';','.','!','?',
                    ':',"'",'"\"','/',"\\",'|','_','@','#',
                    '$','%','^','&','*','~','`','+','"','=',
                    '<','>','(',')','[',']','{','}'}      
for row in rows6:
##    #count=0;
    y_train5.append(int(row[1]))
    #for i in row:
##    #    if count<=62:
##     #       value=int(row[count])
##      #      count=count+1

    str1=str(row[0])
    for i in bad_chars: 
        str1 = str1.replace(i, ' ')
    str1 = str1.replace('-', ' ')
    str1 = str1.replace(' n', ' ')
    text_features=text_cleaner(str1)
    processeddata=preprocess(text_features)
    x_train5.append(listToString(processeddata))

cur7.execute("select id,test_source_chen.source_code,label from test_source_chen order by id")
#rows1 = cur1.fetchall()
#rows2 = cur2.fetchall()
rows3 = cur7.fetchall()
for row in rows3:
##    #count=0;
    y_test5.append(int(row[2]))
    id_test.append(row[0])
    #for i in row:
##    #    if count<=62:
##     #       value=int(row[count])
##      #      count=count+1
    str1=str(row[1])
    for i in bad_chars: 
        str1 = str1.replace(i, ' ')
    str1 = str1.replace('-', ' ')
    str1 = str1.replace(' n', ' ')
    text_features=text_cleaner(str1)
    processeddata=preprocess(text_features)
    x_test5.append(listToString(processeddata))

    
encoder = preprocessing.LabelEncoder()
y_train5 = encoder.fit_transform(y_train5)
y_test5 = encoder.fit_transform(y_test5)

##
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
#tfidf_vect_ngram_chars.fit(text)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.fit_transform(x_train5).toarray() 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(x_test5).toarray()
xtrain_tfidf_ngram_chars = np.array(xtrain_tfidf_ngram_chars, dtype='float32')
xvalid_tfidf_ngram_chars = np.array(xvalid_tfidf_ngram_chars, dtype='float32')
print(xtrain_tfidf_ngram_chars.shape[1])


##tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
##tfidf_vect_ngram.fit(x_train5)
##xtrain_tfidf_ngram2 =  tfidf_vect_ngram.transform(x_train5).toarray() 
##xvalid_tfidf_ngram2 =  tfidf_vect_ngram.transform(x_test5).toarray()
##xtrain_tfidf_ngram2 = np.array(xtrain_tfidf_ngram, dtype='float32')
##xvalid_tfidf_ngram2 = np.array(xvalid_tfidf_ngram, dtype='float32')
##print(xtrain_tfidf_ngram2.shape[1])

#tfidf_vect_ngram = preprocessing.normalize(tfidf_vect_ngram)
#xvalid_tfidf_ngram = preprocessing.normalize(xvalid_tfidf_ngram)
#train_data = preprocessing.normalize(train_data)
#test_data = preprocessing.normalize(test_data)


#combine all features
#tc1=time.time()
j=0
concate_train=[]
concate_train1=[]
concate_test=[]
concate_test1=[]
print(x_train4.shape[1])
#print(x_test2.shape[0])
print(x_test4.shape[1])


from sklearn import preprocessing
# separate the data from the target attributes
# normalize the data attributes
#x_train2 = preprocessing.normalize(x_train2)
#x_test2 = preprocessing.normalize(x_test2)



##print(xtrain_tfidf_ngram.shape[1])
##print(xvalid_tfidf_ngram.shape[1])
##print(xtrain_tfidf_ngram.shape[0])
##print(xvalid_tfidf_ngram.shape[0])

#combine BOW, N-GRAM, FREQUENCY
for i in xtrain_tfidf_ngram:
    concate_train.append(np.concatenate((i,xtrain_tfidf_ngram_chars[j])))
    #print(i,"\n",x_train3[j],"\n")
    j=j+1
    
concate_train=np.array(concate_train, dtype='float32')    
#concate_train = preprocessing.normalize(concate_train)

j=0
##for i in xtrain_count:
##    concate_train1.append(np.concatenate((concate_train[j],i)))
####    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1    
##########
##concate_train1=np.array(concate_train1, dtype='float32')
#####print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
print(concate_train.shape[0],"\n",concate_train.shape[1])
j=0
for i in xvalid_tfidf_ngram:
    concate_test.append(np.concatenate((i,xvalid_tfidf_ngram_chars[j])))
    #print(x_test3[j],"\n")
    j=j+1
concate_test=np.array(concate_test, dtype='float32')

print(concate_test.shape[0],"\n",concate_test.shape[1])


##tc2=time.time()
##tcombine=tc2-tc1
##print("\n vectorize time is:",tcombine)

##concate_train=sparse.csr_matrix(concate_train).toarray() 
##concate_test=sparse.csr_matrix(concate_test).toarray() 


#concate_test = preprocessing.normalize(concate_test)
##j=0   
##for i in xvalid_count:
##    concate_test1.append(np.concatenate((concate_test[j],i)))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##concate_test1=np.array(concate_test1, dtype='float32')
########print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
##print( concate_test.shape[0],"\n",concate_test.shape[1])

#combine n-gram and bagofwords



###combine url+hyperlink
##tc1=time.time()
##j=0
##concate_url_hyperlink_train=[]
##concate_url_hyperlink_text=[]
##for i in train_data:
##    concate_url_hyperlink_train.append(np.concatenate((train_data[j],x_train2[j])))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##
##concate_url_hyperlink_train=np.array(concate_url_hyperlink_train, dtype='float32')
#####concate_url_hyperlink_train = preprocessing.normalize(concate_url_hyperlink_train)
#####print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
#####print( len(concate_url_hyperlink_train),"\n",concate_url_hyperlink_train.shape[0],"\n",concate_url_hyperlink_train.shape[1])
####
##j=0
##for i in test_data:
##    concate_url_hyperlink_text.append(np.concatenate((test_data[j],x_test2[j])))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##
##concate_url_hyperlink_text=np.array(concate_url_hyperlink_text, dtype='float32')
###concate_url_hyperlink_text = preprocessing.normalize(concate_url_hyperlink_text)
##
##tc2=time.time()
##tcombine=tc2-tc1
##print("\n vectorize time is:",tcombine)

##print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
##print( len(concate_url_hyperlink_text),"\n",concate_url_hyperlink_text.shape[0],"\n",concate_url_hyperlink_text.shape[1])

##
#combine url+text
##tc1=time.time()
##j=0
##concate_url_text_train=[]
##concate_url_text_test=[]
##for i in xtrain_tfidf_ngram_chars:
##    concate_url_text_train.append(np.concatenate((i,train_data[j])))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##
##concate_url_text_train=np.array(concate_url_text_train, dtype='float32')
###concate_url_text_train = preprocessing.normalize(concate_url_text_train)
##
###print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
###print( len(concate_url_text_train),"\n",concate_url_text_train.shape[0],"\n",concate_url_text_train.shape[1])
##
##j=0
##for i in xvalid_tfidf_ngram_chars:
##    concate_url_text_test.append(np.concatenate((i,test_data[j])))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##
##concate_url_text_test=np.array(concate_url_text_test, dtype='float32')
##
##tc2=time.time()
##tcombine=tc2-tc1
##print("\n vectorize time of url+text is:",tcombine)
##
#concate_url_text_test = preprocessing.normalize(concate_url_text_test)

##print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
##print( len(concate_url_text_test),"\n",concate_url_text_test.shape[0],"\n",concate_url_text_test.shape[1])

##
##

###combine hyperlink+text
##tc1=time.time()
##j=0
##concate_hyper_text_train=[]
##concate_hyper_text_test=[]
##for i in xtrain_tfidf_ngram_chars:
##    concate_hyper_text_train.append(np.concatenate((i,x_train2[j])))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##
##concate_hyper_text_train=np.array(concate_hyper_text_train, dtype='float32')
###concate_hyper_text_train = preprocessing.normalize(concate_hyper_text_train)
##
###print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
###print( len(concate_url_text_train),"\n",concate_url_text_train.shape[0],"\n",concate_url_text_train.shape[1])
##
##j=0
##for i in xvalid_tfidf_ngram_chars:
##    concate_hyper_text_test.append(np.concatenate((i,x_test2[j])))
##    #print(i,"\n",train_data[j],"\n",x_train2[j],"\n")
##    j=j+1
##
##concate_hyper_text_test=np.array(concate_hyper_text_test, dtype='float32')
##
##tc2=time.time()
##tcombine=tc2-tc1
##print("\n vectorize time is:",tcombine)
#concate_hyper_text_test = preprocessing.normalize(concate_hyper_text_test)

#print(len(xtrain_tfidf_ngram_chars),"\n",len(train_data),"\n",len(x_train2))
#print( len(concate_url_text_test),"\n",concate_url_text_test.shape[0],"\n",concate_url_text_test.shape[1])

####
trainDF = pandas.DataFrame()
trainDF['label'] =train_y

#concate_train = preprocessing.normalize(concate_train)

X=concate_train
y=train_y


##from sklearn.manifold import TSNE
##import plotly.express as px
##
##df = px.data.iris()
##
##features = df.loc[:, :'petal_width']
##
##tsne = TSNE(n_components=3, random_state=0)
##projections = tsne.fit_transform(xtrain_tfidf_ngram, )
##
##fig = px.scatter_3d(
##    projections, x=0, y=1, z=2,
##    color=df.species, labels={'color': 'species'}
##)
##fig.update_traces(marker_size=8)
##fig.show()


##from imblearn.over_sampling import ADASYN
##from imblearn.over_sampling import BorderlineSMOTE
##oversample = ADASYN()
##X, y = oversample.fit_resample(X, y)

#visulzation of data 
##import matplotlib.pyplot as plt
##import seaborn as sns
###%matplotlib inline
##import pandas as pd
##from sklearn.manifold import TSNE
##tsne = TSNE(n_components=3, random_state=0,verbose=1, perplexity=30, n_iter=1000, n_iter_without_progress=20)
##tsne_obj= tsne.fit_transform(X)
##tsne_df = pd.DataFrame({'First dimension':tsne_obj[:,0],
##                        'Second dimension':tsne_obj[:,1],
##                        'Third dimension' : tsne_obj[:,2],
##                        'FLAG':y})
##sns.scatterplot(x="X", y="Y",
##              hue="FLAG",
##              palette=['blue','orange'],
##              legend='full',
##              data=tsne_df);

##fig1 = plt.figure(figsize=(10, 6))
##sns.scatterplot(
##        x="First dimension", y="Second dimension",
##        hue="FLAG",
##        palette=['blue','red'],
##        data=tsne_df,
##        legend="full",
##        alpha=0.2
##    )
##
##fig1.show()





##
##target_count = trainDF.label.value_counts()
##print('Class 0:', target_count[0])
##print('Class 1:', target_count[1])
##print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
##target_count.plot(kind='bar', title='Count (target)');
##
##def plot_2d_space(X, y, label='Classes'):   
##    colors = ['#1F77B4', '#FF7F0E']
##    markers = ['o', 's']
##    for l, c, m in zip(np.unique(y), colors, markers):
##        pt.scatter(
##            X[y==l, 0],
##            X[y==l, 1],
##            c=c, label=l, marker=m
##        )
##    pt.title(label)
##    pt.legend(loc='upper right')
##    pt.show()


##from sklearn.decomposition import PCA
##from collections import Counter
##from numpy import where
### define dataset
###X, y = make_classification(n_samples=10000, n_features=59, n_redundant=0,
####	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
###summarize class distribution
##counter = Counter(y)
##print(counter)
###scatter plot of examples by class label
##for label, _ in counter.items():
##	row_ix = where(y == label)[0]
##	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
##pyplot.legend()
##pyplot.show()
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE

oversample = ADASYN()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
##counter = Counter(y)
##print(counter)
### scatter plot of examples by class label
##for label, _ in counter.items():
##	row_ix = where(y == label)[0]
##	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
##pyplot.legend()
##pyplot.show()

accuracy = train_model(KNeighborsClassifier(),X , y,concate_test)
print ("KNeighborsClassifier, full features: ", accuracy)

accuracy = train_model(DecisionTreeClassifier(),X , y, concate_test)
print ("DecisionTreeClassifier, full features: ", accuracy)

accuracy = train_model(AdaBoostClassifier(),X , y, concate_test)
print ("ensembel, full features-chen: ", accuracy)

accuracy = train_model(BaggingClassifier(),X , y,concate_test)
print ("BaggingClassifier, full features: ", accuracy)

accuracy = train_model(RandomForestClassifier(),X , y, concate_test)
print ("RandomForestClassifier, full features: ", accuracy)
##
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier
####est_AB = AdaBoostClassifier()
##est_RF = RandomForestClassifier()
est_XGB = xgboost.XGBClassifier()
####est_GB = GradientBoostingClassifier()
est_BA = ExtraTreesClassifier()
est_FB=RandomForestClassifier()
##
est_ba=BaggingClassifier()
est_DT = DecisionTreeClassifier()
##est_KN = KNeighborsClassifier()
##est_ET = ExtraTreesClassifier()

est_Ensemble = VotingClassifier(estimators=[('est_XGB', est_XGB), ('est_BA', est_BA)],
                        voting='soft',
                        weights=[1,1])

##est_Ensemble2 = VotingClassifier(estimators=[('est_XGB', est_XGB), ('est_ba', est_ba)],
##                        voting='soft',
##                        weights=[1,1])
##est_Ensemble3 = VotingClassifier(estimators=[('est_BA', est_BA), ('est_FB', est_FB)],
##                        voting='soft',
##                        weights=[1,1])
######
accuracy = train_model(est_Ensemble, X, y, concate_test)
print ("\nEnsemble1, full features chen metho ", accuracy)

##accuracy = train_model(est_Ensemble2, X, y, concate_test)
##print ("\nEnsemble2, full features ", accuracy)
##accuracy = train_model(est_Ensemble3, X, y, concate_test)
##print ("\nEnsemble3, full features ", accuracy)

##accuracy = train_model(est_Ensemble2, X, y, concate_test)
##print ("\nEnsemble2, full features", accuracy)

accuracy = train_model(ExtraTreesClassifier(),X , y, concate_test)
print ("ExtraTreesClassifier, full features: ", accuracy)

accuracy = train_model(GradientBoostingClassifier(),X , y,concate_test)
print ("ensembel, full chen features: ", accuracy)

accuracy = train_model(xgboost.XGBClassifier(),X ,y, concate_test)
print ("XGBClassifier, full features chen: ", accuracy)




##from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier
##est_AB = AdaBoostClassifier()
##est_RF = RandomForestClassifier()
##est_Ensemble = VotingClassifier(estimators=[('AB', est_AB), ('RF', est_RF)],
##                        voting='soft',
##                        weights=[1, 1])
##
######concate_train=sparse.csr_matrix(concate_train)
######concate_test=sparse.csr_matrix(concate_test)
##accuracy = train_model(ensemble.RandomForestClassifier(), concate_train, train_y, concate_test)
##print ("\nRF,CharLevel all features Vectors: ", accuracy)
##print('\n')
##
##accuracy = train_model(xgboost.XGBClassifier(),concate_train, train_y,concate_test)
##print ("Xgb, CharLevel all feaures Vectors: ", accuracy)
##
#### #Linear Classifier on Character Level TF IDF Vectors
##print('\n')
##accuracy = train_model(linear_model.LogisticRegression(), concate_train, train_y, concate_test)
##print ("\nLR, CharLevel Vectors: ", accuracy)
##print('\n')
##accuracy = train_model(GaussianNB(), concate_train, train_y, concate_test)
##print ("\nNB, All: ", accuracy)
###gnb = GaussianNB() 
##print('\n')
##accuracy = train_model(est_Ensemble, concate_train, train_y, concate_test)
##print ("\nEnsemble, All features", accuracy)

####
######concate_url_hyperlink_train=sparse.csr_matrix(concate_url_hyperlink_train)
######concate_url_hyperlink_text=sparse.csr_matrix(concate_url_hyperlink_text)
##accuracy = train_model(ensemble.RandomForestClassifier(), concate_url_hyperlink_train, train_y1, concate_url_hyperlink_text)
##print ("\nRF, url+hyperlink features Vectors: ", accuracy)
####print('\n')
##accuracy = train_model(xgboost.XGBClassifier(),concate_url_hyperlink_train, train_y1,concate_url_hyperlink_text)
##print ("Xgb,  url+hyperlink feaures Vectors: ", accuracy)
##print('\n')
## #Linear Classifier on Character Level TF IDF Vectors
##accuracy = train_model(linear_model.LogisticRegression(), concate_url_hyperlink_train, train_y1, concate_url_hyperlink_text)
##print ("\nLR, Vectors: ", accuracy)
##print('\n')
##accuracy = train_model(GaussianNB(), concate_url_hyperlink_train, train_y1, concate_url_hyperlink_text)
##print ("\nNB, : ", accuracy)
##print('\n')
##accuracy = train_model(est_Ensemble, concate_url_hyperlink_train, train_y1, concate_url_hyperlink_text)
##print ("\nEnsemble,  url+hyperlink feaures", accuracy)


##
####concate_url_text_train=sparse.csr_matrix(concate_url_text_train)
####concate_url_text_test=sparse.csr_matrix(concate_url_text_test)
##accuracy = train_model(ensemble.RandomForestClassifier(), concate_url_text_train, train_y, concate_url_text_test)
##print ("\nRF,CharLevel url+text features Vectors: ", accuracy)
##print('\n')
##accuracy = train_model(xgboost.XGBClassifier(),concate_url_text_train, train_y,concate_url_text_test)
##print ("Xgb, CharLevel url+text feaures Vectors: ", accuracy)
## #Linear Classifier on Character Level TF IDF Vectors
##print('\n')
##accuracy = train_model(linear_model.LogisticRegression(), concate_url_text_train, train_y, concate_url_text_test)
##print ("\nLR, url+text feaures: ", accuracy)
##print('\n')
##accuracy = train_model(GaussianNB(), concate_url_text_train, train_y, concate_url_text_test)
##print ("\nNB, : ", accuracy)
##print('\n')
##accuracy = train_model(est_Ensemble, concate_url_text_train, train_y, concate_url_text_test)
##print ("\nEnsemble,  url+text feaures", accuracy)


##concate_hyper_text_train=sparse.csr_matrix(concate_hyper_text_train)
##concate_hyper_text_test=sparse.csr_matrix(concate_hyper_text_test)
##accuracy = train_model(ensemble.RandomForestClassifier(), concate_hyper_text_train, train_y, concate_hyper_text_test)
##print ("\nRF, hyper+text features Vectors: ", accuracy)
##print('\n')


#accuracy = train_model(xgboost.XGBClassifier(),concate_hyper_text_train, train_y,concate_hyper_text_test)
#print ("Xgb, hyper+text feaures Vectors: ", accuracy)
##print('\n')
#### #Linear Classifier on Character Level TF IDF Vectors
##accuracy = train_model(linear_model.LogisticRegression(), concate_hyper_text_train, train_y, concate_hyper_text_test)
##print ("\nLR, hyper+text features Vectors: ", accuracy)
##print('\n')
##accuracy = train_model(GaussianNB(), concate_hyper_text_train, train_y, concate_hyper_text_test)
##print ("\nNB, :hyper+text features Vectors ", accuracy)
##print('\n')
##accuracy = train_model(est_Ensemble, concate_hyper_text_train, train_y, concate_hyper_text_test)
##print ("\nEnsemble,  hyper+text features Vectors", accuracy)


##for i in concate:
##    print(len(i),"  ",i)

##arr1 = np.array([15., 13. ,71. ,67. ,21. ,14. ,74. ,57. ,74. ,57. ,56. ,61.], dtype='float32')
##
##
##arr2 = np.array([0.08739238, 0. ,0.01537484,0.,0.,0.], dtype='float32')
##
##arr = np.concatenate((arr1, arr2))
##
##print(arr)



##model = xgboost.XGBClassifier(max_depth=6,
##                        subsample=1,
##                        objective='binary:logistic',
##                        n_estimators=200,
##                        learning_rate = 0.1)
##
##eval_set = [(concate_train, train_y), (concate_test, valid_y)]
##model.fit(concate_train, train_y, early_stopping_rounds=20, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
##y_pred = model.predict(concate_test)
##predictions = [round(value) for value in y_pred]
##accuracy = accuracy_score(valid_y, predictions)
##print("Accuracy: %.2f%%" % (accuracy * 100.0))
##print("precision_score: ",metrics.precision_score(valid_y,predictions)*100)
###print("f1_score: ",metrics.f1_score(predictions,valid_y)*100)
##print("f1_score: ",metrics.f1_score(valid_y,predictions)*100)
###print("roc_auc_score: ",metrics.roc_auc_score(predictions,valid_y)*100)
##print("roc_auc_score: ",metrics.roc_auc_score(valid_y,predictions)*100)
###print("recall_score: ",metrics.recall_score(predictions,valid_y)*100)
##print("recall_score: ",metrics.recall_score(valid_y,predictions)*100)
##
##
##
##results = model.evals_result()
##epochs = len(results['validation_0']['error'])
##x_axis = range(0, epochs)
### plot log loss
##fig, ax = pyplot.subplots()
##ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
##ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
##ax.legend()
##pyplot.ylabel('Log Loss')
##pyplot.xlabel('number of iterations')
##pyplot.title('XGBoost Log Loss')
##pyplot.show()
### plot classification error
##fig, ax = pyplot.subplots()
##ax.plot(x_axis, results['validation_0']['error'], label='Train')
##ax.plot(x_axis, results['validation_1']['error'], label='Test')
##ax.legend()
##pyplot.ylabel('Classification Error')
##pyplot.xlabel('number of iterations')
##pyplot.title('XGBoost Classification Error')
##pyplot.show()











