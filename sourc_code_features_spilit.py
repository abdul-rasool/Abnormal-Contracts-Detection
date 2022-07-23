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
from numpy import mean
from numpy import std
#from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
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
import time


np.random.seed(2018)


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


conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
#cur1 = conn.cursor()
#cur2 = conn.cursor()
cur3 = conn.cursor()
#cur1.execute("select * from urls_with_handcraft2")
#cur2.execute("select url,typ from bench_mark2")
cur3.execute("select training_source2.source_code,label from training_source2")
#rows1 = cur1.fetchall()
#rows2 = cur2.fetchall()
rows2 = cur3.fetchall()

train_x, valid_x, train_y, valid_y,id_test = [],[],[],[],[]


bad_chars={';','0','1','2','3','4','5',
                    '6','7','8','9','\n',':','!',"*",
                    '[',']','{','(',')',",",';','.','!','?',
                    ':',"'",'"\"','/',"\\",'|','_','@','#',
                    '$','%','^','&','*','~','`','+','"','=',
                    '<','>','(',')','[',']','{','}'}



for row in rows2:
##    #count=0;
    train_y.append(int(row[1]))
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
    train_x.append(listToString(processeddata))
    #texts2.append(row[0])
####    #print(labels)
####    #print('\n\n')
####    #print(texts)

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()

####
#trainDF['text'] = train_x
#trainDF['label'] =labels

#for i in  trainDF['label']:
 #     print(i)

cur3.execute("select id,test_source2.source_code,label from test_source2")
#rows1 = cur1.fetchall()
#rows2 = cur2.fetchall()
rows3 = cur3.fetchall()
for row in rows3:
##    #count=0;
    valid_y.append(int(row[2]))
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
    valid_x.append(listToString(processeddata))

#trainDF['text2'] = valid_x

##trainDF['new']=trainDF['text'].fillna('').astype(str).map(preprocess)

#trainDF['text']= trainDF['text'].map(preprocess)
#train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.2, random_state=0)

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
#id_test=encoder.fit_transform(id_test)
#print(valid_x.shape())

##for i in valid_x:
##    print(i)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    t0=time.time()
    classifier.fit(feature_vector_train, label)
    t1=time.time()
    print("the training time of FS4 is : ", t1-t0)
    # predict the labels on validation dataset
    t00=time.time()
    predictions = classifier.predict(feature_vector_valid)
    t11=time.time()
    print(" test time of FS4 is: ", t11-t00)

##    for i in valid_y:
##         cur3.execute("INSERT INTO valid_y2(valid_y) VALUES(%s)",str(i))
##    for i in predictions:
##         cur3.execute("INSERT INTO predictions2(predictions) VALUES(%s)",str(i))
         
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
##            cur3.execute("INSERT INTO fp_contracts(id,valid_y) VALUES(%s,%s)",(int(id_test[j]),int(i)))
##            conn.commit()
        if i==1 and predictions[j]==0:
            fn=fn+1
##            cur3.execute("INSERT INTO fn_contracts(id,valid_y) VALUES(%s,%s)",(int(id_test[j]),int(i)))
##            conn.commit()
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
    #print("accuracy: ",metrics.accuracy_score(valid_y, predictions)*100)
    
    #print(metrics.f1_score(*100)
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




##processed_traindata=train_x.map(preprocess)
##processed_testdata=valid_x.map(preprocess)
##train=" "


         
##    
##for i in train_x:
##     print(i)

#print(train)


##dictionary1 = gensim.corpora.Dictionary(processed_traindata)
##dictionary2 = gensim.corpora.Dictionary(processed_testdata)
##dictionary1.filter_extremes(no_below=15, no_above=0.5, keep_n=5000)
##dictionary2.filter_extremes(no_below=15, no_above=0.5, keep_n=5000)

##bowtrain_corpus = [dictionary1.doc2bow(doc) for doc in processed_traindata]
##bowtvalid_corpus = [dictionary2.doc2bow(doc) for doc in processed_testdata]
##
##from gensim import corpora, models
##tfidf1 = models.TfidfModel(bowtrain_corpus)
##tfidf2 = models.TfidfModel(bowtvalid_corpus)
##corpustrain_tfidf = tfidf1[bowtrain_corpus]
##corpusvalid_tfidf = tfidf2[bowtvalid_corpus]

##bowtrain_corpus = np.array(bowtrain_corpus, dtype='float32')
##bowtvalid_corpus = np.array(bowtvalid_corpus, dtype='float32')
##

##for i in corpusvalid_tfidf:
##    print(i)

##lda_model_train = gensim.models.LdaMulticore(bowtrain_corpus, num_topics=5, id2word=dictionary1, random_state=100,chunksize=100,alpha=0.01,
##                                           eta=0.9, passes=2, workers=1,iterations=2)
##lda_model_vailid =gensim.models.LdaMulticore(bowtvalid_corpus, num_topics=5, id2word=dictionary2, passes=10, workers=2,iterations=50)
##
##lda_model_tfidf_train = gensim.models.LdaMulticore(corpustrain_tfidf, num_topics=5, id2word=dictionary1, passes=2, workers=4)
##lda_model_tfidf_vailid = gensim.models.LdaMulticore(corpusvalid_tfidf, num_topics=5, id2word=dictionary2, passes=2, workers=4)

##for idx, topic in lda_model_train.print_topics(-1):
##    print('Topic: {} \nWords: {}'.format(idx, topic))

##for i in  lda_model_train.show_topics():
##    print(i[0], i[1])



## #word level tf-idf
##tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=30000 )
##xtrain_tfidf =  tfidf_vect.fit_transform(train_x).toarray()
##xvalid_tfidf =  tfidf_vect.fit_transform(valid_x).toarray()

#####word level tf-idf
##tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=30000)
##tfidf_vect.fit(trainDF['text'])
##xtrain_tfidf =  tfidf_vect.transform(train_x)
##xvalid_tfidf =  tfidf_vect.transform(valid_x)
######

#ngram word level tf-idf
##tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
##tfidf_vect_ngram.fit(train_x)
##xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x).toarray() 
##xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x).toarray()
##xtrain_tfidf_ngram = np.array(xtrain_tfidf_ngram, dtype='float32')
##xvalid_tfidf_ngram = np.array(xvalid_tfidf_ngram, dtype='float32')
##tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
##tfidf_vect_ngram.fit(trainDF['text'])
##xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
##xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
####
# characters level tf-idf
##text = np.concatenate((trainDF['text'], trainDF['text2']), axis=0)
##text = np.array(text)
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
#tfidf_vect_ngram_chars.fit(text)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.fit_transform(train_x).toarray() 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x).toarray()
print(xvalid_tfidf_ngram_chars.shape[1])

from sklearn.datasets import make_classification

trainDF = pandas.DataFrame()
trainDF['label'] =train_y
##trainDF['feature'] =x_train
X=xtrain_tfidf_ngram_chars
y=train_y

target_count = trainDF.label.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');

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
from sklearn.decomposition import PCA

from collections import Counter
from numpy import where

counter = Counter(y)
print(counter)
###scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
from imblearn.over_sampling import ADASYN
#t3=time.time()
oversample = ADASYN()
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)
#scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

accuracy1 = train_model(KNeighborsClassifier(),X , y,xvalid_tfidf_ngram_chars)
print ("KNeighborsClassifier, source code features: ", accuracy1)

accuracy2 = train_model(DecisionTreeClassifier(),X , y, xvalid_tfidf_ngram_chars)
print ("DecisionTreeClassifier, source code features: ", accuracy2)

accuracy3 = train_model(AdaBoostClassifier(),X , y, xvalid_tfidf_ngram_chars)
print ("AdaBoostClassifier, source code features: ", accuracy3)

accuracy4 = train_model(RandomForestClassifier(),X , y, xvalid_tfidf_ngram_chars)
print ("RandomForestClassifier, source code features: ", accuracy4)

accuracy5 = train_model(ExtraTreesClassifier(),X , y, xvalid_tfidf_ngram_chars)
print ("ExtraTreesClassifier, source code features: ", accuracy5)

accuracy6 = train_model(GradientBoostingClassifier(),X , y,xvalid_tfidf_ngram_chars)
print ("GradientBoostingClassifier, Count vector features: ", accuracy6)

accuracy7 = train_model(BaggingClassifier(),X , y,xvalid_tfidf_ngram_chars)
print ("BaggingClassifier, source code features: ", accuracy7)

accuracy8 = train_model(xgboost.XGBClassifier(),X , y,xvalid_tfidf_ngram_chars)
print ("XGBClassifier, source code features: ", accuracy8)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier
####est_AB = AdaBoostClassifier()
##est_RF = RandomForestClassifier()
est_XGB = xgboost.XGBClassifier()
est_GB = ExtraTreesClassifier()
est_BA = BaggingClassifier()
##
##est_DT = DecisionTreeClassifier()
##est_KN = KNeighborsClassifier()
##est_ET = ExtraTreesClassifier()

est_Ensemble = VotingClassifier(estimators=[('BG', est_BA), ('XGB', est_XGB) ],
                        voting='soft',
                        weights=[1,1])
accuracy9 = train_model(est_Ensemble,X , y,xvalid_tfidf_ngram_chars)
print ("ensemble, source code features: ", accuracy9)
est_Ensemble2 = VotingClassifier(estimators=[('est_GB', est_GB), ('XGB', est_XGB)],
                        voting='soft',
                        weights=[1,1])
accuracy10 = train_model(est_Ensemble2,X , y,xvalid_tfidf_ngram_chars)
print ("ensemble2, source code features: ", accuracy10)



##xtrain_tfidf_ngram_chars=sparse.csr_matrix(xtrain_tfidf_ngram_chars)
##xvalid_tfidf_ngram_chars=sparse.csr_matrix(xvalid_tfidf_ngram_chars)


#X_train = vectorizer_x.fit_transform(X_train).toarray()
#X_test = vectorizer_x.transform(X_test).toarray()


# load the pre-trained word-embedding vectors 
##embeddings_index = {}
##for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
##    values = line.split()
##    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer
##text = np.array(text)
##token = text.Tokenizer(num_words=75000)
##token.fit_on_texts(text)
##word_index = token.word_index
### convert text to sequence of tokens and pad them to ensure equal length vectors 
##train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=500)
##valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=500)


#character embeding
##tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
##tk.fit_on_texts(train_x)
##alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
##char_dict = {}
##for i, char in enumerate(alphabet):
##    char_dict[char] = i + 1
##    tk.word_index = char_dict.copy()
### Add 'UNK' to the vocabulary
##tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
##train_sequences = tk.texts_to_sequences(train_x)
##test_texts = tk.texts_to_sequences(valid_x)
### Padding
##train_data = pad_sequences(train_sequences, maxlen=500, padding='post')
##test_data = pad_sequences(test_texts, maxlen=500, padding='post')
### Convert to numpy array
##train_data = np.array(train_data, dtype='float32')
##test_data = np.array(test_data, dtype='float32')
##from keras.utils import to_categorical
###train_classes = to_categorical(train_y)
###test_classes = to_categorical(valid_y)
##input_size = train_data.shape[1]
##vocab_size = len(tk.word_index)
##print("\n",tk.word_index)
##print(vocab_size)
##embedding_size = 95
##num_of_classes = 2
### Embedding weights
##embedding_weights = []  # (70, 69)
##embedding_weights.append(np.zeros(vocab_size))  # (0, 69)
##for char, i in tk.word_index.items():  # from index 1 to 69
##    onehot = np.zeros(vocab_size)
##    onehot[i - 1] = 1
##    embedding_weights.append(onehot)
##embedding_weights = np.array(embedding_weights)

 #Create CBOW model
##processed_data=trainDF['text2'].map(preprocess)
##model1 = gensim.models.Word2Vec(processed_data, min_count = 10,size = 500, window = 5) 
##X = model1.wv.syn0
##
##for i in X:
##    print(i)
##
##word embeding
##text = np.concatenate((train_x, valid_x), axis=0)
##text = np.array(text)
##tokenizer = Tokenizer()
##tokenizer.fit_on_texts(text)
##sequences1 = tokenizer.texts_to_sequences(train_x)
##sequences2 = tokenizer.texts_to_sequences(valid_x)
##word_index = tokenizer.word_index
##size_of_vocabulary=len(tokenizer.word_index) + 1
###size_of_vocabulary=X.shape[0]
##print(size_of_vocabulary)
##X_seq_train = pad_sequences(sequences1, maxlen=500)
##X_seq_test = pad_sequences(sequences2, maxlen=500)
##X_seq_train=np.array(X_seq_train, dtype='float32')
##X_seq_test=np.array(X_seq_test, dtype='float32')




####
## #load the whole embedding into memory
##embeddings_index = dict()
##f = open('C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python36\\glove.42B.300d (1)\\glove.42B.300d.txt',encoding='cp437')
##for line in f:
##    values = line.split()
##    word = values[0]
##    coefs = np.asarray(values[1:], dtype='float32')
##    embeddings_index[word] = coefs
##f.close()
##print('Loaded %s word vectors.' % len(embeddings_index))
## #create a weight matrix for words in training docs
##embedding_matrix = np.zeros((size_of_vocabulary, 100))
##
##for word, i in tokenizer.word_index.items():
##    embedding_vector = embeddings_index.get(word)
##    if embedding_vector is not None:
##        embedding_matrix[i] = embedding_vector



#embedding_matrix=X
##
##
#CNN model
##input_size = X_seq_train.shape[1]
##print(input_size)
###vocab_size = len(tokenizer.word_index)
###print(vocab_size)
##embedding_size = 100
##conv_layers = [[256, 7, 3],
##               [256, 7, 3],
##               [256, 3, -1],
##               [256, 3, -1],
##               [256, 3, -1],
##               [256, 3, -1],
##               [256, 3, 3]]
##fully_connected_layers = [2028, 2048]
##num_of_classes = 2
##dropout_p = 0.5
##optimizer = 'adam'
##loss = 'sparse_categorical_crossentropy'
### Embedding layer Initialization
##embedding_layer = Embedding(size_of_vocabulary,
##                            embedding_size,
##                            input_length=input_size,
##                            trainable=True)
### Model Construction
### Input
##inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
### Embedding
##x = embedding_layer(inputs)
### Conv
##for filter_num, filter_size, pooling_size in conv_layers:
##    x = Conv1D(filter_num, filter_size)(x)
##    x = Activation('relu')(x)
##    if pooling_size != -1:
##        x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
##x = Flatten()(x)  # (None, 8704)
### Fully connected layers
##for dense_size in fully_connected_layers:
##    x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
##    x = Dropout(dropout_p)(x)
### Output Layer
##predictions = Dense(num_of_classes, activation='softmax')(x)
### Build model
##model = Model(inputs=inputs, outputs=predictions)
##model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
##model.summary()
### Shuffle
##indices = np.arange(X_seq_train.shape[0])
##np.random.shuffle(indices)
##x_train1 = X_seq_train
##y_train1 = train_y
##x_test1 = X_seq_test
##y_test1 = valid_y
### Training
##ckpt_callback = ModelCheckpoint('keras_model', 
##                                 monitor='val_accuracy', 
##                                 verbose=1, 
##                                 save_best_only=True, 
##                                 mode='auto')
##history=model.fit(x_train1, y_train1,
##          validation_data=(x_test1, y_test1),
##          batch_size=128,
##          epochs=10,
##          verbose=2,callbacks=[ckpt_callback])
##model = load_model('keras_model')
##predicted = model.predict(x_test1)
##predicted = np.argmax(predicted, axis=1)
###predicted = model.predict(x_test1)
###predicted = np.argmax(predicted, axis=1)
##print(metrics.classification_report(y_test1, predicted))
##print("\n f1_score(in %):", metrics.f1_score(y_test1, predicted)*100)
##print("model accuracy(in %):", metrics.accuracy_score(y_test1, predicted)*100)
##print("precision_score(in %):", metrics.precision_score(y_test1,predicted)*100)
##print("roc_auc_score(in %):", metrics.roc_auc_score(y_test1,predicted)*100)
##print("recall_score(in %):", metrics.recall_score(y_test1,predicted)*100)

##
###simple neural network
##model=Sequential()
###embedding layer
##model.add(Embedding(size_of_vocabulary,500,input_length=500,trainable=True)) 
###lstm layer
##model.add(LSTM(128,return_sequences=True,dropout=0.2))
###Global Maxpooling
##model.add(GlobalMaxPooling1D())
###Dense Layer
##model.add(Dense(64,activation='relu')) 
##model.add(Dense(2,activation='sigmoid')) 
###Add loss function, metrics, optimizer
##model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=["acc"])
##ckpt_callback = ModelCheckpoint('keras_model', 
##                                 monitor='val_accuracy', 
##                                 verbose=1, 
##                                 save_best_only=True, 
##                                 mode='auto')
#####Adding callbacks
####es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
####mc=ModelCheckpoint('best_m.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  
##
###Print summary of model
##print(model.summary())
##history = model.fit(np.array(X_seq_train),np.array(train_y),batch_size=128,epochs=5,validation_data=(np.array(X_seq_test),np.array(valid_y)),verbose=1,callbacks=[ckpt_callback])
##from keras.models import load_model
##model = load_model('keras_model')
###evaluation 
####_,val_acc = model.evaluate(X_seq_test,valid_y, batch_size=128)
####print(val_acc)
##predicted = model.predict(X_seq_test)
##predicted = np.argmax(predicted, axis=1)
####print(metrics.classification_report(X_seq_test, predicted))
##print("\n f1_score(in %):", metrics.f1_score(valid_y, predicted)*100)
##print("model accuracy(in %):", metrics.accuracy_score(valid_y, predicted)*100)
##print("precision_score(in %):", metrics.precision_score(valid_y,predicted)*100)
##print("roc_auc_score(in %):", metrics.roc_auc_score(valid_y,predicted)*100)
##print("\nrecall_score(in %):", metrics.recall_score(valid_y,predicted)*100)
##_,val_acc = model.evaluate(X_seq_test,valid_y, batch_size=128)
##print(val_acc)


##print('Found %s unique tokens.' % len(word_index))
##indices = np.arange(text.shape[0])
### np.random.shuffle(indices)
##text = text[indices]
##print(text.shape)
##X_seq_train = text[0:len(train_x), ]
##X_seq_test = text[len(train_x):, ]

##for i in X_seq_train:
##    print(i)

### create token-embedding mapping
##embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
##for word, i in word_index.items():
##    embedding_vector = embeddings_index.get(word)
##    if embedding_vector is not None:
##        embedding_matrix[i] = embedding_vector

##def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
##    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
##    X_train = vectorizer_x.fit_transform(X_train).toarray()
##    X_test = vectorizer_x.transform(X_test).toarray()
##    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
##    return (X_train, X_test)


##for i in corpustrain_tfidf:
##    print (i)
##
##### create a count vectorizer object 
##count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',max_features=30000)
##count_vect.fit(trainDF['text'])
###transform the training and validation data using count vectorizer object
##xtrain_count =  count_vect.transform(train_x)
##xvalid_count =  count_vect.transform(valid_x)
##for i in xtrain_tfidf:
##    print (i)

##
##
#neural network
##model=Sequential()
###embedding layer
##model.add(Embedding(size_of_vocabulary,100,input_length=500,trainable=True)) 
###lstm layer
##model.add(LSTM(128,return_sequences=True,dropout=0.2))
###Global Maxpooling
##model.add(GlobalMaxPooling1D())
###Dense Layer
##model.add(Dense(64,activation='relu')) 
##model.add(Dense(2,activation='sigmoid')) 
###Add loss function, metrics, optimizer
##model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=["acc"]) 
###Adding callbacks
##es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
##mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  
###Print summary of model
##print(model.summary())
##history = model.fit(np.array(X_seq_train),np.array(train_y),batch_size=128,epochs=10,validation_data=(np.array(X_seq_test),np.array(valid_y)),verbose=1,callbacks=[es,mc])
##from keras.models import load_model
##model = load_model('best_model.h5')
###evaluation 
##_,val_acc = model.evaluate(X_seq_test,valid_y, batch_size=128)
##print(val_acc)
##predicted = model.predict(X_seq_test)
##predicted = np.argmax(predicted, axis=1)
##print(metrics.classification_report(valid_y, predicted))
##print("\n f1_score(in %):", metrics.f1_score(valid_y, predicted)*100)
##print("model accuracy(in %):", metrics.accuracy_score(valid_y, predicted)*100)
##print("precision_score(in %):", metrics.precision_score(valid_y,predicted)*100)
##print("roc_auc_score(in %):", metrics.roc_auc_score(valid_y,predicted)*100)
##print("recall_score(in %):", metrics.recall_score(valid_y,predicted)*100)

##from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier
##est_AB = AdaBoostClassifier()
##est_RF = RandomForestClassifier()
##est_Ensemble = VotingClassifier(estimators=[('AB', est_AB), ('RF', est_RF)],
##                        voting='soft',
##                        weights=[1, 1])


##
 #Linear Classifier on Character Level TF IDF Vectors
#accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("\nLR, CharLevel Vectors: ", accuracy)
##

##accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
##print ("\n Xgb, tf-idf char Vectors: ", accuracy)

##accuracy = train_model(est_Ensemble, xtrain_tfidf_ngram_chars,train_y, xvalid_tfidf_ngram_chars)
##print ("\nEnsemble, text features", accuracy)

##accuracy = train_model(ensemble.RandomForestClassifier(), train_data, train_y, test_data)
##print ("\nRF,CharLevel Vectors: ", accuracy)
##
##accuracy = train_model(naive_bayes.MultinomialNB(), X_seq_train, train_y, X_seq_test)
##print ("\nNB, MultinomialNB accuracy: ", accuracy)

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
####    model.add(layers.Embedding(vocab_size+1, output_dim=95, weights=[embedding_weights], input_length=1014))
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
##ckpt_callback = ModelCheckpoint('keras_model', 
##                                 monitor='val_accuracy', 
##                                 verbose=1, 
##                                 save_best_only=True, 
##                                 mode='auto')
##
##model_DNN = Build_Model_DNN_Text(xtrain_tfidf_ngram_chars.shape[1], 2)
##history1=model_DNN.fit(xtrain_tfidf_ngram_chars, train_y,
##                              validation_data=(xvalid_tfidf_ngram_chars, valid_y),
##                              epochs=20,
##                              batch_size=128,
##                              verbose=2, callbacks =[ckpt_callback])
##predicted = model_DNN.predict(xvalid_tfidf_ngram_chars)
##
##
##predicted = np.argmax(predicted, axis=1)
##j=0
##for i in valid_y:
##         cur3.execute("INSERT INTO bench_valid3(id,valid_y3) VALUES(%s,%s)",(int(id_test[j]),int(i)))
##         j=j+1
##j=0         
##for i in predicted:
##         cur3.execute("INSERT INTO bench_predic3(id,prediction3) VALUES(%s,%s)",(int(id_test[j]),int(i)))
##         j=j+1
##         
##conn.commit()
##
##print("accuracy_score: ",metrics.accuracy_score(valid_y, predicted)*100)
##print("recall_score: ",metrics.recall_score(valid_y, predicted)*100)
##print("precision_score: ",metrics.precision_score(valid_y,predicted)*100)
##print("roc_auc_score: ",metrics.roc_auc_score(valid_y, predicted)*100)
##print("f1_score: ",metrics.f1_score(valid_y, predicted)*100)
##print("f1_score: ",metrics.classification_report(valid_y[:100], predicted))
###loss, accuracy = model_DNN.evaluate(train_data, train_y, verbose=False)
###loss, accuracy = model_DNN.evaluate(test_data, valid_y, verbose=False)
###plot_history(history1)
##





