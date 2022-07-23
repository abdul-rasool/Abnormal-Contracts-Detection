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
np.random.seed(2018)

# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
#from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
#X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model


conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
#cur1 = conn.cursor()
#cur2 = conn.cursor()
cur3 = conn.cursor()
#cur1.execute("select * from urls_with_handcraft2")
#cur2.execute("select url,typ from bench_mark2")
cur3.execute("select id,opcode,label from ponzi_D111 limit 10")
rows2 = cur3.fetchall()
t0=time.time()

##SUB=int(0)
##STOP=int(0)
##ADD=int(0)
##MUL=int(0)
##SDIV=int(0)
##MOD=int(0)
##SMOD=int(0)
##ADDMOD=int(0)
##MULMOD=int(0)
##EXP=int(0)
##SIGNEXTEND=int(0)
##LT=int(0)
##GT=int(0)
##SLT=int(0)
##SGT=int(0)
##EQ=int(0)
##ISZERO=int(0)
##AND=int(0)
##OR=int(0)
##XOR=int(0)
##NOT=int(0)
##BYTE=int(0)
##SHA3=int(0)
##ADDRESS=int(0)
##BALANCE=int(0)
##ORIGIN=int(0)
##CALLER=int(0)
##CALLVALUE=int(0)
##CALLDATALOAD=int(0)
##CALLDATASIZE=int(0)
##CALLDATACOPY=int(0)
##CODESIZE=int(0)
##CODECOPY=int(0)
##GASPRICE=int(0)
##EXTCODESIZE=int(0)
##EXTCODECOPY=int(0)
##BLOCKHASH=int(0)
##COINBASE=int(0)
##TIMESTAMP=int(0)
##NUMBER=int(0)
##DIFFICULTY=int(0)
##GASLIMIT=int(0)
##POP=int(0)
##MLOAD=int(0)
##MSTORE=int(0)
##MSTORE=int(0)
##SLOAD=int(0)
##SSTORE=int(0)
##JUMP=int(0)
##JUMPI=int(0)
##PC=int(0)
##MSIZE=int(0)
##GAS=int(0)
##JUMPDEST=int(0)
##PUSH=int(0)
##DUP=int(0)
##SWAP=int(0)
##LOG=int(0)
##CREATE=int(0)
##CALL=int(0)
##CALLCODE=int(0)
##RETURN=int(0)
##SUICIDE=int(0)
##int ADD=0,int MUL=0,int SUB=0, int DIV=0,int SDIV=0,int MOD=0,int SMOD=0,int ADDMOD=0,int MULMOD=0,int EXP=0
##int SIGNEXTEND=0
##int LT=0,int GT=0,SLT=0,SGT=0,EQ=0,ISZERO=0,AND=0,OR=0,XOR=0,NOT=0,BYTE=0,SHA3=0
##    ADDRESS=0,BALANCE=0,ORIGIN=0,CALLER=0,CALLVALUE=0,CALLDATALOAD=0,CALLDATASIZE=0,CALLDATACOPY=0,CODESIZE=0,CODECOPY=0
##    GASPRICE=0, EXTCODESIZE=0,EXTCODECOPY=0,BLOCKHASH=0,COINBASE=0,TIMESTAMP=0,NUMBER=0,DIFFICULTY=0,GASLIMIT=0
##    POP=0,MLOAD=0,MSTORE=0,MSTORE=0,SLOAD=0,SSTORE=0,JUMP=0,JUMPI=0,PC=0,MSIZE=0,GAS=0,JUMPDEST=0,PUSH=0,DUP=0
##    SWAP=0,LOG=0,CREATE=0,CALL=0,CALLCODE=0,RETURN=0,SUICIDE=0

for i in rows2:
    print(i[0])
    SUB=int(0)
    STOP=int(0)
    ADD=int(0)
    MUL=int(0)
    DIV=int(0)
    SDIV=int(0)
    MOD=int(0)
    SMOD=int(0)
    ADDMOD=int(0)
    MULMOD=int(0)
    EXP=int(0)
    SIGNEXTEND=int(0)
    LT=int(0)
    GT=int(0)
    SLT=int(0)
    SGT=int(0)
    EQ=int(0)
    ISZERO=int(0)
    AND1=int(0)
    OR1=int(0)
    XOR1=int(0)
    NOT1=int(0)
    BYTE=int(0)
    SHA=int(0)
    ADDRESS2=int(0)
    BALANCE=int(0)
    ORIGIN=int(0)
    CALLER=int(0)
    CALLVALUE=int(0)
    CALLDATALOAD=int(0)
    CALLDATASIZE=int(0)
    CALLDATACOPY=int(0)
    CODESIZE=int(0)
    CODECOPY=int(0)
    GASPRICE=int(0)
    EXTCODESIZE=int(0)
    EXTCODECOPY=int(0)
    BLOCKHASH=int(0)
    COINBASE=int(0)
    TIMESTAMP=int(0)
    NUMBER=int(0)
    DIFFICULTY=int(0)
    GASLIMIT=int(0)
    POP=int(0)
    MLOAD=int(0)
    MSTORE=int(0)
    SLOAD=int(0)
    SSTORE=int(0)
    JUMP=int(0)
    JUMPI=int(0)
    PC=int(0)
    MSIZE=int(0)
    GAS=int(0)
    JUMPDEST=int(0)
    PUSH=int(0)
    DUP=int(0)
    SWAP=int(0)
    LOG=int(0)
    CREATE1=int(0)
    CALL=int(0)
    CALLCODE=int(0)
    RETURN=int(0)
    SUICIDE=int(0)
    lines=""
    lines=str(i[1]).split('\n')
    for line in lines:
        opcode=""
        #print(line)
    #print("new smart \n");
        for element in line:
            if 48 <= ord(element) <= 57:    
                break
            else:
                opcode=opcode+element
                
        if opcode=="STOP":
                STOP=STOP+1
        elif opcode=="ADD":
                ADD=ADD+1
        elif opcode=="MUL":
                MUL=MUL+1
        elif opcode=="SUB":
                SUB=SUB+1
        elif opcode=="DIV":
                DIV=DIV+1
        elif opcode=="SDIV":
                SDIV=SDIV+1
        elif opcode=="MOD":
                MOD=MOD+1
        elif opcode=="SMOD":
                SMOD=SMOD+1
        elif opcode=="ADDMOD":
                ADDMOD=ADDMOD+1
        elif opcode=="MULMOD":
                MULMOD=MULMOD+1
        elif opcode=="EXP":
                EXP=EXP+1
        elif opcode=="SIGNEXTEND":
                SIGNEXTEND=SIGNEXTEND+1
        elif opcode=="LT":
                LT=LT+1
        elif opcode=="GT":
                GT=GT+1
        elif opcode=="SLT":
                SLT=SLT+1
        elif opcode=="SGT":
                SGT=SGT+1
        elif opcode=="EQ":
                EQ=EQ+1
        elif opcode=="ISZERO":
                ISZERO=ISZERO+1
        elif opcode=="AND":
                AND1=AND1+1
        elif opcode=="OR":
                OR1=OR1+1
        elif opcode=="XOR":
                XOR1=XOR1+1
        elif opcode=="NOT":
                NOT1=NOT1+1
        elif opcode=="BYTE":
                BYTE=BYTE+1
        elif opcode=="SHA":
                SHA=SHA+1
        elif opcode=="ADDRESS":
                ADDRESS2=ADDRESS2+1
        elif opcode=="BALANCE":
                BALANCE=BALANCE+1
        elif opcode=="ORIGIN":
                ORIGIN=ORIGIN+1
        elif CALLER=="CALLER":
                CALLER=CALLER+1
        elif opcode=="CALLVALUE":
                CALLVALUE=CALLVALUE+1
        elif opcode=="CALLDATALOAD":
                CALLDATALOAD=CALLDATALOAD+1
        elif opcode=="CALLDATASIZE":
                CALLDATASIZE=CALLDATASIZE+1
        elif opcode=="CALLDATACOPY":
                CALLDATACOPY=CALLDATACOPY+1
        elif opcode=="CODESIZE":
                CODESIZE=CODESIZE+1
        elif opcode=="CODECOPY":
                CODECOPY=CODECOPY+1
        elif opcode=="GASPRICE":
                GASPRICE=GASPRICE+1
        elif opcode=="EXTCODESIZE":
                EXTCODESIZE=EXTCODESIZE+1
        elif opcode=="EXTCODECOPY":
                EXTCODECOPY=EXTCODECOPY+1
        elif opcode=="BLOCKHASH":
                BLOCKHASH=BLOCKHASH+1
        elif opcode=="COINBASE":
                COINBASE=COINBASE+1
        elif opcode=="TIMESTAMP":
                TIMESTAMP=TIMESTAMP+1
        elif opcode=="NUMBER":
                NUMBER=NUMBER+1
        elif opcode=="DIFFICULTY":
                DIFFICULTY=DIFFICULTY+1
        elif opcode=="GASLIMIT":
                GASLIMIT=GASLIMIT+1
        elif opcode=="POP":
                POP=POP+1
        elif opcode=="MLOAD":
                MLOAD=MLOAD+1
        elif opcode=="MSTORE":
                MSTORE=MSTORE+1
        elif opcode=="SLOAD":
                SLOAD=SLOAD+1
        elif opcode=="SSTORE":
                SSTORE=SSTORE+1
        elif opcode=="JUMP":
                JUMP=JUMP+1
        elif opcode=="JUMPI":
                JUMPI=JUMPI+1
        elif opcode=="PC":
                PC=PC+1
        elif opcode=="MSIZE":
                MSIZE=MSIZE+1
        elif opcode=="GAS":
                GAS=GAS+1
        elif opcode=="JUMPDEST":
                JUMPDEST=JUMPDEST+1
        elif opcode=="PUSH":
                PUSH=PUSH+1
        elif opcode=="DUP":
                DUP=DUP+1
        elif opcode=="SWAP":
                SWAP=SWAP+1
        elif opcode=="LOG":
                LOG=LOG+1
        elif opcode=="CREATE":
                CREATE1=CREATE1+1
        elif opcode=="CALL":
                CALL=CALL+1
        elif opcode=="CALLCODE":
                CALLCODE=CALLCODE+1
        elif opcode=="RETURN":
                RETURN=RETURN+1
        elif opcode=="SUICIDE":
                SUICIDE=SUICIDE+1
                
                
##    print(STOP,"\n",ADD,"\n",MUL,"\n",SUB,"\n",SDIV,"\n",MOD,"\n",SMOD,"\n",ADDMOD,"\n",MULMOD,"\n",EXP)
##    print(SIGNEXTEND)
##    print(LT,"\n",GT,"\n",SLT,"\n",SGT,"\n",EQ)
##    print(ISZERO)
##    print(AND)
##    print (OR)
##    print(XOR,"\n",NOT,"\n",BYTE,"\n",SHA,"\n",ADDRESS,"\n",CALLER,"\n",CALLVALUE,"\n",CALLDATALOAD,"\n",CALLDATASIZE)
##    print(CALLDATACOPY,"\n",CODESIZE,"\n",CODECOPY,"\n",GASPRICE,"\n",EXTCODESIZE,"\n",EXTCODECOPY)
##    print(BLOCKHASH,"\n",COINBASE,"\n",TIMESTAMP)
##    print(NUMBER,"\n",DIFFICULTY,"\n",GASLIMIT,"\n",POP,"\n",MLOAD,"\n",MSTORE)
##    print(MSTORE,"\n",SLOAD,"\n",SSTORE,"\n",JUMP,"\n",JUMPI,"\n",PC)
##    print(MSIZE,"\n",GAS,"\n",JUMPDEST,"\n",PUSH,"\n",DUP,"\n",SWAP)
##    print(LOG,"\n",CREATE,"\n",CALL,"\n",CALLCODE,"\n",RETURN,"\n",SUICIDE)

    t1=time.time()
    t3=t1-t0
    print("extract opcode frequency time is: ",t3) 
    cur3.execute("update ponzi_D111 set STOP=%s,ADD=%s,MUL=%s,SUB=%s,DIV=%s,"
    +"SDIV=%s,MOD=%s,SMOD=%s,ADDMOD=%s,MULMOD=%s, EXP=%s,"
    +"SIGNEXTEND=%s,LT=%s,GT=%s,SLT=%s,SGT=%s, EQ=%s,"
    +"ISZERO=%s,AND1=%s,OR1=%s,XOR1=%s,NOT1=%s, BYTE=%s,"
        +"SHA=%s,BALANCE=%s,ORIGIN=%s,CALLVALUE=%s,"
    +"CALLDATALOAD=%s,CALLDATASIZE=%s,CALLDATACOPY=%s,CODESIZE=%s,CODECOPY=%s,GASPRICE=%s,"
    +"EXTCODESIZE=%s,EXTCODECOPY=%s,BLOCKHASH=%s,COINBASE=%s,TIMESTAMP=%s,NUMBER=%s,"
    +"DIFFICULTY=%s,GASLIMIT=%s,POP=%s,MLOAD=%s,MSTORE=%s,SLOAD=%s,"
    +"SSTORE=%s,JUMP=%s,JUMPI=%s,MSIZE=%s, GAS=%s,"
    +"JUMPDEST=%s,PUSH=%s,DUP=%s,SWAP=%s,LOG=%s,CREATE1=%s,"
    +"CALL=%s,CALLCODE=%s,RETURN=%s,ADDRESS2=%s where id=%s",(STOP,ADD,MUL,SUB,DIV,
    SDIV,MOD,SMOD,ADDMOD,MULMOD,EXP,SIGNEXTEND,LT,GT,SLT,SGT,EQ,ISZERO,AND1,OR1,XOR1,NOT1,BYTE,
    SHA,BALANCE,ORIGIN,CALLVALUE,CALLDATALOAD,CALLDATASIZE,CALLDATACOPY,CODESIZE,CODECOPY,GASPRICE,
    EXTCODESIZE,EXTCODECOPY,BLOCKHASH,COINBASE,TIMESTAMP,NUMBER,
    DIFFICULTY,GASLIMIT,POP,MLOAD,MSTORE,SLOAD,SSTORE,JUMP,JUMPI,MSIZE,GAS,
    JUMPDEST,PUSH,DUP,SWAP,LOG,CREATE1,CALL,CALLCODE,RETURN,ADDRESS2,i[0]))
    #conn.commit()











         #str1 = str1.replace(' n', ' ')
         #str1 = str1.replace(' p', ' ')





##for i in valid_x:
##    valid_x2.append(i.replace(' ',''))
##for i in valid_x2:
##    print(i)

    

##for i in train_x:
##    print(i)
####
#trainDF['text'] = texts
#trainDF['text2'] = texts2
#trainDF['label'] =labels

#for i in  trainDF['label']:
 #     print(i)


##trainDF['new']=trainDF['text'].fillna('').astype(str).map(preprocess)

#trainDF['text']= trainDF['text'].map(preprocess)
#train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.4, random_state=0)

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##encoder = preprocessing.LabelEncoder()
##train_y = encoder.fit_transform(train_y)
##valid_y = encoder.fit_transform(valid_y)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

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
        
        #print("precision_score: ",metrics.precision_score(predictions,valid_y)*100)
    print("precision_score: ",metrics.precision_score(valid_y,predictions)*100)
    #print("f1_score: ",metrics.f1_score(predictions,valid_y)*100)
    print("f1_score: ",metrics.f1_score(valid_y,predictions)*100)
    #print("roc_auc_score: ",metrics.roc_auc_score(predictions,valid_y)*100)
    print("roc_auc_score: ",metrics.roc_auc_score(valid_y,predictions)*100)
    #print("recall_score: ",metrics.recall_score(predictions,valid_y)*100)
    print("recall_score: ",metrics.recall_score(valid_y,predictions)*100)
    #print("accuracy: ",metrics.accuracy_score(predictions, valid_y)*100)
    #print("accuracy: ",metrics.accuracy_score(valid_y, predictions)*100)
    
    #print(metrics.f1_score(*100)


    return metrics.accuracy_score(predictions, valid_y)*100




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
##tfidf_vect.fit(train_x)
##xtrain_tfidf =  tfidf_vect.transform(train_x)
##xvalid_tfidf =  tfidf_vect.transform(valid_x)
####

#ngram word level tf-idf
##tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
##tfidf_vect_ngram.fit(train_x)
##xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
##xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
####
# characters level tf-idf
##tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=50000)
##tfidf_vect_ngram_chars.fit(train_x)
##xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x).toarray() 
##xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x).toarray()
##time22=time.time()
##total=time22-time11
##print("\n the total time of text proscessing is:",total)
##print(xtrain_tfidf_ngram_chars.shape[1])
##print(xvalid_tfidf_ngram_chars.shape[1])
##print(xtrain_tfidf_ngram_chars.shape[0])
##print(xvalid_tfidf_ngram_chars.shape[0])
##xtrain_tfidf_ngram_chars=sparse.csr_matrix(xtrain_tfidf_ngram_chars)
##xvalid_tfidf_ngram_chars=sparse.csr_matrix(xvalid_tfidf_ngram_chars)
##print(xtrain_tfidf_ngram_chars.shape[1])
##print(xvalid_tfidf_ngram_chars.shape[1])
##print(xtrain_tfidf_ngram_chars.shape[0])
##print(xvalid_tfidf_ngram_chars.shape[0])
##for i in xtrain_tfidf_ngram_chars:
##    print("\n",len(i)," ",i)
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
##train_data = pad_sequences(train_sequences, maxlen=1000, padding='post')
##test_data = pad_sequences(test_texts, maxlen=1000, padding='post')
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
#word embeding
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
###CNN model
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


##### create a count vectorizer object (BOW)
##count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',max_features=30000)
##count_vect.fit(train_x)
###transform the training and validation data using count vectorizer object
##xtrain_count =  count_vect.transform(train_x)
##xvalid_count =  count_vect.transform(valid_x)
##for i in xtrain_tfidf:
##    print (i)

##
##
###neural network
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





 #Linear Classifier on Character Level TF IDF Vectors
##accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
##print ("\nLR, CharLevel Vectors: ", accuracy)
##
##accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
##print ("\n Xgb, tf-idf char Vectors: ", accuracy)
##
##accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
##print ("\nRF,CharLevel Vectors: ", accuracy)
##
##accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
##print ("\nNB, MultinomialNB accuracy: ", accuracy)
##
###scores = cross_val_score(model, train_data, train_y, scoring='recall', cv=cv, n_jobs=-1)
### report performance
###print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 512 # number of nodes
    nLayers = 4 # number of  hidden layer
##    model.add(layers.Embedding(vocab_size+1, output_dim=95, weights=[embedding_weights], input_length=1014))
    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
##
ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')

##model_DNN = Build_Model_DNN_Text(xtrain_tfidf_ngram.shape[1], 2)
##t0=time.time()
##history1=model_DNN.fit(xtrain_tfidf_ngram, train_y,
##                              validation_data=(xvalid_tfidf_ngram, valid_y),
##                              epochs=20,
##                              batch_size=128,
##                              verbose=2, callbacks =[ckpt_callback])
##t1=time.time()
##t2=t1-t0
##print("\n traing time is:",t2)
##t00=time.time()
##predicted = model_DNN.predict(xvalid_tfidf_ngram)
###predictions = classifier.predict(feature_vector_valid)
##t11=time.time()
##t22=t11-t00
##print("\n testing time is:",t22)
##predicted = np.argmax(predicted, axis=1)

##for i in valid_y:
##         cur3.execute("INSERT INTO valid_y3(valid_y) VALUES(%s)",str(i))
##for i in predicted:
##         cur3.execute("INSERT INTO predictions3(predictions) VALUES(%s)",str(i))
##         
##conn.commit()

##print("accuracy_score: ",metrics.accuracy_score(valid_y, predicted)*100)
##print("recall_score: ",metrics.recall_score(valid_y, predicted)*100)
##print("precision_score: ",metrics.precision_score(valid_y,predicted)*100)
##print("roc_auc_score: ",metrics.roc_auc_score(valid_y, predicted)*100)
##print("f1_score: ",metrics.f1_score(valid_y, predicted)*100)
####
####
##j=0
##tp=0
##tn=0
##fp=0
##fn=0
##p=0
##l=0
##for i in valid_y:
##    if i==1 and predicted[j]==1:
##         tp=tp+1
##    if i==0 and predicted[j]==0:
##        tn=tn+1
##    if i==0 and predicted[j]==1:
##         fp=fp+1
##    if i==1 and predicted[j]==0:
##          fn=fn+1
##    if i==1:
##         p=p+1
##    if i==0:
##        l=l+1
##            
##    j=j+1
##
##tpr=float(tp/p)
##tnr=float(tn/l)
##fpr=float(fp/l)
##fnr=float(fn/p)
##
##print("\n","tpr=",tpr)
##print("\n","fpr=",fpr)
##print("\n","fnr=",fnr)
##print("\n","tnr=",tnr)
##print("\n","number of legit= ",l)
##print("\n","number of phishing= ",p)
#####print("f1_score: ",metrics.classification_report(valid_y[:100], predicted))
#####loss, accuracy = model_DNN.evaluate(train_data, train_y, verbose=False)
#####loss, accuracy = model_DNN.evaluate(test_data, valid_y, verbose=False)
#####plot_history(history1)
####
##
##
##
##
##
