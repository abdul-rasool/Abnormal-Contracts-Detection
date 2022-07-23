


import binascii
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

#from xgboost import XGBClassifier

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
import time


conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur = conn.cursor()
cur.execute("select id,substring(code,3,length(code)) from shuhui_fan_1 where code not like '0x%'")
#cur.execute("select id,substring(code,3,length(code)) from shuhui_fan_1 where code")

rows1 = cur.fetchall()
#for i in rows1:
 #   print (str(i[0]))

from pyevmasm import instruction_tables, disassemble_hex, disassemble_all, assemble_hex
instruction_table = instruction_tables['byzantium']
instruction_table[20]
instruction_table['EQ']
for i in rows1:
    try:
    #print (i[0])
    #print(len(i[0]))
        t0=time.time()
        if int(len(i[1])%2)==0:
            instrs = list(disassemble_all(binascii.unhexlify(str(i[1]))))
            instrs.insert(1, instruction_table['JUMPI'])
            a = assemble_hex(instrs)
            a1=disassemble_hex(a)
        #trimmedString = str(a1);
        #trimmedString2=trimmedString.replace("\n", " ");
        #print(trimmedString2+"\n")
            cur.execute("update shuhui_fan_1 set opcode=%s where id=%s",(a1,i[0]))
            conn.commit()
            t1=time.time()
            t2=t1-t0
            print(" the disassambling time is: ",t2) 

        else:
            print("fuck"+"\n")
            new=""
            new=i[1]+'0'
            instrs = list(disassemble_all(binascii.unhexlify(str(new))))
            instrs.insert(1, instruction_table['JUMPI'])
            a = assemble_hex(instrs)
            a1=disassemble_hex(a)
        #trimmedString = str(a1);
        #trimmedString2=trimmedString.replace("\n", " ");
            cur.execute("update shuhui_fan_1 set opcode=%s where id=%s",(a1,i[0]))  
        #print(trimmedString2+"\n")
        #print(disassemble_hex(a))
            conn.commit()
    except Exception:
            cur.execute("update shuhui_fan_1 set opcode=%s where id=%s",("error",i[0]))
            conn.commit()


        
        
#assemble_hex('PUSH1 0x40\nMSTORE\n')










