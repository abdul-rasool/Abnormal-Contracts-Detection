import psycopg2
import requests
from requests import get
import json
from bs4 import BeautifulSoup
import re
from datetime import datetime
import numpy as np
import statistics 
import numpy as np
from statistics import mean
from decimal import Decimal
import numpy
from psycopg2.extensions import register_adapter, AsIs
import random
import time


def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int64, addapt_numpy_int64)



def timeDiffFirstLast(t1,t2):
    timeDiff = 0
    #if len(timestamp)>0:
    
##        timeDiff = "{0:.2f}".format((datetime.utcfromtimestamp(int(timestamp[-1])) - datetime.utcfromtimestamp(
##            int(timestamp[0]))).total_seconds() / 60)

    timeDiff=(datetime.utcfromtimestamp(int(t1)) - datetime.utcfromtimestamp(
                        int(t2))).total_seconds() / 60
##    print(datetime.utcfromtimestamp(int(t1)))
##    print(datetime.utcfromtimestamp(int(t2)))

          
    return timeDiff


conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur3 = conn.cursor()
cur4 = conn.cursor()
time1=time.time()
cur3.execute("select id,address,label from shuhui_fan_1")
rows1 = cur3.fetchall()
for i in rows1:
    #try:
        print(i[1])
        sql="select * from shuhui_normal_transaction where address='"+i[1]+"'"
        cur4.execute(sql)
        rows2 = cur4.fetchall()
    #for j in rows2:
        TxnSent=0
        TxnReceived=0
        Total_Txn=0
        Valueout=0
        Valuein=0
        Value_difference=0
        distinct_address=0
        distinct_sent_address=0
        distinct_recived_address=0
        Unique_TxnSent=0
        Unique_TxnReceived=0
        AP_gasUsed_in=0
        AP_gasUsed_out=0
        gas_used_in=0
        gas_used_out=0
        gasPrice_out=0
        gasPrice_in=0
        Avg_gasPrice_in=0
        Avg_gasPrice_out=0
        Avg_value_out=0
        Avg_value_in=0
        Failed_Txn_in=0
        Failed_Txn_out=0
        Total_Failed_Txn=0
        Success_Txn_in=0
        Success_Txn_out=0
        Total_Success_Txn=0
        gasUsed_in=0
        gasUsed_out=0
        Per_TxnSent=0
        Per_TxnReceived=0
        Std_value_in=0
        Std_value_out=0
        Std_value_out=0
        Std_value_in=0
        mean_in=0
        mean_out=0
        values_in =[]
        values_out =[]
        valuesgasprice_in =[]
        valuesgasprice_out =[]
        Std_gasPrice_in=0
        Std_gasPrice_out=0
        First_Txn_Value=0
        difftimelast=0
        Avg_time=0
        mean_in_time=0
        Avg_in_time=0
        diff_out_time=0
        Avg_out_time=0
        Contract_create=0
        Txn_fee_contract_create=0
        Per_gasUsed_contract_create=0
        gasPrice_contract_create=0
        Txn_fee_in=0
        Txn_fee_out=0
        Total_Txn_fee=0
        Max_Val_Sent=0
        Min_Val_Sent=0
        Avg_Val_Sent=0
        Max_Value_Received=0
        Min_Value_Received=0
        Avg_Value_Received=0
        Gini_amt_in=0
        Gini_amt_out=0
       
        #for row1 in rows1:
        count=len(rows2)
        print(count)
        for row2 in rows2:
            if row2[1]!=row2[9] and row2[9]!="":
                TxnSent+=1
                Valueout=Valueout+int(row2[10])/1000000000000000000
                values_out.append(int(row2[10])/1000000000000000000)
                gas_used_out=gas_used_out+int(row2[11])
                gasPrice_out=gasPrice_out+int(row2[12])
                valuesgasprice_out.append(int(row2[12]))
                Min_Val_Sent=int(row2[10])/1000000000000000000
                gasUsed_out=gasUsed_out+int(row2[18])
                Txn_fee_out=Txn_fee_out+(int(row2[12])*int(row2[11]))
                if int(row2[13])!=0:
                    Failed_Txn_out+=1
                else:
                    Success_Txn_out+=1
            if row2[1]!=row2[8] and row2[8]!="": 
                TxnReceived+=1
                Valuein=Valuein+int(row2[10])/1000000000000000000
                values_in.append(int(row2[10])/1000000000000000000)
                Min_Value_Received=int(row2[10])/1000000000000000000
                gas_used_in=gas_used_in+int(row2[11])
                gasPrice_in=gasPrice_in+int(row2[12])
                valuesgasprice_in.append(int(row2[12]))
                gasUsed_in=gasUsed_in+int(row2[18])
                Txn_fee_in=Txn_fee_in+(int(row2[12])*int(row2[11]))
                if int(row2[13])!=0:
                    Failed_Txn_in+=1
                else:
                    Success_Txn_in+=1


        for j in values_out:
            if Max_Val_Sent<j:
                Max_Val_Sent=j
        for j in values_out:
            if Min_Val_Sent>j:
                Min_Val_Sent=j
        if TxnSent!=0:             
            Avg_Val_Sent=Valueout/TxnSent

##        print("Max_Val_Sent:",Max_Val_Sent)
##        print("Min_Val_Sent:",Min_Val_Sent)
##        print("Avg_Val_Sent:",Avg_Val_Sent)

        for j in values_in:
            if Max_Value_Received<j:
                Max_Value_Received=j
        for j in values_in:
            if Min_Value_Received>j:
                Min_Value_Received=j
        if TxnReceived!=0:            
            Avg_Value_Received=Valuein/TxnReceived

##        print("Max_Value_Received:",Max_Value_Received)
##        print("Min_Value_Received:",Min_Value_Received)
##        print("Avg_Value_Received:",Avg_Value_Received)


        Total_Success_Txn= Success_Txn_out+ Success_Txn_in
        Total_Failed_Txn=Failed_Txn_in+Failed_Txn_out
        Total_Txn_fee=Txn_fee_in+Txn_fee_out
##        print("Txn_fee_in:",Txn_fee_in)
##        print("Txn_fee_out:",Txn_fee_out)
##        print("Total_Txn_fee:",Total_Txn_fee)
##        print("gasUsed_in:",gasUsed_in)
##        print("gasUsed_out:",gasUsed_out)
##        print("Failed_Txn_in:",Failed_Txn_in)
##        print("Failed_Txn_out:",Failed_Txn_out)
##        print("Total_Failed_Txn:",Total_Failed_Txn)
##        print("Success_Txn_in:",Success_Txn_in)
##        print("Success_Txn_out:",Success_Txn_out)
##        print("Total_Success_Txn:",Total_Success_Txn)



        #data = np.array([7,5,4,9,12,45])

        ##print("Standard Deviation of the sample is % s "% (statistics.stdev(data)))
        ##print("Mean of the sample is % s " % (statistics.mean(data)))
        values_in=np.array(values_in, dtype='float32')
        values_out=np.array(values_out, dtype='float32')
        if TxnReceived!=0:            
            Std_value_in=np.std(values_in)
            mean_in=np.mean(values_in)
            Std_gasPrice_in=np.std(valuesgasprice_in)
##            print("Std_value_in:",Std_value_in)
##            print("mean_in:",mean_in)
##            print("Std_gasPrice_in:",Std_gasPrice_in)
        if TxnSent!=0:
            Std_value_out=np.std(values_out)
            mean_out=np.mean(values_out)
            Std_gasPrice_out=np.std(valuesgasprice_out)
##            print("Std_value_out:",Std_value_out)
##            print("mean_out:",mean_out)
##            print("Std_gasPrice_out:",Std_gasPrice_out)


        ##for i in values_in:
        ##    print(i)
        ##
        ##for i in values_out:
        ##    print(i)



        if TxnSent!=0:
            Avg_value_out=Valueout/TxnSent
            AP_gasUsed_out=(gas_used_out/TxnSent)*100
            Avg_gasPrice_out=gasPrice_out/TxnSent

        if TxnReceived!=0:
            Avg_value_in=Valuein/TxnReceived
            AP_gasUsed_in=(gas_used_in/TxnReceived)*100
            Avg_gasPrice_in=gasPrice_in/TxnReceived


##        print("AP_gasUsed_in:",AP_gasUsed_in)
##        print("AP_gasUsed_out:",AP_gasUsed_out)
##
##        print("Avg_value_out: ",Avg_value_out)
##        print("Avg_value_in: ",Avg_value_in)
##        print("gasPrice_in:",gasPrice_in)
##        print("gasPrice_out:",gasPrice_out)
##
##        print("gas_used_out:",gas_used_out)
##        print("gas_used_in:",gas_used_in)
##
##        print("Avg_gasPrice_in: ",Avg_gasPrice_in)
##        print("Avg_gasPrice_out: ",Avg_gasPrice_out)


        Total_Txn=TxnSent+TxnReceived        
##        print("TxnSent: ",TxnSent)
##        print("TxnReceived:",TxnReceived)
##        print("Total_Txn: ",Total_Txn)
##        print("Valueout:",Valueout)
##        print("Valuein:",Valuein)
##        print("Value_difference:",Valueout-Valuein)
        Per_TxnSent=TxnSent/Total_Txn
        Per_TxnReceived=TxnReceived/Total_Txn
##
##        print("Per_TxnSent:",Per_TxnSent)
##        print("Per_TxnReceived:",Per_TxnReceived)




        cur3.execute("select count(distinct(tx_from)) from shuhui_normal_transaction where address=%s and tx_from!=%s and tx_from!=''",(i[1],i[1]))
        row3 = cur3.fetchall()
        for j in row3:
            distinct_recived_address=j[0]
            
        cur3.execute("select count(distinct(tx_to)) from shuhui_normal_transaction where address=%s and tx_to!=%s and tx_to!=''",(i[1],i[1]))
        row4 = cur3.fetchall()
        for j in row4:
            distinct_sent_address=j[0]

       # print("distinct address: ",distinct_sent_address+distinct_recived_address)

        cur3.execute("select  tx_from, count(tx_from) from shuhui_normal_transaction where address=%s and tx_from!=%s and tx_from!='' group by tx_from having count(tx_from)=1",(i[1],i[1]))       
        row5=cur3.fetchall()
        for j in row5:
            Unique_TxnReceived+=1
       # print("Unique_TxnReceived: ", Unique_TxnReceived)

        cur3.execute("select tx_to, count(tx_to) from shuhui_normal_transaction where address=%s and tx_to!=%s and tx_to!='' group by tx_to having count(tx_to)=1",(i[1],i[1]))       
        row6=cur3.fetchall()
        for j in row6:
            Unique_TxnSent+=1
##        print("Unique_TxnSent: ", Unique_TxnSent)
##        print("distinct_recived_address:",distinct_recived_address)
##        print("distinct_sent_address:",distinct_sent_address)

        cur3.execute("select timeStamp from shuhui_normal_transaction where address='"+i[1]+"'order by 1 asc")
        row6=cur3.fetchall()

        times=np.array(row6, dtype='int')

        dims=times.shape[0]
        firsttime=int(times[0])
        lasttime=int(times[dims-1])
##        print("lasttime: ",lasttime)
##        print("firsttime: ",firsttime)
        difftimelast=timeDiffFirstLast(lasttime,firsttime)
##        print("difftimelast:",difftimelast)
        #Avg_time =  "{0:.2f}".format(mean(int(difftimelast)))
        Avg_time=difftimelast/dims
        #print("Avg_time: ",Avg_time)

        cur3.execute("select value from shuhui_normal_transaction where address='"+i[1]+"' order by timestamp asc")
        row7=cur3.fetchall()
        lenght=len(row7)
        count=0
        for j in row7[0]:
            #print(i)
            First_Txn_Value=int(j)/1000000000000000000
        for j in row7:
            count+=1
            if count==lenght:
                lstvalue=int(j[0])/1000000000000000000

        ##values=np.array(row7, dtype='int')
        ##dims=values.shape[0]
##        print("lstvalue:",lstvalue)
##        print("First_Txn_Value:",First_Txn_Value)


        cur3.execute("select tx_from from shuhui_normal_transaction where address='"+i[1]+"' and tx_from!='' order by timestamp asc")
        row8=cur3.fetchall()
        lenght=len(row8)
        count=0
        Last_Txn_Bit=1
        First_Txn_Bit=1
        for j in row8[0]:
            if j!=i[1]:
                First_Txn_Bit=0
        for j in row8:
            count+=1
            if count==lenght:
                if j[0]!=i[1]:
                    Last_Txn_Bit=0
                    
##        print("Last_Txn_Bit:",Last_Txn_Bit)
##        print("First_Txn_Bit:",First_Txn_Bit)


        cur3.execute("select timeStamp  from  shuhui_normal_transaction where address=%s and tx_from!='' and  tx_from!=%s order by 1",(i[1],i[1])) 
        row9=cur3.fetchall()
        if len(row9)!=0:
            times=np.array(row9, dtype='int')
            print(dims)
            dims=times.shape[0]
            firsttime_in=int(times[0])
            lasttime_in=int(times[dims-1])
           

##            print("lasttime_in: ",lasttime_in)
##            print("firsttime_in: ",firsttime_in)
            mean_in_time=timeDiffFirstLast(lasttime_in,firsttime_in)
##            print("mean_in_time:",mean_in_time)
            Avg_in_time=mean_in_time/dims
##            print("Avg_time_in: ",Avg_in_time)

        cur3.execute("select timeStamp  from  shuhui_normal_transaction where address=%s and tx_to!='' and  tx_to!=%s order by 1",(i[1],i[1])) 
        row10=cur3.fetchall()
        if len(row10)!=0:
            times=np.array(row10, dtype='int')
            dims=times.shape[0]
            print(dims)
            firsttime_out=int(times[0])
            lasttime_out=int(times[dims-1])
##            print("lasttime_out: ",lasttime_out)
##            print("firsttime_out: ",firsttime_out)
            diff_out_time=timeDiffFirstLast(lasttime_out,firsttime_out)
##            print("mean_out_time:",diff_out_time)
            Avg_out_time=diff_out_time/dims
##            print("Avg_time_out: ",Avg_out_time)


        cur3.execute("select gasprice,gasused,gas,timeStamp from shuhui_normal_transaction where address='"+i[1]+"' and tx_to=''")
        row11=cur3.fetchall()
        for j in row11:
            Contract_create=datetime.utcfromtimestamp(int(j[3]))
            Txn_fee_contract_create=(int(j[0])*int(j[1]))
            Per_gasUsed_contract_create=int(j[1])/int(j[2])
            gasPrice_contract_create=int(j[0])
##        print("Contract_create:",Contract_create)
##        print("Txn_fee_contract_create:",Txn_fee_contract_create)
##        print("Per_gasUsed_contract_create:",Per_gasUsed_contract_create)
##        print("gasPrice_contract_create:",gasPrice_contract_create)


        import numpy as np
        import matplotlib.pyplot as plt

        # ensure your arr is sorted from lowest to highest values first!

        def gini(arr):
            count = arr.size
            coefficient = 2 / count
            indexes = np.arange(1, count + 1)
            weighted_sum = (indexes * arr).sum()
            total = arr.sum()
            constant = (count + 1) / count
            return coefficient * weighted_sum / total - constant

        def lorenz(arr):
            # this divides the prefix sum by the total sum
            # this ensures all the values are between 0 and 1.0
            scaled_prefix_sum = arr.cumsum() / arr.sum()
            # this prepends the 0 value (because 0% of all people have 0% of all wealth)
            return np.insert(scaled_prefix_sum, 0, 0)


        if len(values_out)!=0:
            arr1 = np.array(values_out)
            arr1.sort()
            Gini_amt_out=gini(arr1)

        if len(values_in)!=0:
            arr2 = np.array(values_in)
            arr2.sort()
            Gini_amt_in=gini(arr2)


        # show the gini index!
        #print(gini(arr))
        #Gini_amt_out=addapt_numpy_float32(Gini_amt_out)
        #Gini_amt_in=addapt_numpy_float32(Gini_amt_in)

##        print("Gini_amt_out:",Gini_amt_out)
##        print("Gini_amt_in:",Gini_amt_in)
        

##        Gini_amt_out=round(float(Gini_amt_out),2)
##        Gini_amt_in=round(float(Gini_amt_in),2)
        #Per_gasUsed_contract_create=round(float(Per_gasUsed_contract_create),3)
        #Per_gasUsed_contract_create=round(float(Per_gasUsed_contract_create),3)

##        print("Gini_amt_out:",Gini_amt_out)
##        print("Gini_amt_in:",Gini_amt_in)


        ##lorenz_curve1 = lorenz(arr1)
        ##
        ### we need the X values to be between 0.0 to 1.0
        ##plt.plot(np.linspace(0.0, 1.0, lorenz_curve1.size), lorenz_curve1)
        ### plot the straight line perfect equality curve
        ##plt.plot([0,1], [0,1])
        ##plt.show()
        ##
        ##lorenz_curve2 = lorenz(arr2)
        ### we need the X values to be between 0.0 to 1.0
        ##plt.plot(np.linspace(0.0, 1.0, lorenz_curve2.size), lorenz_curve2)
        ### plot the straight line perfect equality curve
        ##plt.plot([0,1], [0,1])
        ##plt.show()
        
        time2=time.time()
        #print("extraction time of beahviour features is:",time2-time1)

        cur3.execute("INSERT INTO behaivour_features_shuhui_fan (id,address,Max_Val_Sent,Min_Val_Sent,Avg_Val_Sent,Max_Value_Received,"
        +"Min_Value_Received,Avg_Value_Received,Txn_fee_in,Txn_fee_out,Total_Txn_fee,gasUsed_in,gasUsed_out,Failed_Txn_in,"
        +"Failed_Txn_out,Total_Failed_Txn,Success_Txn_in,Success_Txn_out,Total_Success_Txn,Std_value_in,"
        +"mean_in,Std_gasPrice_in,AP_gasUsed_in,AP_gasUsed_out,Avg_value_out,Avg_value_in,gasPrice_in,gasPrice_out,"
        +"gas_used_out ,gas_used_in ,Avg_gasPrice_in   ,Avg_gasPrice_out   ,  TxnSent ,TxnReceived ,"
        +"Total_Txn  ,Valueout  ,Valuein  ,Value_difference   ,Per_TxnSent  ,Per_TxnReceived  ,distinct_address   ,"
        +"Unique_TxnReceived  ,"
        +"Unique_TxnSent  ,distinct_recived_address ,distinct_sent_address ,"
        +"difftimelast  ,Avg_time  ,lstvalue  ,First_Txn_Value  ,Last_Txn_Bit ,First_Txn_Bit ,mean_in_time  ,"
        +"Avg_time_in  ,Txn_fee_contract_create ,Per_gasUsed_contract_create  , gasPrice_contract_create, "
        +"Gini_amt_out,Gini_amt_in,label )"
        +"VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    +"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    +"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(i[0],i[1],Max_Val_Sent,Min_Val_Sent,Avg_Val_Sent,Max_Value_Received,
        Min_Value_Received,Avg_Value_Received,Txn_fee_in,Txn_fee_out,Total_Txn_fee,gasUsed_in,gasUsed_out,Failed_Txn_in,
        Failed_Txn_out,Total_Failed_Txn,Success_Txn_in,Success_Txn_out,Total_Success_Txn,Std_value_in,
        mean_in,Std_gasPrice_in,AP_gasUsed_in,AP_gasUsed_out,Avg_value_out,Avg_value_in,gasPrice_in,gasPrice_out,
        gas_used_out ,gas_used_in ,Avg_gasPrice_in   ,Avg_gasPrice_out,  TxnSent, TxnReceived ,
        Total_Txn  ,Valueout  ,Valuein  ,Value_difference,Per_TxnSent  ,Per_TxnReceived  ,distinct_address   ,
        Unique_TxnReceived, Unique_TxnSent,distinct_recived_address ,distinct_sent_address ,
        difftimelast  ,Avg_time  ,lstvalue  ,First_Txn_Value  ,Last_Txn_Bit ,First_Txn_Bit ,mean_in_time,
        Avg_in_time  ,Txn_fee_contract_create ,Per_gasUsed_contract_create  , gasPrice_contract_create,Gini_amt_in
        ,Gini_amt_out, i[2]))
        conn.commit()

##    except Exception:
##        print("not fount")
##        print(i[0])
##        cur3.execute("insert into contracts_id_without_behavour values(%s)",(i[0],))
##        conn.commit()
##




        





















    





        
    
    
