import psycopg2
import requests
from requests import get
import json
from bs4 import BeautifulSoup
import re


conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=asd")
cur3 = conn.cursor()




#while True:
print("start")
cur3.execute("select id,address from contracts_id_withoutX")
rows2 = cur3.fetchall()
##if len(rows2)==0:
##    break;

for row in rows2:
    try:    
            #address="0xc1caec30b787711108957660a3f9306a5d967f66"
        url=""
        ##        url="https://api.etherscan.io/api?module=account&action=txlist&address="+str(row[1])+\
        ##        "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=YourApiKeyToken"
        url2="https://api.etherscan.io/api?module=account&action=txlist&address="+row[1]+\
                    "&startblock=0&endblock=99999999&sort=asc&apikey=YourApiKeyToken"
            ##
                
           
        response=requests.get(url2)
        content=response.json()
        result=content.get("result")
                #result2 = json.load(response) 

            #count=0
                #for n, transaction in enumerate(result):
        for transaction in result:
                blockNumber=transaction.get("blockNumber")
                timeStamp=transaction.get("timeStamp")
                hash=transaction.get("hash")
                nonce=transaction.get("nonce")
                blockHash=transaction.get("blockHash")
                transactionIndex=transaction.get("transactionIndex")
                From=transaction.get("from")
                to=transaction.get("to")
                value=transaction.get("value")
                gas=transaction.get("gas")
                gasPrice=transaction.get("gasPrice")
                isError=transaction.get("isError")
                txreceipt_status=transaction.get("txreceipt_status")
                input=transaction.get("input")
                contractAddress=transaction.get("contractAddress")
                cumulativeGasUsed=transaction.get("cumulativeGasUsed")
                gasUsed=transaction.get("gasUsed")
                confirmations=transaction.get("confirmations")
                    
                #count+=1
            ##        print("id:",n)
            ##        print("blockNumber:",blockNumber)
            ##        print("timeStamp:",timeStamp)
            ##        print("hash:",hash)
            ##        print("nonce:",nonce)
            ##        print("blockHash:",blockHash)
            ##        print("transactionIndex:",transactionIndex)
            ##        print("from:",From)
            ##        print("to:",to)
            ##        print("value:",value)
            ##        print("gas:",gas)
            ##        print("gasPrice:",gasPrice)
            ##        print("isError:",isError)
            ##        print("txreceipt_status:",txreceipt_status)
            ##        print("input:",input)
            ##        print("contractAddress:",contractAddress)
            ##        print("cumulativeGasUsed:",cumulativeGasUsed)
            ##        print("gasUsed:",gasUsed)
            ##        print("confirmations:",confirmations)
            ##        print("\n")

                cur3.execute("insert into Shuhui_normal_transaction values(%s,%s,%s,%s,"
                +"%s,%s,%s,%s,%s,%s,"
                +"%s,%s,%s,%s,%s,%s,%s,"
                +"%s,%s,%s)",(row[0],row[1],blockNumber,timeStamp,hash,nonce,blockHash,transactionIndex,
                                From,to,value,gas,gasPrice,isError,txreceipt_status,input,
                                contractAddress,cumulativeGasUsed,gasUsed,confirmations))
                conn.commit()
        print("found")
    except Exception:
            print("not fount")
            cur3.execute("insert into contracts_id_withoutX_2 values(%s,%s)",(row[0],row[1]))
            conn.commit()
##    cur3.execute("delete from contracts_id_withoutX2 where id in (select distinct (id) from normal_transaction)")
##    conn.commit()


        #print(row[1])
##        htmlString = get(url).text
##        html = BeautifulSoup(htmlString, 'lxml')
##        word="nonce"
##        #calculate length of the word
##        lword=len(word)
##        searchstring=str(html)
##        #text = 'Allowed Hello Hollow'
##        index = 0
##        while index < len(searchstring):
##            index = searchstring.find(word, index)
##            if index == -1:
##                break
##            print('found at', index)
    
##            index += lword # +2 because len('ll') == 2
##        #text = 'Allowed Hello Hollow'
####        for m in re.finditer(word, searchstring):
####            print('found', m.start(), m.end())
####        matches = re.finditer(word, searchstring)
####        matches_positions = [match.start() for match in matches]
####        print(matches_positions)
####        start_index=searchstring.find(word)
####        extracted_string= searchstring[start_index:start_index+lword]
####        print("Extracted word is:")
####        print(extracted_string)
        
##        cleantext = BeautifulSoup(htmlString, "lxml").get_text()
##        print(htmlString.find("timeStamp"))

        

#print(count)
