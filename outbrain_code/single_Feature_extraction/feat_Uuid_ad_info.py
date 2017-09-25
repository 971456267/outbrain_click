#thanks to https://github.com/alno/kaggle-outbrain-click-prediction
#coding:utf-8
import numpy as np
import pandas as pd
#  the tables occupy less memory,use pandas directly
def get_uuid_ad_table():
    click_train = pd.read_csv("../../data/clicks_train.csv")
    event = pd.read_csv("../processed_data/events.csv")
    uuid_ad = pd.merge(click_train,event,on  = 'display_id',how = 'left')
    uuid_ad = uuid_ad[uuid_ad['clicked']==1]
    promoted_content = pd.read_csv("../data/promoted_content.csv")
    uuid_ad = pd.merge(uuid_ad,promoted_content,on = 'ad_id',how = 'left')
    uuid_ad.to_csv("../processed_data/uuid_clicked_ad.csv")
    print uuid_ad.head(10)

def write(data,write_file,name):
    df = pd.DataFrame()
    df['uid'] = data.keys()
    ss = data.values()
    item = []
    max_length = 0
    for subss in ss:
        max_length = max(max_length,len(subss))
        s = ",".join(str(i) for i in subss)
        item.append(s)
    df[name] = np.array(item) 
    print max_length
    df.to_csv(write_file,index = False)

def get_uuid_ad_info():
    uid_doc = {}
    uid_campaign  = {}
    uid_advertiser= {}
    count = 0
    max_lenght = 0
    for c,line in enumerate(open('../processed_data/uuid_clicked_ad.csv')):
        line = line.strip('\n').split(',')
        #print c
        if c == 0:
            continue
        if line[14] not in uid_doc:
             uid_doc[line[14]] = [int(line[15])]
        else:
             uid_doc[line[14]].append(line[15])
        
        if line[14] not in uid_campaign:
             uid_campaign[line[14]] = [int(line[16])]
        else:
             uid_campaign[line[14]].append(line[16])
        if line[14] not in uid_advertiser:
             uid_advertiser[line[14]] = [int(line[17])]
        else:
             uid_advertiser[line[14]].append(line[17])
    print len(uid_doc)
    print len(uid_campaign)
    print len(uid_advertiser)
    write(uid_doc,'../processed_data/uid_doc_click.csv','uid_doc_id_click')
    write(uid_campaign,'../processed_data/uid_campagin_click.csv','uid_campagin_id_click')
    write(uid_advertiser,'../processed_data/uid_advertiser_click.csv','uid_advertiser_id_click')
             
if __name__ == '__main__':
    #get_uuid_ad_table()
    get_uuid_ad_info()
