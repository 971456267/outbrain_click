import numpy as np
import pandas as pd
import gc
def read_feature(file):
    print "read feature"
    feature_ad_doc_uid = pd.read_csv(file)
    return feature_ad_doc_uid

def split_data(train):
    print "split data"
    ids = train.display_id.unique() 
    ids = np.random.choice(ids, size=len(ids)//10, replace=False)
    valid = train[train.display_id.isin(ids)]
    train = train[~train.display_id.isin(ids)]	
    print (train.shape, valid.shape)
    return train,valid

def process_doc(data,max_length):
    print "process_doc"
    res = []
    for line in data:
        try: 
            line = line.replace("\"","",2)
            line = line.strip("\n").split(',')
            line = [int(x) for x in line]
            if len(line) < max_length:
                line = line + [0] * (max_length - len(line))
            if len(line) > max_length:
                line = line[0:max_length]
        except:
            line = [0] * max_length
        res.append(line)
    res = np.array(res)
    return res        
def process_uid(data,max_length):
    print "process_uid"
    res = []
    for line in data:
        try:
            line = line.split(" ")
            line = [int(x) for x in line]
            if len(line) < max_length:
                line = line + [0] * (max_length - len(line))
            if len(line) > max_length:  
               line = line[0:max_length]   
        except:
            line = [0] * max_length 
        res.append(line)
    res = np.array(res)  
    return res
    
def split_feature_into_multi_input(data):
    print "split_feature_into_multi_input"
    data = data.fillna(0)    
    display_id = data['display_id'].values
    display_id = display_id.astype('int64')
    display_id = display_id.reshape(len(display_id),1)
    display_id[np.where( display_id < 0)] = 0
    
    ad_id = data['ad_id'].values
    ad_id = ad_id.astype('int64')
    ad_id = ad_id.reshape(len(ad_id),1)
    ad_id[np.where(ad_id < 0)] = 0
    
    clicked = data['clicked'].values
    clicked = clicked.reshape(len(clicked),1)
    
    platform = data['platform'].values
    print('platform',platform.max())
    platform = platform.astype('int')
    platform = platform.reshape(len(platform),1)
    
    hour = data['hour'].values  
    print('hour',hour.max())
    hour = hour.astype('int')
    hour = hour.reshape(len(hour),1)
    
    weekday = data['weekday'].values 
    print('weekday',weekday.max())
    weekday = weekday.astype('int')
    weekday = weekday.reshape(len(weekday),1)
   
    uid = data['uid'].values
    uid = uid.astype('int64')
    uid = uid.reshape(len(uid),1)
    uid[np.where( uid < 0)] = 0
    
    document_id = data["document_id"].values
    print('document_id',document_id.max())
    document_id = document_id.astype('int64')
    document_id = document_id.reshape(len(document_id),1)
    document_id[np.where( document_id < 0)] = 0
    
    campaign_id = data["campaign_id"].values
    print('campaign_id',campaign_id.max())
    campaign_id = campaign_id.astype('int64')
    campaign_id = campaign_id.reshape(len(campaign_id),1)
    campaign_id[np.where( campaign_id < 0)] = 0
   
    advertiser_id = data['advertiser_id'].values
    print('advertiser_id',advertiser_id.max())
    advertiser_id = advertiser_id.astype('int64')
    advertiser_id = advertiser_id.reshape(len(advertiser_id),1)
    advertiser_id[np.where(advertiser_id < 0)] = 0 
    

    source_id_x = data['source_id_x'].values
    print('source_id_x',source_id_x.max())
    source_id_x = source_id_x.astype('int64') 
    source_id_x = source_id_x.reshape(len(source_id_x),1)   
    source_id_x[np.where(source_id_x < 0)] = 0

    category_id = data["category_id"].values
    print('category_id',category_id.max())
    category_id = process_doc(category_id,2)
    category_id[np.where(category_id < 0)] = 0

    entity_id = data["entity_id"].values
    print('entity_id',entity_id.max())
    entity_id = process_doc(entity_id,10)
    entity_id[np.where(entity_id < 0)] = 0 
  
    topic_id = data["topic_id"].values
    print('topic_id',topic_id.max())
    topic_id = process_doc(topic_id,39)
    topic_id[np.where(topic_id < 0)] = 0

    doc_trf_ids = data['doc_trf_ids'].values#uid viewed doc
    print('doc_trf_ids',doc_trf_ids.max())
    doc_trf_ids = process_uid(doc_trf_ids,306) 
    doc_trf_ids[np.where(doc_trf_ids < 0)] = 0

    source_id_y = data['source_id_y'].values#uid viewed source
    print('source_id_y',source_id_y.max())
    source_id_y = process_uid(source_id_y,160)    
    source_id_y[np.where(source_id_y < 0)] = 0
    
    
    doc_ids = data['doc_ids'].values#uid viewed doc one hour
    print('doc_ids',doc_ids.max())
    doc_ids = process_uid(doc_ids,123)
    doc_ids[np.where(doc_ids < 0)] = 0
   
    ctr = data['ctr'].values
    ctr = ctr.reshape(len(ctr),1)
    
    uid_doc_id_click = data['uid_doc_id_click'].values
    print('uid_doc_id_click',uid_doc_id_click.max())
    uid_doc_id_click = process_doc(uid_doc_id_click,37)
    uid_doc_id_click[np.where(uid_doc_id_click  < 0)] = 0

    uid_advertiser_id_click = data['uid_advertiser_id_click'].values
    print('uid_advertiser_id_click ',uid_advertiser_id_click .max())
    uid_advertiser_id_click = process_doc(uid_advertiser_id_click,37)
    uid_advertiser_id_click[np.where(uid_advertiser_id_click < 0)] = 0
       
    uid_campagin_id_click = data['uid_campagin_id_click'].values
    print('uid_campagin_id_click',uid_campagin_id_click.max())
    uid_campagin_id_click = process_doc(uid_campagin_id_click,37)
    uid_campagin_id_click[np.where(uid_campagin_id_click < 0)] = 0
    
    
    return [clicked,display_id,ad_id,platform,hour,weekday,uid,document_id,campaign_id,advertiser_id,source_id_x,category_id,\
             entity_id,topic_id,doc_trf_ids,source_id_y,doc_ids,uid_doc_id_click,uid_advertiser_id_click,uid_campagin_id_click,ctr]#21

if __name__ == "__main__":
    data = read_feature("../processed_data/more_feature.csv")
    split_feature_into_multi_input(data)
    #train,valid = split_data(data)
    #train_info = split_feature_into_multi_input(train)
    #valid_info = split_feature_into_multi_input(valid)
    #print train_info
    #print valid_info
    #max_displayid = max( train_info[1].max(),valid_info[1].max() )+ 1
    #max_adid = max(train_info[2].max(), valid_info[2].max() ) + 1
    #print max_displayid
    #print max_adid
