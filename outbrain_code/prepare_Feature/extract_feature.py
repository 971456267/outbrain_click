import numpy as np
import os
import pandas as pd
import gc
#merge the single feature by display_id and ad_id

train = pd.read_csv("../processed_data/clicks_train.csv")
event = pd.read_csv("../processed_data/events.csv")
print('1',train.shape)
print event.shape

train_event = pd.merge(train,event,on = ['display_id'],how = 'left')
train_event.rename(columns = {"document_id":"document_id_event"},inplace = True)
del train,event
gc.collect()
print train_event.shape
print train_event.head(10)

ad = pd.read_csv("../../data/promoted_content.csv")
print('2',ad.shape)

train_ad = pd.merge(train_event,ad,on = ['ad_id'],how = 'left')
del train_event,ad
gc.collect()
print train_ad.shape


doc_meta = pd.read_csv("../../data/documents_meta.csv")
print("doc_meta's size",doc_meta.shape)
doc_topic = pd.read_csv("../processed_data/documents_categories_merge.csv")
print("doc_topic's size",doc_topic.shape)
doc_meta_topic = pd.merge(doc_meta,doc_topic,on = 'document_id',how = 'left') 
print("doc_meta_topic's size",doc_meta_topic.shape)
del doc_meta,doc_topic
gc.collect()

doc_entity = pd.read_csv("../processed_data/documents_entities_merge.csv")
print("doc_entity's size",doc_entity.shape)
doc_meta_topic_entity = pd.merge(doc_meta_topic,doc_entity,on = 'document_id',how = 'left') 
print("doc_meta_topic_entity's size",doc_meta_topic_entity.shape)
del doc_entity,doc_meta_topic
gc.collect()

doc_category = pd.read_csv("../processed_data/documents_topics_merge.csv")
print("doc_category's size",doc_category.shape)
doc_meta_topic_entity_category = pd.merge(doc_meta_topic_entity,doc_category, on = 'document_id',how = 'left')
print("doc_meta_topic_entity_category's size", doc_meta_topic_entity_category.shape)
del doc_category,doc_meta_topic_entity
gc.collect()

print('3',doc_meta_topic_entity_category.shape)
train_ad_doc = pd.merge(train_ad,doc_meta_topic_entity_category,on ='document_id', how = 'left')
print train_ad_doc.shape
del doc_meta_topic_entity_category ,train_ad
gc.collect()

#add the uuid info
viewed_doc = pd.read_csv("../processed_data/viewed_doc_trf_source.csv")
viewed_doc.rename(columns = {"uuid":"uid"},inplace = True)
print("viewed_doc's size",viewed_doc.shape)
train_ad_doc_vieweddoc = pd.merge(train_ad_doc,viewed_doc,on = 'uid',how = 'left')
print("train_ad_doc_vieweddoc's size", train_ad_doc_vieweddoc.shape)
del viewed_doc,train_ad_doc
gc.collect()

viewed_source = pd.read_csv("../processed_data/viewed_doc_sources.csv")
viewed_source.rename(columns = {"uuid":"uid"},inplace = True)
print("viewed_source's size",viewed_source.shape)
train_ad_doc_vieweddoc_viewedsource = pd.merge(train_ad_doc_vieweddoc,viewed_source,on = 'uid',how = 'left')
print("train_ad_doc_vieweddoc_viewedsource's size", train_ad_doc_vieweddoc_viewedsource.shape)
del viewed_source,train_ad_doc_vieweddoc
gc.collect()

viewed_doc_onehour =  pd.read_csv("../processed_data/drop_duplicates_viewed_docs_one_hour_after.csv")
viewed_doc_onehour.rename(columns = {"uuid":"uid"},inplace = True)
print("viewed_doc_onehour's size",viewed_doc_onehour.shape)
train_ad_doc_vieweddoc_viewedsource_vieweddoconehour = pd.merge(train_ad_doc_vieweddoc_viewedsource,viewed_doc_onehour,on = 'uid',how = 'left')
print("train_ad_doc_viewedoc_viewedsource_vieweddoconehour's size", train_ad_doc_vieweddoc_viewedsource_vieweddoconehour.shape)
del viewed_doc_onehour,train_ad_doc_vieweddoc_viewedsource
gc.collect()
#train_ad_doc_vieweddoc_viewedsource_vieweddoconehour.to_csv('./processed_data/drop_new_feature_train_ad_doc_uid.csv',index=False)

#add the uuid->adid info 
uid_advertiser = pd.read_csv("../processed_data/uid_advertiser_click.csv")
uid_campagin = pd.read_csv("../processed_data/uid_campagin_click.csv")
uid_doc = pd.read_csv("../processed_data/uid_doc_click.csv")
uid = pd.merge(uid_advertiser,uid_campagin,on = "uid",how = 'left')
uid = pd.merge(uid,uid_doc,on = "uid",how = 'left')
del uid_advertiser,uid_campagin,uid_doc
gc.collect()
print uid.shape

feature = pd.merge(train_ad_doc_vieweddoc_viewedsource_vieweddoconehour,uid,on='uid', how = 'left')

del train_ad_doc_vieweddoc_viewedsource_vieweddoconehour
gc.collect()

feature.to_csv("../processed_data/more_feature.csv",index=False)
print feature.shape
del feature
gc.collect()

