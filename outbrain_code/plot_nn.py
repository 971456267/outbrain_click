import numpy as np
import pandas as pd
import gc
from prepare_feature import read_feature,split_data,process_doc,process_uid,split_feature_into_multi_input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,Merge,Activation,Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from ml_metrics import mapk
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
def get_max():
    max_dict = {}
    max_dict['max_displayid'] = 172668 
    max_dict['max_adid'] = 573098
    max_dict['max_platform'] = 4
    max_dict['max_hour'] = 23 + 1
    max_dict['max_weekday'] = 14 + 1
    max_dict['max_uid'] = 166069
    max_dict['max_documentid'] = 2999334+1
    max_dict['max_campaignid'] = 35554+1
    max_dict['max_advertiserid'] = 4532+1
    max_dict['max_sourceidx'] = 14404+1
    max_dict['max_categoryid'] = 97+1
    max_dict['max_entityid'] = 1326009+1
    max_dict['max_topicid'] = 300 + 1
    max_dict['max_doctrfids'] = 2999334+1 
    max_dict['max_sourceidy'] = 14404+1
    max_dict['max_docids'] = 2999334+1
    max_dict['max_id_doc_id_click'] = 2999334+1
    max_dict['uid_advertiser_id_click'] = 35554+1
    max_dict['uid_campagin_id_click'] = 4532
    return max_dict

   


def sub_input_model(max_dict,embedding_size,max_name,input_shape_dim1,name):
    print "sub_input_model"
    sub_model = Sequential()
    sub_model.add(Embedding(input_dim = max_dict[max_name], output_dim=embedding_size,input_shape=(input_shape_dim1,),name = name)) #input_shape=(1,)
    sub_model.add(Flatten(name = 'Flatten' + name))
    return sub_model

def sub_input_model(max_dict,embedding_size,max_name,input_shape_dim1,name):
    print "sub_input_model"
    sub_model = Sequential()
    sub_model.add(Embedding(input_dim = max_dict[max_name], output_dim=embedding_size,input_shape=(input_shape_dim1,),name = name)) #input_shape=(1,)
    sub_model.add(Flatten(name = 'Flatten' + name))
    return sub_model

def multi_input_model(model,max_dict):
    print "multi_input_model"
 
    #model_displayid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_displayid',input_shape_dim1 = 1,name='displayid')
    
    #model_adid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_adid',input_shape_dim1 = 1,name='adid')
    
    model_platform = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_platform',input_shape_dim1 = 1,name = 'platform')
    
    model_hour = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_hour',input_shape_dim1 = 1,name = 'hour')
    
    model_weekday = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_weekday',input_shape_dim1 = 1 ,name = 'weekday')
    
    #model_uid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_uid',input_shape_dim1 = 1, name = 'uid')
    
    model_documentid = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_documentid',input_shape_dim1 = 1,name = 'documentid')
    
    model_campaignid = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_campaignid',input_shape_dim1 = 1,name = 'campaignid')
    
    model_advertiserid = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_advertiserid',input_shape_dim1 = 1, name = 'advertiserid')
 
    model_sourceidx = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_sourceidx',input_shape_dim1 = 1,name = 'sourceid')
   
    model_categoryid = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_categoryid',input_shape_dim1 = 2,name = 'categoryid')
    
    model_entityid = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_entityid',input_shape_dim1 = 10,name = 'entityid')
    
    model_topicid = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_topicid',input_shape_dim1 = 39,name = 'topicid') 
    
    model_uidview_doc = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_doctrfids',input_shape_dim1 = 306,name = 'uidview_doc')
    
    model_uidview_source = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_sourceidy',input_shape_dim1 = 160,name = 'uidview_source')
    
    model_uidview_onehour_doc = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_docids',input_shape_dim1 = 123,name = 'uidview_onehour_doc')
    
    model_uid_click_doc = sub_input_model(max_dict,embedding_size = 2,max_name = 'max_id_doc_id_click',input_shape_dim1 = 37,name = 'uid_click_doc')
    
    model_uid_click_campagin = sub_input_model(max_dict,embedding_size = 2,max_name = 'uid_campagin_id_click',input_shape_dim1 = 37,name = 'uid_click_campagin')
    
    model_uid_click_advert = sub_input_model(max_dict,embedding_size = 2,max_name = 'uid_advertiser_id_click',input_shape_dim1 = 37,name = 'uid_click_advert')
    print model_uid_click_advert.output_shape
    ctr = Sequential()
    ctr.add(Dense(1,input_shape=(1,),name ='ctr'))
    print ctr.output_shape
    model.add(Merge([model_platform,model_hour,model_weekday,model_documentid,model_campaignid,\
                     model_advertiserid,model_sourceidx,model_categoryid,model_entityid,model_topicid,model_uidview_doc,\
                      model_uidview_source,model_uidview_onehour_doc,model_uid_click_doc,model_uid_click_campagin,model_uid_click_advert,ctr], mode='concat', concat_axis=1))
    
    print('the model\'s input shape ', model.input_shape)
    print ('the mode\'s output shape ', model.output_shape)
    
    model.add(Dense(1000,activation = 'relu'))
    print model.output_shape
    model.add(Dense(500,activation = 'relu'))
    print model.output_shape
    model.add(Dense(1, activation='sigmoid'))
    print('the final model\'s shape', model.output_shape)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


if __name__ == "__main__":  
    max_dict = get_max() 
    model = Sequential()
    multi_input_model(model,max_dict)

    
