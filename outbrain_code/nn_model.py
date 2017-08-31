from operator import itemgetter,attrgetter
import numpy as np
import pandas as pd
import gc
import sys
import theano
theano.config.openmp = True
sys.path.append("./prepare_Feature")
from prepare_feature import read_feature,split_data,process_doc,process_uid,split_feature_into_multi_input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,Merge,Activation,Flatten,Input
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from ml_metrics import mapk
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import roc_auc_score,recall_score,precision_score,accuracy_score,classification_report
def get_max():
    """
    max_displayid = max(train_info[1].max() , valid_info[1].max())+ 1
    max_adid = max(train_info[2].max() , valid_info[2].max()) + 1 
    max_platform = max(train_info[3].max() , valid_info[3].max()) + 1
    max_hour = max(train_info[4].max() , valid_info[4].max()) + 1
    max_weekday = max(train_info[5].max() , valid_info[5].max()) + 1
    max_uid = max(train_info[6].max() , valid_info[6].max()) + 1
    max_documentid = max(train_info[7].max() , valid_info[7].max()) + 1
    max_campaignid = max(train_info[8].max() , valid_info[8].max()) + 1
    max_advertiserid = max(train_info[9].max() , valid_info[9].max()) + 1
    max_sourceidx = max(train_info[10].max() , valid_info[10].max()) + 1
    max_categoryid = max(train_info[11].max() , valid_info[11].max()) + 1
    max_entityid = max(train_info[12].max() , valid_info[12].max()) + 1
    max_topicid = max(train_info[13].max() , valid_info[13].max()) + 1
    max_doctrfids = max(train_info[14].max() , valid_info[14].max()) + 1
    max_sourceidy = max(train_info[15].max() , valid_info[15].max()) + 1
    max_docids = max(train_info[16].max() , valid_info[16].max()) + 1
    max_id_doc_id_click = max(train_info[17].amx,valid_info[17].max()) + 1
    uid_advertiser_id_click = max(train_info[18].amx,valid_info[18].max()) + 1
    uid_campagin_id_click = max(train_info[19].amx,valid_info[19].max()) + 1
    wf = open("./processed_data/max_record.txt",'w')
    wf.write("%s\t%s\n" % ('max_displayid',max_displayid))
    wf.write("%s\t%s\n" % ('max_adid',max_adid))
    wf.write("%s\t%s\n" % ('max_platform',max_platform))
    wf.write("%s\t%s\n" % ('max_hour',max_hour))
    wf.write("%s\t%s\n" % ('max_weekday',max_weekday))
    wf.write("%s\t%s\n" % ('max_uid',max_uid))
    wf.write("%s\t%s\n" % ('max_documentid',max_documentid))
    wf.write("%s\t%s\n" % ('max_campaignid',max_campaignid))
    wf.write("%s\t%s\n" % ('max_advertiserid',max_advertiserid))
    wf.write("%s\t%s\n" % ('max_sourceidx',max_sourceidx))
    wf.write("%s\t%s\n" % ('max_categoryid',max_categoryid))
    wf.write("%s\t%s\n" % ('max_entityid',max_entityid))
    wf.write("%s\t%s\n" % ('max_topicid',max_topicid))
    wf.write("%s\t%s\n" % ('max_doctrfids',max_doctrfids))
    wf.write("%s\t%s\n" % ('max_sourceidy',max_sourceidy))
    wf.write("%s\t%s\n" % ('max_docids',max_docids))
    wf.write("%s\t%s\n" % ('max_id_doc_id_click',max_id_doc_id_click))
    wf.write("%s\t%s\n" % ('uid_advertiser_id_click',uid_advertiser_id_click))
    wf.write("%s\t%s\n" % ('uid_campagin_id_click',uid_campagin_id_click))
    wf.close()
    """
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
                      model_uidview_source,model_uidview_onehour_doc,model_uid_click_doc,model_uid_click_campagin,model_uid_click_advert], mode='concat', concat_axis=1))
    
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

def eval_result(display_id,ad_id,clicked,score):
    valid = pd.DataFrame()
    valid['display_id'] = display_id.reshape(len(display_id),)
    valid['ad_id'] = ad_id.reshape(len(ad_id),)
    valid['clicked'] = clicked.reshape(len(clicked),)
    valid['score'] = score.reshape(len(score),)
    #print valid
    """
    y = valid[valid.clicked == 1].ad_id.values
    y = [[_] for _ in y]
    #print y
    """
    display_id_group = valid.groupby('display_id') 
    
    def help(arr):
        ad_click = arr[arr['clicked'] == 1].ad_id.values
        ad_sort = arr.sort_values(by = ["score"],ascending = False).ad_id.values
        return [ad_click,ad_sort]
    ans = display_id_group.apply(help)
    
    #print ans
    x = ans.apply(list).values
    y = []
    p = []
    for item in x:
        y.append(item[0])
        p.append(item[1])
    
    #print('y',y)
    #print ('p',p)
    print ('map@12',mapk(y,p,12))

if __name__ == "__main__":  
    #data = read_feature("./processed_data/sample_feature_train_ad_doc_uid_0_0_1.csv")
    #train,valid = split_data(data)
    """
    train_info = pd.read_csv("./processed_data/split_data/train_1.csv")
    train_info = split_feature_into_multi_input(train_info)
     
    max_dict = get_max()
   
    model = Sequential()
    multi_input_model(model,max_dict)

    y_train = train_info[0] #np.concatenate((train_info[0],train_info[0]), axis=1);y_train = y_train.reshape(len(y_train),2,1)
    print('the proportion of postive sample in train data', np.sum(y_train == 1) * 1.0 / len(y_train)) 
    
    train_data_all = [train_info[3],train_info[4],train_info[5],\
                      train_info[7],train_info[8],train_info[9],train_info[10],
                      train_info[11],train_info[12],train_info[13],train_info[14],\
                      train_info[15],train_info[16],train_info[17],train_info[18],train_info[19]]
    del train_info
    gc.collect()  
    model.fit(train_data_all,y_train,batch_size = 1024,epochs = 10)
    model.save('./save_model/model.h5')
    del train_data_all ,y_train
    gc.collect()
    """
    max_dict = get_max()
    model = Sequential()
    multi_input_model(model,max_dict)
    model.load_weights('./save_model/model.h5')
    #model = load_model("./save_model/model.h5")
    valid_info = pd.read_csv("./processed_data/split_data/valid_1.csv")
    valid_info = split_feature_into_multi_input(valid_info) 
    y_valid = valid_info[0] #np.concatenate((valid_info[0],valid_info[0]),axis=1);y_valid = y_valid.reshape(len(y_valid),2,1)
    print('the proportion of postive sample in valid data', np.sum(y_valid == 1) * 1.0 / len(y_valid)) 
    valid_data_all = [valid_info[3],valid_info[4],valid_info[5],\
                      valid_info[7],valid_info[8],valid_info[9],valid_info[10],
                      valid_info[11],valid_info[12],valid_info[13],valid_info[14],
                      valid_info[15],valid_info[16],valid_info[17],valid_info[18],valid_info[19]]
    
    score = model.evaluate(valid_data_all,y_valid,batch_size = 1024)
    print score 
    preds = model.predict(valid_data_all)
    label_pred = model.predict_classes(valid_data_all)
    #print("acc:",accuracy_score(y_valid,label_pred))
    #print ("recall:",recall_score(y_valid,label_pred))
    #print("precision:",precision_score(y_valid,label_pred))
    print("roc:",roc_auc_score(y_valid,preds))
    print("metric")
    target_names = ['show','click']
    print(classification_report(y_valid,label_pred,target_names = target_names))
    #print "valid data map@12"
    #eval_result(valid_info[1],valid_info[2],y_valid,preds)
    del valid_data_all,valid_info
    gc.collect()
    #print preds
