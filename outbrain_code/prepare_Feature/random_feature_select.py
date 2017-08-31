import numpy as np
import pandas as pd
import random
import gc
"""
train = pd.read_csv("./processed_data/train.csv")
train.to_csv("./processed_data/train_1.csv",index = False)
valid = pd.read_csv("./processed_data/valid.csv")
valid.to_csv("./processed_data/valid_1.csv",index = False)
"""
#df = pd.read_csv("./processed_data/new_feature_train_ad_doc_uid.csv")
#df = pd.read_csv("./processed_data/drop_new_feature_train_ad_doc_uid.csv")
df = pd.read_csv("../processed_data/more_feature.csv")
display_id = list(set(df['display_id'].values))
valid_display_id = random.sample(display_id,len(display_id) / 10)
train_display_id = list(set(display_id) - set(valid_display_id))

valid_display = pd.DataFrame()
valid_display['display_id'] = valid_display_id
valid = pd.merge(valid_display,df,on = ['display_id'],how = 'left')
print('valid',valid.shape)
valid.to_csv("../processed_data/more_valid.csv",index = False)
del valid
gc.collect()

#train_display_id = random.sample(train_display_id, len(train_display_id) / 100)
train_display = pd.DataFrame()
train_display['display_id'] = train_display_id
df = pd.merge(train_display,df,on = ['display_id'],how = 'left')
df = df.sample(frac = 1)
df_pos = df[df['clicked'] == 1]
df_neg = df[df['clicked'] == 0]
del df
gc.collect()

sample_pos = df_pos.sample(frac = 1)
sample_neg = df_neg.sample(frac = 0.25)
sample_train = sample_pos.append(sample_neg,ignore_index=True)
sample_train = sample_train.sample(frac = 1) 
sample_train.to_csv("../processed_data/more_train.csv",index = False)
del sample_train,sample_pos,sample_neg
gc.collect()

"""
df = df.sample(frac = 1)
sample_valid = df.tail(1000000)
sample_valid.to_csv("./processed_data/sample_valid.csv")

df = df.iloc[:-1000000]
#df.to_csv("./processed_data/random_feature_train_ad_doc_uid.csv")



#df = pd.read_csv("./processed_data/random_feature_train_ad_doc_uid.csv")
df_pos = df[df['clicked'] == 1]
df_neg = df[df['clicked'] == 0]
del df
gc.collect()

sample_pos = df_pos.sample(frac = 0.04)
sample_neg = df_neg.sample(frac = 0.01)
sample_train = sample_pos.append(sample_neg,ignore_index=True)
sample_train = sample_train.sample(frac = 1) 
sample_train.to_csv("./processed_data/sample_train.csv")
del sample_train,sample_pos,sample_neg
gc.collect()
"""
