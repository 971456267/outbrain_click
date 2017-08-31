import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np 
import gc
reg = 10 # trying anokas idea of regularization
eval = True

train = pd.read_csv("../data/clicks_train.csv")

#split train_data into train and validation
if eval:
	ids = train.display_id.unique()
	ids = np.random.choice(ids, size=len(ids)//10, replace=False)

	valid = train[train.display_id.isin(ids)]
	train = train[~train.display_id.isin(ids)]
	
	print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts()#ocurrence times of positive samples
cntall = train.ad_id.value_counts()
del train
gc.collect()
def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k]/(float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))

if eval:
	from ml_metrics import mapk
	
	y = valid[valid.clicked==1].ad_id.values
	print type(y)
        print y[0:10]
        y = [[_] for _ in y]
        print type(y)
        print y[0:10]
	p = valid.groupby('display_id').ad_id.apply(list)
	print type(p)
        print p[0:10]
        p = [sorted(x, key=get_prob, reverse=True) for x in p]
	print type(p)
        print p[0:10]
	print (mapk(y, p, k=12))
else:
	subm = pd.read_csv("../input/sample_submission.csv") 
	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
	subm.to_csv("subm_reg_1.csv", index=False)

