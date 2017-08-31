import pandas as pd
import numpy

def get_ad_ctr():
    reg = 10
    train = pd.read_csv("../../data/clicks_train.csv")
    click = train[train['clicked'] == 1].ad_id.value_counts()
    pd_click = pd.DataFrame({'ad_id':click.index,'click_num':click.values})
    show = train.ad_id.value_counts()
    pd_show = pd.DataFrame({'ad_id':show.index,'show_num':show.values})

    print pd_click.head(10)
    print pd_show.head(10)
    train = pd.merge(train,pd_click,on = 'ad_id',how = 'left')
    train = pd.merge(train,pd_show, on = 'ad_id',how = 'left')
    train ['ctr'] = train['click_num']/(train['show_num'] + reg * 1.0) 
    del train['click_num'],train['show_num']
    train.to_csv("../processed_data/clicks_train.csv",index = False)

if __name__ == '__main__':
    get_ad_ctr()
     
