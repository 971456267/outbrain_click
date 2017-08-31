import pandas as pd
import numpy as np
import gc
def encode_feature(values):
    uniq = values.unique()
    mapping = dict(zip(uniq,range(1,len(uniq) + 1)))
    return values.map(mapping)

def test(file,max_length):
    res = []
    first_line = 1
    with open(file) as fin:
        for line in fin:
            if first_line == 1:
               first_line = 0
               continue
            line = line.replace("\"","",2)
            line = line.strip("\n").split(',')
            line = [int(x) for x in line]
            if len(line) < max_length:
                line = line + [0] * (max_length - len(line))
            if len(line) > max_length:
                line = line[0:max_length]
            res.append(line)
    res = np.array(res)
    print res
    print res.shape
def process_document(read_file,write_file,n,name):
    doc_id = {}
    count = 0
    max_length = 0
    for c,line in enumerate(open(read_file)):
        line = line.strip('\r\n').split(',')
	if c == 0 or len(line) != 3:
           continue
        #if c % 10000 == 0:
        #    print c
        #print line[1]
        if line[0] not in doc_id:
            doc_id[line[0]] = [(int(line[1]))]
        else:
            doc_id[line[0]].append(int(line[1]))
    for key ,value in doc_id.items():
       if max_length < len(value):
           max_length = len(value) 
    print(name,max_length)
    df_doc_id = pd.DataFrame()
    df_doc_id['document_id'] = doc_id.keys()
    ss = doc_id.values()
    item = []
    for subss in ss:
        s = ",".join(str(i) for i in subss)
        item.append(s)
    
    df_doc_id[name] = np.array(item)

    df_doc_id.to_csv(write_file,index=False)

if __name__ == '__main__':
    #test('./processed_data/documents_topics_mergei_temp.csv',39)
    process_document('../processed_data/documents_topics.csv','../processed_data/documents_topics_merge.csv',300,name = "topic_id")
    process_document('../processed_data/documents_entities.csv','../processed_data/documents_entities_merge.csv',1326009,name = "entity_id")
    process_document('../processed_data/documents_categories.csv','../processed_data/documents_categories_merge.csv',97,name = "category_id")
    #test = pd.read_csv("./processed_data/documents_entities_merge.csv")
    #id = test["entity_id"].values
    #print id
    #print type(id)
    #print type(id[0])
    #print id[0]
    #print id[1]
