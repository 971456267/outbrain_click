# -*- coding: utf-8 -*-
import csv, os

input_dir = os.getenv('INPUT', '../../data')

# define distionaries
uuid_uid = {}  # Map of uuids to numeric ids
uuid_ev = {}  # a check for if the user exists in events
ctdoc = {}   # check if the document occured at least 100 times.

for c, row in enumerate(csv.DictReader(open('../processed_data/events.csv'))):
    if row['uuid'] != '':
        uuid_ev[row['uuid']] = 1
        uuid_uid[row['uuid']] = row['uid']
    if row['document_id'] not in ctdoc:
        ctdoc[row['document_id']] = 1
    else:
        ctdoc[row['document_id']] += 1
print('all docs : ' + str(len(ctdoc)))
ctdoc = { key:value for key, value in ctdoc.items() if value > 100 }
print('common docs > 100 : ' + str(len(ctdoc)))

count = 0
outfile = "../processed_data/viewed_doc_trf_source.csv"
filename = input_dir + '/page_views.csv'
# filename = input_dir + '/page_views_sample.csv.gz' # comment this out locally

for c, row in enumerate(csv.DictReader(open(filename))):
    if c % 1000000 == 0:
        print (c, count)
    if row['document_id'] not in ctdoc:
        continue
    if row['uuid'] not in uuid_ev:
        continue

    if uuid_ev[row['uuid']] == 1:
        uuid_ev[row['uuid']] = set()
    lu = len(uuid_ev[row['uuid']])
    uuid_ev[row['uuid']].add(row['document_id'])
    if lu != len(uuid_ev[row['uuid']]):
        count += 1

# Delete output file if it already exists
try:
    os.remove(outfile)
except OSError:
    pass

# Open the file to write to
fo = open(outfile, 'w')
fo.write('uuid,doc_trf_ids\n')
for i in uuid_ev:
    if uuid_ev[i] != 1:
        tmp = list(uuid_ev[i])
        fo.write('%s,%s\n' % (uuid_uid[i], ' '.join(tmp)))
        del tmp
fo.close()
