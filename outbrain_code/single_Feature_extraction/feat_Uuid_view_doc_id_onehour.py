# -*- coding: utf-8 -*-

import csv, os

input_dir = os.getenv('INPUT', '../data')

uuid_uid = {}  # Map of uuids to numeric ids
uuidtstamps = {}   # store the users along with the ties they were presented ads
uuidhrafter = {}   # store the folloing infomration from each event
                   # key:{user, event timestamps} and value{all documents clicked within one hour}
ctdoc = {}         # the count of how often docs appear

# Get timestamp user combination for each events
for c, row in enumerate(csv.DictReader(open('../processed_data/events.csv'))):
    if row['uuid'] not in uuidtstamps:
        uuidtstamps[row['uuid']] = set()
    uuidtstamps[row['uuid']].add(row['timestamp'])
    uuidhrafter[row['uuid']+'_'+row['timestamp']] = 1
    uuid_uid[row['uuid']] = row['uid']


count = 0
outfile = "../processed_data/viewed_docs_one_hour_after.csv"
filename = '../../data/page_views.csv'
# filename = input_dir + '/page_views_sample.csv.gz' # comment this out locally

# Count documents which occured less than 80 times and exclude them
for c, row in enumerate(csv.DictReader(open(filename))):
    if row['uuid'] not in uuidtstamps:
        continue
    if row['document_id'] not in ctdoc:
        ctdoc[row['document_id']] = 1
    else:
        ctdoc[row['document_id']] += 1
print('all docs : ' + str(len(ctdoc)))
ctdoc = { key:value for key, value in ctdoc.items() if value > 80 }
print('common docs > 80 : ' + str(len(ctdoc)))

# for each page_views row where we get a uuid match, and the document occurs over
# the required 80 count, we loop through the users click timestamps to find if
# is within one hour of any of the event timestamps.
for c, row in enumerate(csv.DictReader(open(filename))):
    if c % 1000000 == 0:
        print (c, count)
    if row['document_id'] not in ctdoc:
        continue
    if row['uuid'] not in uuidtstamps:
        continue

    for time in uuidtstamps[row['uuid']]:
        diff = int(row['timestamp']) - int(time)
        if abs(diff) < 3600*1000:
            if diff > 0:
                if uuidhrafter[row['uuid'] + '_' + time] == 1:
                    uuidhrafter[row['uuid'] + '_' + time] = set()
                uuidhrafter[row['uuid'] + '_' + time].add(row['document_id'])
        del diff

# Delete output file if it already exists
try:
    os.remove(outfile)
except OSError:
    pass

# Open the file to write to
fo = open(outfile, 'w')
fo.write('uuid,timestamp,doc_ids\n')
for i in uuidhrafter:
    if uuidhrafter[i] != 1:
        tmp = list(uuidhrafter[i])
        utime = i.split('_')
        fo.write('%s,%s,%s\n' % (uuid_uid[utime[0]], utime[1], ' '.join(tmp)))
        del tmp, utime
fo.close()
