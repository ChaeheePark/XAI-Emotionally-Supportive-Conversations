import json
import random
import csv
import copy


with open('ESConv.json', 'r') as f:
    jcorp = json.load(f)

with open('sentiment_chatbot_dataset.csv', newline='') as f:
    reader = csv.reader(f)
    tag_data = list(reader)

tag = copy.deepcopy(tag_data[1:])  # remove column name

# start and end index of dialog
tag_idx = list()
for i, t in enumerate(tag):
    if t[0].split()[2] == '0':
        tag_idx.append(i)
s_e = list()
for j in range(1300):
    if j != 1299:
        s_e.append((tag_idx[j], tag_idx[j+1]-1))
    else:
        s_e.append((tag_idx[j], len(tag)-1))

# emotion tag
emt = {'anger': 0, 'anxiety': 0, 'depression': 0, 'disgust': 0, 'fear': 0, 'guilt': 0,
       'jealousy': 0, 'nervousness': 0, 'pain': 0, 'sadness': 0, 'shame': 0, 'neutral': 0}
ds = list()

for cnt in range(len(jcorp)):
    d = dict()
    u_emo = 'neutral'
    emt_ = copy.deepcopy(emt)
    d["emotion_type"] = jcorp[cnt]["emotion_type"]
    d["problem_type"] = jcorp[cnt]["problem_type"]
    d["situation"] = jcorp[cnt]["situation"]
    d_list = list()
    s, e = s_e[cnt][0], s_e[cnt][1]

    for n in range(e - s + 1):
        tmp = dict()
        if tag[s][2] != '':
            if tag[s][1] == '':
                tmp["text"] = tag[s][2]
                tmp["speaker"] = "usr"
                emt_[tag[s][3]] += 1
                if (s < e and tag[s + 1][1] != '') or s == e:
                    u_emo = max(emt_, key=emt_.get)
                    if u_emo == 'neutral':
                        emt_.pop('neutral')
                        if emt_[max(emt_, key=emt_.get)] != 0:
                            u_emo = max(emt_, key=emt_.get)
                    emt_ = copy.deepcopy(emt)
            else:
                tmp["text"] = tag[s][2]
                tmp["speaker"] = "sys"
                tmp["strategy"] = tag[s][1]
                tmp["emotion"] = u_emo
            d_list.append(tmp)
        s += 1

    d["dialog"] = d_list

    ds.append(d)

# shuffle dataset
random.shuffle(ds)

# split and write files
# train : vaild : test = 0.7 : 0.15 : 0.15
f = open('train.txt', 'w+', encoding='utf-8')
for d_dict in ds[:910]:
    tmp = json.dumps(d_dict)
    f.write(tmp+'\n')
f.close()

f = open('vaild.txt', 'w+', encoding='utf-8')
for d_dict in ds[910:1105]:
    tmp = json.dumps(d_dict)
    f.write(tmp+'\n')
f.close()

f = open('test.txt', 'w+', encoding='utf-8')
for d_dict in ds[1105:]:
    tmp = json.dumps(d_dict)
    f.write(tmp+'\n')
f.close()
