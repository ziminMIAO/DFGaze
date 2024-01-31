f='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2.csv'
a='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2-0.csv'
b='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2-1.csv'
c='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2-2.csv'
string='real_'  #'real_'  #realff      real
import csv

filename =f

with open(filename, mode='r') as file:
    reader = csv.reader(file)
    rows = [row for row in reader]

header = ['img_name', 'fake_c', 'real_c', 'label']
rows.insert(0, header)  

output_filename = a
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows) 

import pandas as pd

df = pd.read_csv(a)

avg_df = df.groupby("img_name")["real_c"].mean()

new_df = avg_df.reset_index(name='avg_real_conf')

new_df = new_df[["img_name", 'avg_real_conf']].rename(columns={'avg_real_conf': 'avg_real_conf'})

new_df.to_csv(b, index=False)

import pandas as pd

df = pd.read_csv(b)

df["label_realcon"] = (df["avg_real_conf"] >= 0.5).astype(int)

df["label_ori"] = (df["img_name"].str.contains(string)).astype(int)

new_df = df[["img_name", "avg_real_conf", "label_realcon", "label_ori"]].rename(columns={"avg_real_conf": "avg_real_conf"})

df.to_csv(c, index=False)
import numpy
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.integrate import simps
df = pd.read_csv(c)
df['label_realcon'] = pd.to_numeric(df['label_realcon'], errors='coerce')
df['label_ori'] = pd.to_numeric(df['label_ori'], errors='coerce')


accuracy = sum(df['label_realcon'] == df['label_ori']) / len(df)
print('Accuracy:', accuracy)

# auc = roc_auc_score(df['label_ori'], df['label_realcon'])
auc= roc_auc_score(df['label_ori'], df['avg_real_conf'])


print('AUC:', auc)
a='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2-0.csv'
b='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2-1.csv'
c='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/8_32/trainwild_testwild/2-2.csv'
string='real_'  #'real_'  #realff      real
import csv

filename =f

with open(filename, mode='r') as file:
    reader = csv.reader(file)
    rows = [row for row in reader]

header = ['img_name', 'fake_c', 'real_c', 'label']
rows.insert(0, header)

output_filename = a
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows) 

import pandas as pd

df = pd.read_csv(a)

avg_df = df.groupby("img_name")["real_c"].mean()

new_df = avg_df.reset_index(name='avg_real_conf')

new_df = new_df[["img_name", 'avg_real_conf']].rename(columns={'avg_real_conf': 'avg_real_conf'})

new_df.to_csv(b, index=False)

import pandas as pd

df = pd.read_csv(b)

df["label_realcon"] = (df["avg_real_conf"] >= 0.5).astype(int)

df["label_ori"] = (df["img_name"].str.contains(string)).astype(int)

new_df = df[["img_name", "avg_real_conf", "label_realcon", "label_ori"]].rename(columns={"avg_real_conf": "avg_real_conf"})

df.to_csv(c, index=False)
import numpy
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.integrate import simps
df = pd.read_csv(c)
df['label_realcon'] = pd.to_numeric(df['label_realcon'], errors='coerce')
df['label_ori'] = pd.to_numeric(df['label_ori'], errors='coerce')


accuracy = sum(df['label_realcon'] == df['label_ori']) / len(df)
print('Accuracy:', accuracy)

# auc = roc_auc_score(df['label_ori'], df['label_realcon'])
auc= roc_auc_score(df['label_ori'], df['avg_real_conf'])


print('AUC:', auc)
