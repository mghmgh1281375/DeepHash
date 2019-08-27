import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import sys, json, csv
from tqdm import tqdm

filename = sys.argv[1]
img_path, img_lbl = 'image', 'data'

df = pd.read_csv(filename)
if not 'image' in df.columns:
    img_path, img_lbl = 'path', 'label'

labels = df[img_lbl].tolist()
try:
    labels = [a[0].split('|')[1] if a[1] else '' for lbl in labels for a in json.loads(lbl)]
except TypeError as _:
    print('skip label parsing.')
labels = LabelBinarizer().fit_transform(labels)
labels_str = [' '.join(list(map(str, list(row)))) for row in labels]
#for row in tqdm(labels):
#    labels_str.append(' '.join(list(map(str, labels))))
print(len(df[img_lbl]), '==', len(labels_str))
df['one_hot'] = labels_str
df.to_csv('%s.deephash.txt'%filename, header=None, index=None, columns=[img_path, 'one_hot'], sep='\t')

