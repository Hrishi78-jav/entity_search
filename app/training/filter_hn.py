from sentence_transformers import CrossEncoder
import pickle
from tqdm import tqdm

print('Loading Model---')
model = CrossEncoder('cross-encoder/stsb-roberta-base')
DATA_DIR = '/home/ubuntu/data'
hn1_path = DATA_DIR + '/hard_negative_nprobe_50_miniLM_best.pkl'
hn2_path = DATA_DIR + '/hard_negative_nprobe_50_gte.pkl'
# hn1_path = DATA_DIR + '/query_hard_negative_nprobe_10_minilm_best.pkl'
# hn2_path = DATA_DIR + '/query_hard_negative_nprobe_10_gte.pkl'

print('Loading Data---')
with open(hn2_path, 'rb') as f:
    hn = pickle.load(f)
# with open(hn2_path, 'rb') as f:
#     hn = pickle.load(f)

print('Creating sentence pairs')
sp = []
for word, val in tqdm(hn.items()):
    temp = [[word, x] for x in val[:30]]
    sp += temp
n = len(sp)

r = list(model.predict(sp[:int(0.33 * n)], show_progress_bar=True, batch_size=32))
with open(f'/home/ubuntu/checkpoints/models_checkpoints/entity_search/hard_negatives/query_false_negatives_hn2_1.pkl',
          'wb') as f:
    pickle.dump(r, f)

del r
r = list(model.predict(sp[int(0.33 * n):int(0.66 * n)], show_progress_bar=True, batch_size=32))
with open(f'/home/ubuntu/checkpoints/models_checkpoints/entity_search/hard_negatives/query_false_negatives_hn2_2.pkl',
          'wb') as f:
    pickle.dump(r, f)

del r
r = list(model.predict(sp[int(0.66 * n):], show_progress_bar=True, batch_size=32))
with open(f'/home/ubuntu/checkpoints/models_checkpoints/entity_search/hard_negatives/query_false_negatives_hn2_3.pkl',
          'wb') as f:
    pickle.dump(r, f)

# print('Creating sentence pairs-2')
# sp2 = []
# for word, val in tqdm(hn2.items()):
#     temp = [[word, x] for x in val[:30]]
#     sp2 += temp
#
# r2 = list(model.predict(sp2, show_progress_bar=True, batch_size=32))
# with open(f'/home/ubuntu/checkpoints/models_checkpoints/entity_search/hard_negatives/false_negatives_hn2.pkl',
#           'wb') as f:
#     pickle.dump(r2, f)
