from sentence_transformers import SentenceTransformer
import pickle
import faiss
from tqdm import tqdm


def predict2(vocab, query_vocab, model, index, top=100, n_egs=100):
    d = {}
    for i in tqdm(range(400000, len(query_vocab[:n_egs]))):
        query = query_vocab[i]
        query_emb = model.encode(query, convert_to_numpy=True).reshape(1, -1)
        D, output = index.search(query_emb, top)
        temp = [vocab[idx] for idx in output[0] if vocab[idx] not in query]
        d[query] = temp
    return d


def build(x, vocab, query_vocab, model, index, n_egs=10):
    index.nprobe = x
    r = predict2(vocab, query_vocab, model, index, top=200, n_egs=n_egs)
    with open(
            f'/home/ubuntu/checkpoints/models_checkpoints/entity_search/hard_negatives/query_hard_negative_nprobe_{x}_gte_400000_to_end.pkl',
            'wb') as f:
        pickle.dump(r, f)


DATA_DIR = '/home/ubuntu/data'
print('Load Model.....')
# model = SentenceTransformer(DATA_DIR + '/finetuned_miniLM_hn_32_10_old_data')
model = SentenceTransformer(DATA_DIR + '/Finetuned_gte_small_3M')

print('Load Data......')
with open(DATA_DIR + '/ner_vocab_for_3M.pkl',
          'rb') as f:  # for gte,
    vocab = pickle.load(f)

with open(DATA_DIR + '/query_vocab.pkl',
          'rb') as f:  # for gte,
    query_vocab = pickle.load(f)

print('Load Embeddings......')
# with open(DATA_DIR + '/best_model_embeddings.pkl', 'rb') as f:
#     finetuned_miniLM_embeddings = pickle.load(f)

with open(DATA_DIR + '/ner_vocab_finetuned_gte_small_embeddings_for_3M.pkl', 'rb') as f:
    finetuned_gte_embeddings = pickle.load(f)

nlist = 50  # No. of  voronoi cell/ clusters we want our embedding space to get divided
quantizer = faiss.IndexFlatL2(384)
index = faiss.IndexIVFFlat(quantizer, 384, nlist)

# index_gpu.train(finetuned_miniLM_embeddings)  # train the model to cluster - find centroids
# index_gpu.add(finetuned_miniLM_embeddings)  # add the embeddings
index.train(finetuned_gte_embeddings)
index.add(finetuned_gte_embeddings)

print('Process Starting ....')
n = len(query_vocab)
build(10, vocab, query_vocab, model, index, n_egs=n)
