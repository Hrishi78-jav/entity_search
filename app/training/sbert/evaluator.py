from sklearn.metrics.pairwise import paired_cosine_distances
from typing import Dict, List
from torch import Tensor
from sklearn.metrics import mean_squared_error


class custom_EmbeddingSimilarityEvaluator:
    def __init__(self):
        self.mse = 0

    def __call__(self, model, sentence_features: List[Dict[str, Tensor]], labels: Tensor):
        embeddings1 = model(sentence_features[0])['sentence_embedding'].cpu().detach().numpy()
        embeddings2 = model(sentence_features[1])['sentence_embedding'].cpu().detach().numpy()
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        self.mse = mean_squared_error(labels.cpu().detach().numpy(), cosine_scores, squared=False)
        return self.mse
