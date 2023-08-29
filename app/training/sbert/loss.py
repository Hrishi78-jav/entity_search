import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from typing import Dict, Iterable
from sentence_transformers import util


class MultipleNegativesRankingLoss2(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss = 0

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_q = reps[0]  # query embedding
        embeddings_d = torch.cat(
            reps[1:])  # doc embedding  = (d1,d2....,n1,n2....) = where d1,d2... are docs and n1,n2.. hard negatives

        # query-doc, query-query, doc-doc similarity matrices
        qd = torch.exp(self.similarity_fct(embeddings_q,
                                           embeddings_d) * self.scale)  # q * d = BS * ((k+1)*BS) = 4*8   if k=1 ===>(a,p,n)
        qq = torch.exp(self.similarity_fct(embeddings_q, embeddings_q) * self.scale)  # q * q = BS * BS = 4*4
        dd = torch.exp(
            self.similarity_fct(embeddings_d, embeddings_d) * self.scale)  # d * d = ((k+1)*BS) * ((k+1)*BS) = (8*8)
        batchsize = len(embeddings_q)
        # forming required masks
        mask_qq = torch.eye(qd.shape[0])
        mask_dd = torch.eye(qd.shape[1])
        extra_zeros = torch.zeros(batchsize, qd.shape[1] - qd.shape[0])
        mask_numerator = torch.cat((mask_qq, extra_zeros), dim=1)  # q*d
        mask_qiqj = torch.ones_like(mask_qq) - mask_qq  # q*q
        mask_djdi = torch.ones_like(mask_dd) - mask_dd  # d*d

        # numerator and denominator term according to gte paper improved contrastive loss
        numerator = (qd * mask_numerator).sum(dim=-1)
        denominator_qidj = qd.sum(dim=-1)
        denominator_qjdi = qd.sum(dim=0).reshape(-1, batchsize).sum(dim=0)
        denominator_qiqj = (qq * mask_qiqj).sum(dim=-1)
        denominator_djdi = (dd * mask_djdi).sum(dim=-1).reshape(-1, batchsize).sum(dim=0)
        self.loss = (-(1 / batchsize) * torch.log(
            numerator / (denominator_qidj + denominator_qjdi + denominator_qiqj + denominator_djdi))).sum()
        return self.loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


class MultipleNegativesRankingLoss3(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss = 0

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_q = reps[0]  # query embedding
        embeddings_d = torch.cat(
            reps[1:])  # doc embedding  = (d1,d2....,n1,n2....) = where d1,d2... are docs and n1,n2.. hard negatives

        # query-doc, query-query, doc-doc similarity matrices
        qd = torch.exp(self.similarity_fct(embeddings_q,
                                           embeddings_d) * self.scale)  # q * d = BS * ((k+1)*BS) = 4*8   if k=1 ===>(a,p,n)
        batchsize = len(embeddings_q)
        # forming required masks
        mask_qq = torch.eye(qd.shape[0])
        extra_zeros = torch.zeros(batchsize, qd.shape[1] - qd.shape[0])
        mask_numerator = torch.cat((mask_qq, extra_zeros), dim=1)          # q*d

        # numerator and denominator term according to gte paper improved contrastive loss
        numerator = (qd * mask_numerator).sum(dim=-1)
        denominator_qidj = qd.sum(dim=-1)
        denominator_qjdi = qd.sum(dim=0).reshape(-1, batchsize).sum(dim=0)

        self.loss = (-(1 / batchsize) * torch.log(numerator / (denominator_qidj + denominator_qjdi))).sum()
        return self.loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}
