from models import FeatureEncoder, BaseEncoder, ScoreEncoder, PatchReconstructor
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt


def test1():
    FE = FeatureEncoder()
    BE = BaseEncoder()
    SE = ScoreEncoder()
    t = torch.randn(4, 2048, 7, 7)
    print("Embedding...")
    embedding = FE(t)
    print("embedding shape:", embedding.shape)
    print("Generating bases")
    bases = BE(embedding)
    print("bases shape:", bases.shape)
    h_bases = bases[:, 0]
    v_bases = bases[:, 1]
    print("V bases shape:",
          v_bases.shape)
    scores = SE(embedding)
    print("Generating scores")
    print("scores shape:", scores.shape)
    h_scores = scores[:, 0]
    v_scores = scores[:, 1]
    print("V scores shape:", h_scores.shape)
    PR = PatchReconstructor()
    res = PR(h_scores, v_scores, h_bases, v_bases)

def test2():






