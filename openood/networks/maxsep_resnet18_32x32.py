import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet18_32x32 import ResNet18_32x32


class MaxSepResNet18_32x32(ResNet18_32x32):
    """ResNet18_32x32 whose learned classifier is replaced by a fixed maximum-
    separation (simplex equiangular tight frame) matrix.

    Following Kasarla et al., "Maximum Class Separation as Inductive Bias in
    One Matrix" (NeurIPS 2022): a fixed ETF maps the 512-d feature directly to
    the class logits, so ``self.fc`` becomes an identity and the only learned
    parameters are in the backbone.
    """
    def __init__(self, num_classes=10, seed=0):
        super(MaxSepResNet18_32x32, self).__init__(num_classes=num_classes)

        feat_dim = self.feature_size  # 512 for ResNet18
        C = num_classes
        assert feat_dim >= C - 1, \
            'feature dim ({}) must be >= num_classes - 1 ({})'.format(
                feat_dim, C - 1)

        generator = torch.Generator().manual_seed(seed)
        # orthonormal basis spanning a C-dim subspace of R^{feat_dim}
        U = torch.linalg.qr(torch.randn(feat_dim, C, generator=generator))[0]
        # simplex ETF directions in that subspace; columns are the C class
        # vertices, pairwise cosine = -1/(C-1) (maximally separated)
        centering = torch.eye(C) - torch.ones(C, C) / C
        M = math.sqrt(C / (C - 1)) * (U @ centering)

        # buffer (not a Parameter) -> never updated, saved in state_dict
        self.register_buffer('etf', M)  # shape (feat_dim, C)
        self.fc = nn.Identity()

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = feature @ self.etf
        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = feature @ self.etf

        return logits_cls

    def get_fc(self):
        # mirror nn.Linear semantics: logits = feature @ etf == feature @ W.T
        # so the equivalent weight is etf.T and the bias is zero
        w = self.etf.t().cpu().detach().numpy()
        b = np.zeros(w.shape[0], dtype=w.dtype)
        return w, b

    def get_fc_layer(self):
        return self.fc
