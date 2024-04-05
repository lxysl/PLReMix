import torch
import torch.nn as nn
import torch.nn.functional as F


class PLReMixModel(nn.Module):
    def __init__(self, backbone, dataset, n_clusters, n_heads=2):
        super(PLReMixModel, self).__init__()
        self.backbone_dim = backbone['dim']
        self.n_heads = n_heads
        self.dataset = dataset
        self.low_dim = 128
        self.temperature = 0.1
        self.proto_m = 0.99
        assert (isinstance(self.n_heads, int))

        self.backbone = backbone['backbone']
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(), nn.Linear(self.backbone_dim, self.low_dim))
        self.cls_head = nn.Linear(self.backbone_dim, n_clusters)
        self.register_buffer("prototypes", torch.zeros(n_clusters, self.low_dim))

    def forward(self, x, forward_pass='default', sharpen=True):
        if forward_pass == 'default':
            features = self.backbone(x)
            logits = self.cls_head(features)
            return logits

        elif forward_pass == 'backbone':
            logits = self.backbone(x)
            return logits

        elif forward_pass == 'cls':
            features = self.backbone(x)
            logits = self.cls_head(features)
            return logits

        elif forward_pass == 'proj':
            features = self.backbone(x)
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            return q

        elif forward_pass == 'cls_proj':
            features = self.backbone(x)
            logits = self.cls_head(features)
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            return logits, q

        elif forward_pass == 'cls_head':
            logits = self.cls_head(x)
            return logits

        elif forward_pass == 'proj_head':
            q = self.projector(x)
            q = F.normalize(q, dim=1)
            return q

        elif forward_pass == 'head':
            logits = self.cls_head(x)
            q = self.projector(x)
            q = F.normalize(q, dim=1)
            return logits, q

        elif forward_pass == 'all':
            features = self.backbone(x)
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            prototypes = self.prototypes.clone().detach()
            if sharpen:
                logits_proto = torch.mm(q, prototypes.t()) / self.temperature
            else:
                logits_proto = torch.mm(q, prototypes.t())
            logits = self.cls_head(features)
            return logits, logits_proto, q

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

    @torch.no_grad()
    def init_prototypes(self, features, labels):
        # features, labels are lists of tensors
        assert set(range(len(self.prototypes))) == set(labels.cpu().numpy()), \
            f'The missing labels are {set(range(len(self.prototypes))) - set(labels.cpu().numpy())}'
        for i in range(self.prototypes.shape[0]):
            self.prototypes[i] = features[labels == i].mean(dim=0)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    @torch.no_grad()
    def update_prototypes(self, features, labels):
        avg_features = torch.zeros_like(self.prototypes).to(self.prototypes.device)
        for i in range(avg_features.shape[0]):
            avg_features[i] = features[labels == i].mean(dim=0)
        avg_features = F.normalize(avg_features, p=2, dim=1)
        for i in range(avg_features.shape[0]):
            if not torch.isnan(avg_features[i]).any():
                self.prototypes[i] = self.proto_m * self.prototypes[i] + (1 - self.proto_m) * avg_features[i]
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
