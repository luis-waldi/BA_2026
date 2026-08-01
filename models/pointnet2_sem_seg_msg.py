import os

import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

# Radienpaare der vier Set-Abstraction-Layer, per HEATH_RADII umschaltbar.
# Analog zu pointnet2_sem_seg.py, siehe die Begruendung dort.
#   default  Werte der S3DIS-Vorlage
#   wide     der grobe Zweig entspricht dem Radius aus dem SSG-Modell mit
#            HEATH_RADII=wide, der feine Zweig ist halb so gross
_RADII = {
    'default': ([0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]),
    'wide':    ([0.125, 0.25], [0.25, 0.5], [0.5, 1.0], [1.0, 2.0]),
}

RADII_NAME = os.environ.get('HEATH_RADII', 'default')
if RADII_NAME not in _RADII:
    raise ValueError(f'HEATH_RADII "{RADII_NAME}" unbekannt, erlaubt: {list(_RADII)}')
RADII = _RADII[RADII_NAME]


class get_model(nn.Module):
    # in_channel: Features je Punkt. 6 = xyz+RGB, 7 = zusaetzlich NIR,
    # 9 = zusaetzlich z_rel und z_range.
    def __init__(self, num_classes, in_channel=7):
        super(get_model, self).__init__()
        r1, r2, r3, r4 = RADII

        # PointNetSetAbstractionMsg rechnet die drei xyz-Kanaele intern selbst
        # dazu (last_channel = in_channel + 3), anders als PointNetSetAbstraction
        # im SSG-Modell. Hier wird deshalb in_channel ohne Aufschlag uebergeben.
        self.sa1 = PointNetSetAbstractionMsg(1024, r1, [16, 32], in_channel, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, r2, [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, r3, [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, r4, [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))