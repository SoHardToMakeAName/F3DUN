import torch
from torch import nn
import torch.nn.functional as F

def make_model(args):
    return F3DModel(args)

def default_conv3d(in_channels, out_channels,  kernel_size=3, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class Block3D(nn.Module):
    def __init__(self, wn, n_feats):
        super(Block3D, self).__init__()
        self.body = nn.Sequential(wn(default_conv3d(n_feats, n_feats, 3)),
                                      nn.ReLU(inplace=True),
                                      wn(default_conv3d(n_feats, n_feats, 3)))

    def forward(self, x):
        return self.body(x) + x

class Block3Dv2(nn.Module):
    def __init__(self, wn, n_feats):
        super(Block3Dv2, self).__init__()
        self.fusion = wn(default_conv3d(2*n_feats, n_feats, 1))
        self.body = nn.Sequential(wn(default_conv3d(n_feats, n_feats, 3)),
                                  nn.ReLU(inplace=True),
                                  wn(default_conv3d(n_feats, n_feats, 3)))

    def forward(self, x):
        x = self.fusion(x)
        return self.body(x) + x


class F3DModel(nn.Module):
    def __init__(self, args):
        super(F3DModel, self).__init__()

        scale = args.n_scale
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_threeUint = args.n_threeUnit

        if args.dataset_name.startswith('Cave'):
            band_mean = (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834,
                     0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194,
                     0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541)  # CAVE
        elif args.dataset_name == 'NTIRE':
            band_mean = (0.1251, 0.1251, 0.1251, 0.1261, 0.1394, 0.1537, 0.1626, 0.1660, 0.1671, 0.1678, 0.1695,
                     0.1719, 0.1760, 0.1827, 1.1887, 0.1918, 0.1918, 0.1903, 0.1896, 0.1896, 0.1907, 0.1916,
                     0.1905, 0.1890, 0.1883, 0.1875, 0.1879, 0.1883, 0.1846, 0.1823, 0.1884) #NTIRE
        elif args.dataset_name == 'Pavia':
            band_mean = (0.0945, 0.0863, 0.0805, 0.0813, 0.0833, 0.0840, 0.0844, 0.0841, 0.0843, 0.0851, 0.0859,
                         0.0855, 0.0854, 0.0860, 0.0866, 0.0867, 0.0878, 0.0891, 0.0901, 0.0913, 0.0927, 0.0945,
                         0.0959, 0.0974, 0.0998, 0.1027, 0.1043, 0.1052, 0.1060, 0.1076, 0.1096, 0.1111, 0.1122,
                         0.1138, 0.1153, 0.1166, 0.1177, 0.1188, 0.1200, 0.1211, 0.1217, 0.1219, 0.1223, 0.1232,
                         0.1245, 0.1255, 0.1259, 0.1260, 0.1266, 0.1266, 0.1272, 0.1281, 0.1287, 0.1293, 0.1292,
                         0.1288, 0.1289, 0.1291, 0.1293, 0.1293, 0.1297, 0.1305, 0.1313, 0.1322, 0.1332, 0.1346,
                         0.1372, 0.1409, 0.1460, 0.1510, 0.1562, 0.1616, 0.1665, 0.1713, 0.1769, 0.1833, 0.1894,
                         0.1955, 0.2001, 0.2040, 0.2078, 0.2108, 0.2113, 0.2070, 0.2059, 0.2088, 0.2112, 0.2117,
                         0.2118, 0.2124, 0.2137, 0.2132, 0.2122, 0.2125, 0.2130, 0.2124, 0.2116, 0.2104, 0.2089,
                         0.2073, 0.2048, 0.2054)  # PaviaCenter
        elif args.dataset_name == 'Harvard':
            band_mean = (0.0100, 0.0137, 0.0219, 0.0285, 0.0376, 0.0424, 0.0512, 0.0651, 0.0694, 0.0723, 0.0816,
                                             0.0950, 0.1338, 0.1525, 0.1217, 0.1187, 0.1337, 0.1481, 0.1601, 0.1817, 0.1752, 0.1445,
                                             0.1450, 0.1378, 0.1343, 0.1328, 0.1303, 0.1299, 0.1456, 0.1433, 0.1303) #Hararvd
        else:
            band_mean = (0.0059, 0.0121, 0.0142, 0.0171, 0.0178, 0.0197, 0.0195, 0.0182, 0.0173, 0.0159, 0.0163,
                         0.0173, 0.0183, 0.0203, 0.0193, 0.0209, 0.0210, 0.0214, 0.0230, 0.0246, 0.0251, 0.0243,
                         0.0245, 0.0244, 0.0235, 0.0233, 0.0233, 0.0239, 0.0246, 0.0270, 0.0283, 0.0296, 0.0323,
                         0.0349, 0.0374, 0.0385, 0.0398, 0.0411, 0.0432, 0.0420, 0.0417, 0.0407, 0.0405, 0.0391,
                         0.0387, 0.0380, 0.0378, 0.0380, 0.0379, 0.0380, 0.0371, 0.0370, 0.0365, 0.0364, 0.0360,
                         0.0357, 0.0347, 0.0341, 0.0326, 0.0314, 0.0306, 0.0307, 0.0320, 0.0356, 0.0443, 0.0541,
                         0.0646, 0.0754, 0.0862, 0.0970, 0.1076, 0.1182, 0.1285, 0.1384, 0.1474, 0.1533, 0.1567,
                         0.1595, 0.1616, 0.1634, 0.1649, 0.1641, 0.1651, 0.1647, 0.1642, 0.1636, 0.1630, 0.1626,
                         0.1618, 0.1611, 0.1612, 0.1608, 0.1605, 0.1603, 0.1620, 0.1632, 0.1585, 0.1621, 0.1625,
                         0.1630, 0.1627, 0.1623, 0.1620, 0.1609, 0.1603, 0.1591, 0.1591, 0.1588, 0.1581, 0.1575,
                         0.1567, 0.1557, 0.1463, 0.1429, 0.1377, 0.1298, 0.1268, 0.1229, 0.1285, 0.1243, 0.1225,
                         0.1188, 0.1145, 0.1091, 0.1032, 0.0975, 0.0920, 0.0887)  # Chikusei

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, len(band_mean), 1, 1])
        self.relu = nn.ReLU(inplace=True)

        head = []
        head.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
        head.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))
        self.head = nn.Sequential(*head)

        self.nearest = nn.Upsample(scale_factor=scale, mode='nearest')

        self.threeUnits = nn.ModuleList([Block3D(wn, n_feats) for _ in range(n_threeUint)])
        self.threeUnits2 = nn.ModuleList([Block3D(wn, n_feats)] + [Block3Dv2(wn, n_feats) for _ in range(n_threeUint)])

        tail = []
        if scale == 8:
            tail.append(wn(
                nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + 2, 2 + 2), stride=(1, 2, 2),
                                   padding=(1, 1, 1))))
            tail.append(nn.ReLU(True))
            tail.append(wn(
                nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + 4, 2 + 4), stride=(1, 4, 4),
                                   padding=(1, 1, 1))))
        else:
            tail.append(wn(
                nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                   padding=(1, 1, 1))))
        tail.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))
        self.tail = nn.Sequential(*tail)

    def forward(self, x, band_mean=None):
        if band_mean is not None:
            self.band_mean = band_mean
        x = x - self.band_mean.to(x.device)
        CSKC = self.nearest(x)
        x = x.unsqueeze(1)
        x = self.head(x)

        res = x
        feas_3D = []
        for m in self.threeUnits:
            feas_3D.append(res)
            res = m(res)
        x = res + x
        res = x
        for i in range(len(self.threeUnits2)):
            m = self.threeUnits2[i]
            if i == 0:
                # print(res.shape)
                res = m(res)
            else:
                fea_3D = feas_3D[-1*i]
                # print(fea_3D.shape, res.shape)
                res = m(torch.cat([fea_3D, res], dim=1))
        x = res + x

        x = self.tail(x)
        x = x.squeeze(1)

        x = torch.add(x, CSKC)
        x = x + self.band_mean.to(x.device)
        return x