import torch
from torch import nn
import torch.nn.functional as F
from common import pixel_unshuffle

class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss

class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, SR, HR):
        b, c, h, w = HR.shape
        eps = 1e-8
        SR_ = SR.view(b, c, -1)
        HR_ = HR.view(b, c, -1)
        cos_value = F.cosine_similarity(SR_, HR_, dim=1)
        mask = torch.logical_and(cos_value < 1 - eps, cos_value > -1 + eps)
        cos_value_ = torch.masked_select(cos_value, mask)
        # loss = torch.mean(1 + eps - cos_value)
        loss = torch.mean(torch.acos(cos_value_))
        return loss
        # losses = []
        # for i in range(HR.shape[-2]):
        #     for j in range(HR.shape[-1]):
        #         x, y = SR[:, :, i, j], HR[:, :, i, j]
        #         if x.isnan().any() or torch.linalg.norm(x, ord=2) == 0 or torch.linalg.norm(y, ord=2) == 0:
        #             continue
        #         cos_value = F.cosine_similarity(x, y, dim=1)
        #         losses.append(torch.mean(torch.acos(cos_value)))
        # return sum(losses)/len(losses)

# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class AggregationLoss(nn.Module):
    def __init__(self, loss_args):
        super(AggregationLoss, self).__init__()
        losses = loss_args.split('+')
        self.loss = list()
        for term in losses:
            weight, loss = term.split('*')
            weight = float(weight)
            if loss == 'MSE':
                loss_func = nn.MSELoss(reduction='mean')
            elif loss == 'L1':
                loss_func = nn.L1Loss(reduction='mean')
            elif loss == 'Hybrid':
                loss_func = HybridLoss(spatial_tv=True, spectral_tv=True)
            elif loss == 'SAM':
                loss_func = SAMLoss()
            else:
                raise Exception("{} loss is not supported for now!")
            self.loss.append({
                'weight': weight,
                'function': loss_func
            })


    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight']*loss
                losses.append(effective_loss)
        loss_sum = sum(losses)
        return loss_sum

class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()
        pass

    def forward(self, fea_list1, fea_list2):
        if len(fea_list1) != len(fea_list2):
            raise Exception("The length of two lists must be equal.")
        n = len(fea_list1)
        losses = []
        for i in range(n):
            fea1, fea2 = fea_list1[i], fea_list2[i]
            b, c, h, w = fea1.shape
            fea1_reshaped = fea1.view(b, c, h*w)
            fea2_reshaped = fea2.view(b, c, h * w)
            aff1 = F.softmax(torch.matmul(fea1_reshaped.transpose(1, 2), fea1_reshaped))
            aff2 = F.softmax(torch.matmul(fea2_reshaped.transpose(1, 2), fea2_reshaped))
            aff_loss = F.l1_loss(aff1, aff2, reduction='mean')
            losses.append(aff_loss)
        return sum(losses)/len(losses)

class OutputLoss(nn.Module):
    def __init__(self):
        super(OutputLoss, self).__init__()
        pass

    def forward(self, fea_list1, fea_list2):
        if len(fea_list1) != len(fea_list2):
            raise Exception("The length of two lists must be equal.")
        n = len(fea_list1)
        losses = []
        for i in range(n):
            fea1, fea2 = fea_list1[i], fea_list2[i]
            out_loss = F.l1_loss(fea1, fea2, reduction='mean')
            losses.append(out_loss)
        return sum(losses)/len(losses)

class OutputLoss2(nn.Module):
    def __init__(self, channels, distill_channel, distill_num):
        super(OutputLoss2, self).__init__()

        # self.convs0 = nn.ModuleList([nn.Conv2d(channels[0], distill_channel, 1) for _ in range(distill_num)])
        self.convs1 = nn.ModuleList([nn.Conv2d(channels[1], distill_channel, 1) for _ in range(distill_num)])

    def forward(self, fea_list1, fea_list2):
        if len(fea_list1) != len(fea_list2):
            raise Exception("The length of two lists must be equal.")
        n = len(fea_list1)
        losses = []
        for i in range(n):
            fea1, fea2 = fea_list1[i], fea_list2[i]
            # print(fea1.shape, fea2.shape)
            out_loss = F.l1_loss(fea1, self.convs1[i](fea2), reduction='mean')
            losses.append(out_loss)
        return sum(losses)/len(losses)

class OutputLoss3(nn.Module):
    def __init__(self, channels, distill_num, conv, scale, repeats):
        super(OutputLoss3, self).__init__()

        # self.convs = nn.ModuleList([conv(channels[0], channels[0]//(scale**2), 1) for _ in range(distill_num)])
        self.scale = scale
        self.repeats = repeats

    def forward(self, fea_list1, fea_list2):
        if len(fea_list1) != len(fea_list2):
            raise Exception("The length of two lists must be equal.")
        n = len(fea_list1)
        losses = []
        for i in range(n):
            fea1, fea2 = fea_list1[i], fea_list2[i]
            fea2 = F.interpolate(fea2, size=fea1.shape[-2:], mode='nearest')
            if fea1.shape[0] != fea2.shape[0]:
                fea2 = torch.repeat_interleave(fea2, repeats=self.repeats, dim=0)
            out_loss = F.l1_loss(fea1, fea2, reduction='mean')
            losses.append(out_loss)
        return sum(losses)/len(losses)