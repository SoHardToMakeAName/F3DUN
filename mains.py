import argparse
import os
import sys
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json
import yaml
from easydict import EasyDict as edict
import random
import gc
from thop import profile
from torchstat import stat

from HStest import HSTestData2
from HStrain import HSTrainingData2
from utils import get_model_by_name, adapt_state_dict, compute_num_params

# loss
from loss import HybridLoss, AggregationLoss
# from loss import HyLapLoss
from metrics import quality_assessment

# global settings

dataset_name = ''
resume = False
log_interval = 55
upscale  = 4
n_blocks = 0
n_subs   = 0
n_ovls   = 0
n_feats  = 0
n_scale1 = 0
gpus     = "0"
f_share  = False
model_title = ''
test_data_dir = ''


Loss = 'H_Real_1_1' #H_Real_1_1
other = '_X2X2_Epo40_Mix13'
save_model_title = '_Chikusei_SSPSR_Blocks=6_Subs8_Ovls2_Feats=256_Share=True_X2X2_Epo60BS32_ckpt_epoch_40'
model_name = './checkpoints/' + save_model_title + '.pth'
result_dir = './result/' + save_model_title + '.mat'

# CUDA_VISIBLE_DEVICES=5 python mains.py train --name SSPSR_HF --save_dir ./trained_model/SSPSR_HF/ --ckp_dir ./checkpoints/SSPSR_HF/ --epochs 5 --dataset_name Cave

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 64")
    train_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 40")
    train_parser.add_argument("--decay_rate", type=float, default=0.1, help="decay rate, default set to 0.1")
    train_parser.add_argument("--decay_start", type=int, default=21, help="the starting epoch to decay lr, default set to 30")
    train_parser.add_argument("--decay_stop", type=int, default=-1,
                              help="the ending epoch to decay lr, default set to 40")
    train_parser.add_argument("--stop_lr", type=float, default=-1.0, help="the ending lr in decay process")
    train_parser.add_argument("--decay_step", type=int, default=10, help="the epoch gap for lr decay, default set to 10")
    train_parser.add_argument("--name", type=str, default=None, required=True, help="model name")
    train_parser.add_argument("--model_config", type=str, default=None, help="model config file path")
    train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 3")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 2")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 4")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--base_epoch", type=int, default=0, help="the checkpoint to load")
    train_parser.add_argument("--resume", action='store_true', help='whether continue training from a saved checkpoint')
    train_parser.add_argument("--plateau", action='store_true', help='whether use ReduceLROnPlateau to reduce learning rate')
    train_parser.add_argument("--input", type=str, required=False,default='single', help='choose the input (\'single\' '
                                'means using LR patch, \'multi\' means using LR and bicubic upsampled patch as input,'
                                '\'upsampled\' means using bicubic upsampled patch as input')
    train_parser.add_argument("--loss", type=str, default='1*Hybrid', help='choose the losses used for training')
    train_parser.add_argument("--skip_threshold", type=float, default=None, help="loss skip threshold")
    train_parser.add_argument("--weight_clip", type=float, default=None, help="weight norm (default is 0)")

    train_parser.add_argument("--seed", type=int, default=None, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--ckp_dir", type=str, default="./checkpoints/",
                              help="directory for saving checkpoints, default is checkpoints folder")
    train_parser.add_argument("--out_dir", type=str, default="./results/",
                              help="directory for saving output results, default is results folder")
    train_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    test_parser.add_argument("--ckp", type=str, default=None, help="checkpoint file to load")
    test_parser.add_argument("--pth", type=str, default=None, help="parameter file to load")
    test_parser.add_argument("--model_config", type=str, default=None, help="model config file path")
    test_parser.add_argument("--dataset_name", type=str, default="Chikusei",
                              help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 4")
    test_parser.add_argument("--input", type=str, required=False, default='single',
                              help='choose the input (\'single\' '
                                   'means using LR patch, \'multi\' means using LR and bicubic upsampled patch as input,'
                                   '\'upsampled\' means using bicubic upsampled patch as input')
    test_parser.add_argument("--name", type=str, default=None, required=True, help="model name")

    args = main_parser.parse_args()
    # print(args.gpus)
    if args.model_config is not None:
        with open(args.model_config, 'r') as fid:
            yaml_config = edict(yaml.load(fid, Loader=yaml.FullLoader))
        for k, v in yaml_config.items():
            if k in vars(args).keys() and type(vars(args)[k]).__name__ == 'str':
                vars(args)[k] = str(v)
            else:
                vars(args)[k] = v
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    args.seed = random.randint(1, 10000) if args.seed is None else args.seed
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # CAVE dataset
    dataset_name = args.dataset_name
    n_blocks = args.n_blocks
    n_subs   = args.n_subs
    n_ovls   = args.n_ovls
    n_feats  = args.n_feats
    n_scale  = args.n_scale
    gpus     = 0
    use_share= args.use_share

    print('===> Loading datasets')
    repeat = 8 if args.dataset_name != 'NTIRE' else 1
    train_path = './datasets/{}_x{}/train/'.format(dataset_name, n_scale)
    eval_path = './datasets/{}_x{}/eval/'.format(dataset_name, n_scale)
    train_set = HSTrainingData2(image_dir=train_path, augment=True, repeat=repeat)
    eval_set = HSTrainingData2(image_dir=eval_path, augment=False)
    test_data_dir = './datasets/{}_x{}/{}_test_x{}.mat'.format(dataset_name, n_scale, dataset_name, n_scale)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=2, shuffle=False)

    if dataset_name=='Chikusei':
        args.n_colors = 128
    elif dataset_name=='Pavia':
        args.n_colors = 102
    else:
        args.n_colors = 31

    if args.name == 'F3DN' and args.n_threeUnit is None:
        args.n_threeUnit = int(args.n_blocks)

    print('===> Building model')
    net = get_model_by_name(args.name, args)

    other = '_X' +str(n_scale//2)+'X' +str(2)+'_BS'+str(args.batch_size)
    if args.name.find("SSPSR") != -1:
        model_title = dataset_name + '_SSPSR_Blocks='+str(n_blocks)+'_Subs'+str(n_subs)+'_Ovls'+str(n_ovls)+'_Feats='+str(n_feats)+'_Share='+str(use_share)+other
    else:
        model_title = "{}_{}".format(dataset_name, args.name) + '_X' +str(n_scale) + '_BS'+str(args.batch_size)
    model_name = args.ckp_dir + '/' + dataset_name + "_" + model_title + "_ckpt_epoch_" + str(args.base_epoch) + ".pth"
    args.model_title = model_title
    
    save_checkpoint(args, net, 1234, 1e6)
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    best_eval, best_epoch = 1e6, 0
    if args.resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            best_eval, best_epoch = checkpoint["eval_loss"], checkpoint["epoch"]
            best_ckp = checkpoint
            net.load_state_dict(adapt_state_dict(checkpoint["model"].state_dict(), False))
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()
    print('model: #params={}'.format(utils.compute_num_params(net, text=True)))
    # exit(0)

    # loss functions to choose
    agg_loss = AggregationLoss(args.loss)
    # mse_loss = torch.nn.MSELoss()
    # h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    # hylap_loss = HyLapLoss(spatial_tv=False, spectral_tv=True)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('runs/'+model_title+'_'+str(time.ctime()))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=args.stop_lr, verbose=True)
    
    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        if not args.plateau:
            adjust_learning_rate(args.learning_rate, optimizer, e+1, args.decay_rate, args.decay_start, args.decay_step,
                             args.decay_stop, args.stop_lr)
        epoch_meter.reset()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        print(model_title)
        for iteration, data in enumerate(train_loader):
            optimizer.zero_grad()
            x, gt = data['ms'].to(device), data['gt'].to(device)
            lms = F.interpolate(x, scale_factor=args.n_scale, mode='bicubic')
            # optimizer.zero_grad()
            if args.input == 'multi':
                y = net(x, lms)
            elif args.input == 'upsampled':
                y = net(lms)
            else:
                y = net(x)
            if isinstance(y, tuple):
                y = y[0]
            loss = agg_loss(y, gt)
            epoch_meter.add(loss.item())
            # print("y:", y.isnan().any())
            # print("loss:", loss.item())
            if args.skip_threshold is not None and loss.item() > args.skip_threshold or loss.isnan().any():
                pass
            else:
                # with torch.autograd.detect_anomaly():
                loss.backward()
                # for name, para in net.named_parameters():
                    # print(name, para.grad.isnan().any())
                if args.weight_clip is not None and args.weight_clip > 0:
                    torch.nn.utils.clip_grad_value_(net.parameters(), args.weight_clip)
                optimizer.step()
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), n_blocks, n_subs, n_feats, gpus, e+1, iteration + 1,
                                                                   len(train_loader), loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)
            del data, loss, x, gt, lms
            gc.collect()

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(time.ctime(), e+1, epoch_meter.value()[0]))
        # run validation set every epoch
        eval_loss = validate(args, eval_loader, net, L1_loss)
        if args.plateau:
            scheduler.step(eval_loss)
        if eval_loss < best_eval:
            best_eval = eval_loss
            best_epoch = e+1
            best_ckp = {"epoch": e+1, "model": net, "eval_loss": eval_loss}
        print("Best epoch is {}".format(best_epoch))
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 5 epochs
        if (e + 1) % 5 == 0:
            save_checkpoint(args, net, e+1, eval_loss)

    # save model after training
    net.eval().cpu()
    save_checkpoint(args, best_ckp["model"], best_ckp["epoch"], best_ckp["eval_loss"])
    save_model_filename = dataset_name + "_" + model_title + "_epoch_" + str(best_ckp["epoch"]) + "_" + \
                          str(time.ctime()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(args.save_dir, save_model_filename)
    if torch.cuda.device_count() > 1:
        torch.save(best_ckp["model"].module.state_dict(), save_model_path)
    else:
        torch.save(best_ckp["model"].state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)
    

    ## test the best model after training
    print("Running testset")
    print('===> Loading testset')
    test_set = HSTestData2(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net = best_ckp["model"]
    net.eval().cuda()
    upscale = args.n_scale
    with torch.no_grad():
        output = []
        test_number = 0
        for i, data in enumerate(test_loader):
            ms, gt = data['ms'].to(device), data['gt'].to(device)
            lms = F.interpolate(ms, scale_factor=args.n_scale, mode='bicubic')
            # Divide the image to Pi*Pi patches
            Wide, High = gt.shape[2], gt.shape[3]
            P = 1
            Widei, Highi = Wide//P, High//P   
            print(gt.shape,'gt')        

            for m in range(P):
                for n in range(P):
                    msi  = ms[:, :, int(np.ceil(Widei * m / upscale)):int(np.ceil(Widei * (m + 1) / upscale)),
                          int(np.ceil(Highi * n / upscale)):int(np.ceil(Highi * (n + 1) / upscale))]
                    lmsi = lms[:,:,Widei*m:Widei*(m+1),Highi*n:Highi*(n+1)]
                    gti  = gt[:,:,Widei*m:Widei*(m+1),Highi*n:Highi*(n+1)]
                    if args.input == 'multi':
                        yi   = net(msi, lmsi)
                    elif args.input == 'upsampled':
                        yi = net(lmsi)
                    else:
                        yi = net(msi)
                    if isinstance(yi, tuple):
                        yi = yi[0]
                    yi, gti = yi.squeeze().cpu().numpy().transpose(1, 2, 0), gti.squeeze().cpu().numpy().transpose(1, 2, 0)
                    gti = gti[:yi.shape[0], :yi.shape[1], :]
                    if i==0:
                        indices = quality_assessment(gti, yi, data_range=1., ratio=4)
                    else:
                        indices = sum_dict(indices, quality_assessment(gti, yi, data_range=1., ratio=4))
                    output.append(yi)
                    test_number += 1
            print(i,test_number)
            print(gti.shape,'gti')
        for index in indices:
            indices[index] = indices[index] / test_number
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_dir = args.out_dir + model_title + '_' + str(time.ctime()) + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .mat file at ", save_dir)
    print(indices)

    QIstr = './results/QIs/' + model_title+'_P'+str(P) +'_'+str(time.ctime())+ ".txt"
    json.dump(indices, open(QIstr, 'w'))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp



def adjust_learning_rate(start_lr, optimizer, epoch, decay_rate=0.1, start_epoch=0, step=30, stop_epoch=-1, stop_lr=-1):
    """Sets the learning rate to the initial LR decayed by decay_rate every step epochs"""
    if epoch <= start_epoch:
        return None
    if stop_epoch != -1 and epoch >= stop_epoch:
        return None
    lr = start_lr * (decay_rate ** ((epoch-start_epoch) // step))
    if lr <= stop_lr:
        return None
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, data in enumerate(loader):
            ms, gt = data['ms'].to(device), data['gt'].to(device)
            lms = F.interpolate(ms, scale_factor=args.n_scale, mode='bicubic')
            if args.input == 'multi':
                y = model(ms, lms)
            elif args.input == 'upsampled':
                y = model(lms)
            else:
                y = model(ms)
            if isinstance(y, tuple):
                y = y[0]
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]


# def test(args):
#     device = torch.device("cuda" if args.cuda else "cpu")
#     print('===> Loading testset')
#     test_set = HSTestData(test_data_dir)
#     test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
#     print('===> Start testing')
#     with torch.no_grad():
#         epoch_meter = meter.AverageValueMeter()
#         epoch_meter.reset()
#         # loading model
#         if args.ckp is not None:
#             model = torch.load(os.path.join(args.ckp_dir, args.ckp))["model"]
#         elif args.pth is not None:
#             model = get_model_by_name(args.name, args)
#             state_dict = torch.load(os.path.join(args.save_dir, args.pth))
#             model.load_state_dict(state_dict)
#         model.to(device).eval()
#         mse_loss = torch.nn.MSELoss()
#         output = torch.tensor([])
#         for i, (ms, lms, gt) in enumerate(test_loader):
#             # compute output
#             ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
#             # y = model(ms)
#             y = model(ms, lms)
#             loss = mse_loss(y, gt)
#             epoch_meter.add(loss.item())
#             y = y.cpu()
#             output = torch.cat((output, y), 0)
#         print("avg loss for each pic is :" + str(epoch_meter.value()[0]))
#         utils.save_result(result_dir, output)
#     print("\nDone, test results saved at", result_dir)

def test(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Running testset")
    print('===> Loading testset')
    test_data_dir = './datasets/{}_x{}/{}_test_x{}.mat'.format(args.dataset_name, args.n_scale, args.dataset_name, args.n_scale)
    test_set = HSTestData2(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    # loading model
    if args.ckp is not None:
        checkpoint = torch.load(args.ckp)
        net = checkpoint["model"]
        epoch = checkpoint["epoch"]
    elif args.pth is not None:
        net = get_model_by_name(args.name, args)
        state_dict = torch.load(os.path.join(args.save_dir, args.pth))
        net.load_state_dict(state_dict)
    else:
        print("load model failed!")
        # return None
    model_title = "{}_{}".format(args.dataset_name, args.name) + '_X' +str(args.n_scale)
    net.eval().to(device)
    upscale = args.n_scale
    with torch.no_grad():
        output = []
        test_number = 0
        for i, data in enumerate(test_loader):
            ms, gt = data['ms'].cuda(), data['gt'].cuda()
            lms = F.interpolate(ms, scale_factor=args.n_scale, mode='bicubic')
            noise = torch.Tensor(np.random.normal(0,5,ms.shape)/255.0).cuda()
            ms = ms + noise
            # Divide the image to Pi*Pi patches
            Wide, High = gt.shape[2], gt.shape[3]
            P = 2 if args.dataset_name == 'Chikusei' else 1
            Widei, Highi = Wide // P, High // P
            for m in range(P):
                for n in range(P):
                    msi = ms[:, :, int(np.ceil(Widei * m / upscale)):int(np.ceil(Widei * (m + 1) / upscale)),
                          int(np.ceil(Highi * n / upscale)):int(np.ceil(Highi * (n + 1) / upscale))]
                    lmsi = lms[:, :, Widei * m:Widei * (m + 1), Highi * n:Highi * (n + 1)]
                    gti = gt[:, :, Widei * m:Widei * (m + 1), Highi * n:Highi * (n + 1)]
                    if args.input == 'multi':
                        yi = net(msi, lmsi)
                    elif args.input == 'upsampled':
                        yi = net(lmsi)
                    else:
                        yi = net(msi)
                    if isinstance(yi, tuple):
                        yi = yi[0]
                    # yi = lmsi
                    yi, gti = yi.squeeze().cpu().numpy().transpose(1, 2, 0), gti.squeeze().cpu().numpy().transpose(1, 2,
                                                                                                                   0)
                    gti = gti[:yi.shape[0], :yi.shape[1], :]
                    # print(yi.shape, gti.shape)
                    # indices_s = quality_assessment(gti, yi, data_range=1., ratio=4)
                    # print(indices_s)
                    if i == 0:
                        indices = quality_assessment(gti, yi, data_range=1., ratio=4)
                    else:
                        indices = sum_dict(indices, quality_assessment(gti, yi, data_range=1., ratio=4))
                    output.append(yi)
                    test_number += 1
            print(i, test_number)
            print(gti.shape, 'gti')
        for index in indices:
            indices[index] = indices[index] / test_number
    save_dir = "./results" + model_title + '_' + str(time.ctime()) + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = './results/QIs/' + model_title + '_P' + str(P) + '_' + str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))


def save_checkpoint(args, model, epoch, eval_loss):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = args.ckp_dir
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model, "eval_loss": eval_loss}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()
