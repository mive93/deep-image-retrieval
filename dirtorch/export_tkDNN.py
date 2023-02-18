import sys
import os
import os.path as osp
import pdb
import datetime

import json
from turtle import shape
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader

import dirtorch.test_dir as test
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl

from torchsummary import summary

import pickle as pkl
import hashlib
import struct
def bin_write(f, data):
    data =data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)


###############################################################################
################################### TKDNN #####################################
###############################################################################


def create_folders():
    if not os.path.exists('debug'):
        os.makedirs('debug')
    if not os.path.exists('layers'):
        os.makedirs('layers')

def hook(module, input, output):
    setattr(module, "_value_hook", output)

def exp_input(input_batch):
    i = input_batch.cpu().data.numpy()
    i = np.array(i, dtype=np.float32)
    i.tofile("debug/input.bin", format="f")
    print("input: ", i.shape)

def print_wb_output(model, first_different=False):
    f = None
    count = 0
    for n, m in model.named_modules():
        print(n)
        if count == 0 and first_different:
            in_output = m._value_hook[1]
        else:
            in_output = m._value_hook
        count = count +1

        # print(in_output)
        o = in_output.cpu().data.numpy()
        o = np.array(o, dtype=np.float32)
        t = '-'.join(n.split('.'))
        o.tofile("debug/" + t + ".bin", format="f")
        print('------- ', n, ' ------') 
        print("debug  ",o.shape)
        
        if not(' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type)):
            continue
        
        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type):
            file_name = "layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')

        w = np.array([])
        b = np.array([])
        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            w = m._parameters['weight'].cpu().data.numpy()
            w = np.array(w, dtype=np.float32)
            print ("    weights shape:", np.shape(w))
        
        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            print ("    bias shape:", np.shape(b))
        
        if 'BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].cpu().data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.cpu().data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.cpu().data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f,b)
            bin_write(f,s)
            bin_write(f,rm)
            bin_write(f,rv)
            print ("    b shape:", np.shape(b))
            print ("    s shape:", np.shape(s))
            print ("    rm shape:", np.shape(rm))
            print ("    rv shape:", np.shape(rv))

        else:
            bin_write(f,w)
            if b.size > 0:
                bin_write(f,b)

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            f.close()
            print("close file")
            f = None

def inference(model, batch, export=False):
    model.eval()
    with torch.no_grad():
        start = datetime.datetime.now()
        if export:
            print('Exporting for tkDNN')
            # create folders debug and layers if do not exist
            create_folders()

            # add output attribute to the layers
            for n, m in model.named_modules():
                m.register_forward_hook(hook)

        desc = model(batch)

        if export:
            # export input bin
            exp_input(batch)

            print_wb_output(model, True)

            with open("Resnet-101-AP-GeM.txt", 'w') as f:
                for item in list(model.children()):
                    f.write("%s\n" % item)

            summary(model, (3, 768, 1024))

        end = datetime.datetime.now()
        delta = end - start
        delta_ms = int(delta.total_seconds() * 1000) # milliseconds
        print('inference time: ', delta_ms, ' ms ')
        return desc

###############################################################################
################################### TKDNN #####################################
###############################################################################

def extract_features(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
                     threads=8, batch_size=16, output=None, dbg=()):
    """ Extract features from trained model (network) on a given dataset.
    """
    print("\n>> Extracting features...")
    try:
        query_db = db.get_query_db()
    except NotImplementedError:
        query_db = None

    print("QUERY DB: ", query_db)

    # extract DB feats
    bdescs = []
    qdescs = []

    trfs_list = [trfs] if isinstance(trfs, str) else trfs

    for trfs in trfs_list:
        kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
        bdescs.append(test.extract_image_features(db, trfs, net, desc="DB", **kw))

        # extract query feats
        if query_db is not None:
            qdescs.append(bdescs[-1] if db is query_db
                          else test.extract_image_features(query_db, trfs, net, desc="query", **kw))
    
    files = bdescs[0][0]
    bdescs = [bdescs[0][1]]
    
    # pool from multiple transforms (scales)
    bdescs = tonumpy(F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1))
    if query_db is not None:
        qdescs = tonumpy(F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1))

    if whiten is not None:
        bdescs = common.whiten_features(bdescs, net.pca, **whiten)
        if query_db is not None:
            qdescs = common.whiten_features(qdescs, net.pca, **whiten)

    for i in range(len(files)):
        out_file = files[i][0] + ".emb"
        of = open(out_file, 'wb')

        d = bdescs[i, :]
        bin_write(of, d)


    mkdir(output, isfile=True)
    if query_db is db or query_db is None:
        np.save(output, bdescs)
    else:
        o = osp.splitext(output)
        np.save(o[0]+'.qdescs'+o[1], qdescs)
        np.save(o[0]+'.dbdescs'+o[1], bdescs)
    print('Features extracted.')


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--output', type=str, default="", help='path to output features')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')

    parser.add_argument('--whitenp', type=float, default=0.5, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)

    dataset = datasets.create(args.dataset)
    print("Dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda)

    batch_size = 8
    print(net.preprocess)
    
    loader = get_loader(dataset, trf_chain=args.trfs, preprocess=net.preprocess, iscuda=True,
                        output=['img', 'img_filename'], batch_size=batch_size, threads=args.threads, shuffle=False)

    torch.set_printoptions(profile="full")
    
    print(net.preprocess)
    


    with torch.no_grad():
        for inputs in tqdm.tqdm(loader, "Testing model", total=1+(len(dataset)-1)//batch_size):
            imgs = inputs[0]
            
            
            imgs = common.variables(inputs[:1], net.iscuda)[0]
            # print(imgs)
            
            desc_tkDDN = inference(net, imgs, False)
            desc = net(imgs)
            
            print(desc, desc.shape)
            input('')
            # print(desc_tkDDN, desc_tkDDN.shape)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    exit(0)

    # Evaluate
    res = extract_features(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                           threads=args.threads, dbg=args.dbg, whiten=args.whiten, output=args.output)


