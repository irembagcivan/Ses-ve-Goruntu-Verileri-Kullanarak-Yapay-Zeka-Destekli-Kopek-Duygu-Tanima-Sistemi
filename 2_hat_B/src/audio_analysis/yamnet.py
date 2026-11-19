# Adapted from TensorFlow Authors' YAMNet, Copyright 2019, Apache-2.0.
# Official architecture/features: tensorflow/models/research/audioset/yamnet.
"""Inference-only PyTorch port reading the official non-executable HDF5 weights."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def logmel_patches(waveform):
    wave=np.asarray(waveform,dtype=np.float32)
    if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Expected a nonempty finite mono 16 kHz waveform')
    original=len(wave)
    target=15600+math.ceil(max(0,original-15600)/7680)*7680
    wave=np.pad(wave,(0,target-original))
    frames=np.lib.stride_tricks.sliding_window_view(wave,400)[::160]
    window=(.5-.5*np.cos(2*np.pi*np.arange(400)/400)).astype(np.float32)
    magnitude=np.abs(np.fft.rfft(frames*window,n=512,axis=1)).astype(np.float32)
    frequencies=np.linspace(0,8000,257,dtype=np.float32)
    mel=lambda hz:1127*np.log1p(np.asarray(hz)/700)
    edges=np.linspace(mel(125),mel(7500),66,dtype=np.float32)
    hzmel=mel(frequencies)[:,None]
    lower=(hzmel-edges[:-2])/(edges[1:-1]-edges[:-2])
    upper=(edges[2:]-hzmel)/(edges[2:]-edges[1:-1])
    weights=np.maximum(0,np.minimum(lower,upper)).astype(np.float32)
    weights[0]=0
    logmel=np.log(magnitude@weights+.001).astype(np.float32)
    patches=np.stack([logmel[start:start+96] for start in range(0,len(logmel)-95,48)])
    return logmel,patches


class SameConv(nn.Module):
    def __init__(self,weight,bias,stride,groups=1):
        super().__init__()
        self.register_buffer('weight',torch.from_numpy(np.ascontiguousarray(weight)))
        self.register_buffer('bias',torch.from_numpy(np.ascontiguousarray(bias)))
        self.stride,self.groups=stride,groups

    def forward(self,x):
        height,width=x.shape[-2:]
        kh,kw=self.weight.shape[-2:]
        ph=max(0,(math.ceil(height/self.stride)-1)*self.stride+kh-height)
        pw=max(0,(math.ceil(width/self.stride)-1)*self.stride+kw-width)
        x=F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2))
        return F.relu(F.conv2d(x,self.weight,self.bias,stride=self.stride,groups=self.groups))


class Yamnet(nn.Module):
    """Read all convolution/BN/dense arrays; never load pickle checkpoints."""
    def __init__(self,weights):
        super().__init__()
        import h5py
        path=Path(weights)
        if not path.is_file():
            raise FileNotFoundError('Local official yamnet.h5 is required')
        arrays={}
        with h5py.File(path,'r') as file:
            def collect(name,obj):
                if isinstance(obj,h5py.Dataset):arrays[name]=np.asarray(obj,dtype=np.float32)
            file.visititems(collect)
        def find(name):
            matches=[value for key,value in arrays.items() if key.endswith(name+':0') or key.endswith(name)]
            if len(matches)!=1:
                raise ValueError(f'Expected one HDF5 tensor for {name}; found {len(matches)}')
            return matches[0]
        layers=[]
        def conv(prefix,stride,depthwise=False):
            kernel=find(prefix+('/depthwise_kernel' if depthwise else '/kernel'))
            if depthwise:
                weight=kernel.transpose(2,3,0,1)
                groups=weight.shape[0]
            else:
                weight=kernel.transpose(3,2,0,1);groups=1
            beta=find(prefix+'/bn/beta')
            mean=find(prefix+'/bn/moving_mean')
            variance=find(prefix+'/bn/moving_variance')
            scale=1/np.sqrt(variance+1e-4)
            weight=weight*scale[:,None,None,None]
            bias=beta-mean*scale
            layers.append(SameConv(weight,bias,stride,groups))
        conv('layer1/conv',2)
        strides=[1,2,1,2,1,2,1,1,1,1,1,2,1]
        for number,stride in enumerate(strides,2):
            conv(f'layer{number}/depthwise_conv',stride,True)
            conv(f'layer{number}/pointwise_conv',1)
        self.layers=nn.Sequential(*layers)
        self.register_buffer('dense_weight',torch.from_numpy(np.ascontiguousarray(find('logits/kernel').T)))
        self.register_buffer('dense_bias',torch.from_numpy(find('logits/bias')))
        if self.dense_weight.shape!=(521,1024):
            raise ValueError('Unexpected YAMNet classifier shape')
        self.eval()

    def forward(self,patches):
        embedding=self.layers(patches[:,None]).mean(dim=(-2,-1))
        scores=torch.sigmoid(F.linear(embedding,self.dense_weight,self.dense_bias))
        return scores,embedding

    def score(self,waveform,batch_size=32):
        _,patches=logmel_patches(waveform)
        scores=[]
        with torch.inference_mode():
            for start in range(0,len(patches),batch_size):
                output,_=self(torch.from_numpy(patches[start:start+batch_size]))
                scores.append(output.numpy())
        return np.concatenate(scores)
