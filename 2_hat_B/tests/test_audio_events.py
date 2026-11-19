from pathlib import Path
import csv
import numpy as np
import pytest
import torch
from src.audio_analysis.yamnet import Yamnet, logmel_patches, SameConv
from src.audio_analysis.run import summarize

@pytest.mark.parametrize('length,count', [(1,1),(15600,1),(15601,2),(23280,2)])
def test_patch_padding(length,count):
    _,patches=logmel_patches(np.zeros(length))
    assert patches.shape==(count,96,64)
    assert np.isfinite(patches).all()

@pytest.mark.parametrize('wave',[[],[np.nan],[[0,1]]])
def test_invalid_audio(wave):
    with pytest.raises(ValueError):logmel_patches(wave)


def test_tf_same_padding_is_asymmetric():
    conv=SameConv(np.ones((1,1,3,3),np.float32),np.zeros(1,np.float32),2)
    result=conv(torch.ones(1,1,4,4))
    assert result.tolist()==[[[[9.,6.],[6.,4.]]]]


def test_overlapping_sources_and_clip_bounds():
    wave=np.zeros(16000,np.float32)
    scores=np.array([[.8,.9],[.7,.8]])
    result=summarize(wave,scores,['Bark','Speech'])
    assert {event['class'] for event in result['events']}=={'Bark','Speech'}
    assert all(event['end']<=1 for event in result['events'])
    assert len(result['events'])==2
    assert result['duration']==1


def test_official_checkpoint_smoke():
    weights=Path('models/yamnet.h5')
    if not weights.exists():pytest.skip('Official local checkpoint is optional')
    torch.set_num_threads(2)
    model=Yamnet(weights)
    names=[r['display_name'] for r in csv.DictReader(open('configs/audio/yamnet_class_map.csv'))]
    rng=np.random.RandomState(51773)
    for wave,expected in [(np.zeros(48000),'Silence'),(rng.uniform(-1,1,48000),'White noise'),
                          (np.sin(2*np.pi*440*np.arange(48000)/16000),'Sine wave')]:
        scores=model.score(wave)
        assert scores.shape[1]==521
        assert expected in [names[i] for i in scores.mean(0).argsort()[-10:]]
