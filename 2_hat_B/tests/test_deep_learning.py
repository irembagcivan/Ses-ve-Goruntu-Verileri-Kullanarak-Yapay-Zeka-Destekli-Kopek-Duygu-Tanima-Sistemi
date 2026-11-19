import io
import numpy as np
import torch
from src.deep_learning.train import FusionNet,normalize_fit,normalize_apply,losses,weights_for,seed


def test_fusion_both_branches_and_five_heads_receive_gradients():
    seed();model=FusionNet(6,8,[3,2,4,3,2]);y=torch.tensor([[0,1,2,-1,0],[1,0,1,2,1]])
    outputs=model(torch.randn(2,6),torch.randn(2,8));loss=losses(outputs,y,weights_for(y.numpy(),[3,2,4,3,2]));loss.backward()
    assert model.visual[0].weight.grad.abs().sum()>0
    assert model.audio[0].weight.grad.abs().sum()>0
    assert all(head.weight.grad is not None for head in model.heads)
    assert torch.isfinite(loss)


def test_fold_scaler_uses_only_supplied_training_values_and_handles_empty_column():
    train=np.array([[1,np.nan],[3,np.nan]])
    stats=normalize_fit(train)
    assert stats['mean']==[2.,0.]
    transformed=normalize_apply([[1000,np.nan]],stats)
    assert transformed.tolist()==[[10.,0.]]
    assert stats['mean']==[2.,0.]


def test_checkpoint_roundtrip_and_inactive_branch():
    seed();model=FusionNet(6,8,[3,2,4,3,2],'visual');model.eval()
    assert not any(p.requires_grad for p in model.audio.parameters())
    visual=torch.randn(2,6);audio=torch.randn(2,8)
    before=model(visual,audio)
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=FusionNet(6,8,[3,2,4,3,2],'visual');restored.load_state_dict(torch.load(stream,weights_only=True));restored.eval()
    for a,b in zip(before,restored(visual,audio)):assert torch.equal(a,b)
    for a,b in zip(before,model(visual,audio*100)):assert torch.equal(a,b)
