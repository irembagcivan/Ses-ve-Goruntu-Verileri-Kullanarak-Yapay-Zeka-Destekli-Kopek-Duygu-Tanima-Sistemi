"""Replay the selected neural head from saved features without raw videos."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .train import train_one


def main(args):
    torch.set_num_threads(2);meta=json.loads(args.metadata.read_text())
    with np.load(args.snapshot,allow_pickle=False) as data:
        v,a,labels,train=data['visual'],data['audio'],data['labels'],data['train_indices']
    epochs=meta['protocol']['epoch_choices'][meta['modality']]['epochs']
    model,*_=train_one(v,a,labels,train,np.array([],int),meta['class_counts'],meta['modality'],epochs,False)
    original=torch.load(args.reference,weights_only=True,map_location='cpu')
    exact=all(torch.equal(original[key],model.state_dict()[key]) for key in original)
    if args.output.exists():raise FileExistsError('Choose a new checkpoint output')
    torch.save(model.state_dict(),args.output)
    print(json.dumps({'replayed_epochs':epochs,'all_parameter_tensors_equal':exact,'train_rows':len(train),'test_used_for_fit':False}))
    if not exact:raise RuntimeError('Replay differs from reference checkpoint')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot','metadata','reference','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
