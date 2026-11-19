"""Controlled LSTM/GRU experiment inspired by doi:10.3389/fvets.2026.1799281.

An adaptation on existing pose/YAMNet sequences, NOT a QuadEmoNet reproduction.
Run from repository root; no raw videos or new dependencies required.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from .sequence import scale_fit, scale_apply
from .train import TARGETS, seed, weights_for, losses, scores

SOURCE = 'https://doi.org/10.3389/fvets.2026.1799281'
KINDS = ('gru', 'lstm', 'lstm_lstm_gru')
EPOCHS = (10, 20, 40)


class HybridNet(nn.Module):
    def __init__(self, sizes, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.visual = nn.Sequential(nn.Linear(117, 32), nn.LayerNorm(32), nn.GELU())
        self.audio = nn.Sequential(nn.Linear(1024, 32), nn.LayerNorm(32), nn.GELU())
        cells = [nn.GRU(64, 32, batch_first=True)] if kind == 'gru' else [nn.LSTM(64, 32, batch_first=True)]
        if kind == 'lstm_lstm_gru':
            cells += [nn.LSTM(32, 32, batch_first=True), nn.GRU(32, 32, batch_first=True)]
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(.35)
        self.heads = nn.ModuleList(nn.Linear(32, size) for size in sizes)

    def forward(self, x, lengths):
        features = torch.cat([self.visual(x[:, :, :117]), self.audio(x[:, :, 117:])], -1)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for i, cell in enumerate(self.cells):
            packed, state = cell(packed)
            if i < len(self.cells) - 1:
                packed = PackedSequence(self.dropout(packed.data), packed.batch_sizes,
                                        packed.sorted_indices, packed.unsorted_indices)
        hidden = state[0][-1] if isinstance(state, tuple) else state[-1]
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.heads]


def fit(x, lengths, labels, fit_indices, val_indices, sizes, kind, epochs):
    seed()
    stats = scale_fit(x, lengths, fit_indices)
    xt = scale_apply(x, lengths, stats)
    lt, yt = torch.tensor(lengths), torch.tensor(labels)
    model = HybridNet(sizes, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    weights = weights_for(labels[fit_indices], sizes)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = losses(model(xt[fit_indices], lt[fit_indices]), yt[fit_indices], weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if val_indices is not None and epoch in EPOCHS:
            model.eval()
            with torch.inference_mode():
                values = scores(model(xt[val_indices], lt[val_indices]), labels[val_indices], sizes)
            history.append(dict(epoch=epoch, train_loss=float(loss.detach()),
                                validation_mean_f1=float(np.mean(values))))
    model.eval()
    return model, stats, history, xt, lt


def main(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.snapshot, allow_pickle=False) as data:
        x, lengths, labels, train, test = [data[k].copy() for k in
            ('sequences', 'lengths', 'labels', 'train_indices', 'test_indices')]
    if set(train) & set(test):
        raise ValueError('Overlapping train/test indices')
    metadata = json.loads(args.metadata.read_text())
    sizes = metadata['sizes']
    # Same frozen training sample order/folds as sequence.py. Each snapshot row
    # represents a unique source hash; verify this against manifest explicitly.
    import pandas as pd
    manifest = pd.read_csv(args.manifest).fillna('')
    hashes = manifest.source_sha256.to_numpy()
    if len(manifest) != len(x) or len(set(hashes)) != len(hashes):
        raise ValueError('Snapshot must align with unique source manifest')
    if not np.array_equal(train, np.flatnonzero(manifest.split.eq('train'))) or not np.array_equal(test, np.flatnonzero(manifest.split.eq('test'))):
        raise ValueError('Manifest/snapshot split mismatch')
    for t, target in enumerate(TARGETS):
        expected = np.array([metadata['classes'][target].index(v) if v in metadata['classes'][target] else -1 for v in manifest[target]])
        if not np.array_equal(expected, labels[:, t]):
            raise ValueError('Manifest/snapshot label mismatch')
    folds = [(train[a], train[b]) for a, b in GroupKFold(3).split(train, groups=hashes[train])]
    selection, traces = {}, {}
    for kind in KINDS:
        traces[kind] = []
        for fold, (a, b) in enumerate(folds):
            _, _, trace, _, _ = fit(x, lengths, labels, a, b, sizes, kind, 40)
            traces[kind].append(trace)
            print(f'{kind}: CV fold {fold + 1}/3', flush=True)
        values = {e: float(np.mean([next(v['validation_mean_f1'] for v in h if v['epoch'] == e) for h in traces[kind]])) for e in EPOCHS}
        epoch = min(values, key=lambda e: (-values[e], e))
        selection[kind] = dict(epochs=epoch, cv_mean_f1=values[epoch], candidates=values)
    selected = max(KINDS, key=lambda k: selection[k]['cv_mean_f1'])
    # Persist every choice BEFORE accessing test metrics.
    (args.output / 'selection.json').write_text(json.dumps(dict(models=selection, selected_by_training_cv=selected), indent=2))
    results = []
    for kind in KINDS:
        model, stats, _, xt, lt = fit(x, lengths, labels, train, None, sizes, kind, selection[kind]['epochs'])
        with torch.inference_mode():
            output = model(xt[test], lt[test])
        values = scores(output, labels[test], sizes)
        predictions = [o.argmax(1).tolist() for o in output]
        torch.save(model.state_dict(), args.output / (kind + '.pt'))
        (args.output / (kind + '.json')).write_text(json.dumps(dict(kind=kind, scaler=stats, classes=metadata['classes'], sizes=sizes)))
        results.append(dict(kind=kind, mean_test_macro_f1=float(np.mean(values)),
                            per_target_macro_f1=dict(zip(TARGETS, values)),
                            predictions_by_target=predictions, parameters=sum(p.numel() for p in model.parameters())))
        print(results[-1]['kind'], results[-1]['mean_test_macro_f1'], flush=True)
    result = dict(source=SOURCE, adaptation='32-unit pose/YAMNet hybrid; not MFCC QuadEmoNet reproduction',
                  snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                  seed=42, train_count=len(train), test_count=len(test), selected_by_training_cv=selected,
                  protocol='Same 1 Hz first-60s features/folds; train-only scaling and epoch/model selection; single seed; previously inspected holdout; behavior not emotion labels',
                  selection=selection, histories=traces, results=results)
    (args.output / 'experiment.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('results/demo_v11/sequence_snapshot.npz'))
    parser.add_argument('--metadata', type=Path, default=Path('results/demo_v11/sequence_gru.json'))
    parser.add_argument('--manifest', type=Path, default=Path('results/demo_v4/manifest.csv'))
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
