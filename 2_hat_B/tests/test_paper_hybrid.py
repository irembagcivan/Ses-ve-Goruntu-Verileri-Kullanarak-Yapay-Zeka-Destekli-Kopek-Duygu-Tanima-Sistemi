import pytest
import torch

from src.deep_learning.paper_hybrid import HybridNet, KINDS


@pytest.mark.parametrize('kind', KINDS)
def test_padding_and_batch_order_do_not_change_predictions(kind):
    torch.manual_seed(42)
    model = HybridNet([4, 3, 4, 4, 3], kind).eval()
    x = torch.randn(3, 7, 1141)
    lengths = torch.tensor([2, 7, 4])
    with torch.inference_mode():
        batch = model(x, lengths)
        changed = x.clone()
        for i, length in enumerate(lengths):
            changed[i, length:] = 10000
        padded = model(changed, lengths)
        for i, length in enumerate(lengths):
            single = model(x[i:i+1, :length], lengths[i:i+1])
            for b, p, s in zip(batch, padded, single):
                torch.testing.assert_close(b, p)
                torch.testing.assert_close(b[i], s[0], atol=1e-6, rtol=1e-5)


def test_hybrid_backpropagates_through_all_recurrent_layers():
    model = HybridNet([4, 3, 4, 4, 3], 'lstm_lstm_gru')
    output = model(torch.randn(3, 5, 1141), torch.tensor([5, 2, 4]))
    sum(o.square().mean() for o in output).backward()
    for cell in model.cells:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
        assert sum(p.grad.abs().sum() for p in cell.parameters()) > 0
