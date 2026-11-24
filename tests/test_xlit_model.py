import pytest

torch = pytest.importorskip("torch")

from models.xlit.model import EOS, PAD, XlitTransformer, beam_search, score_targets  # noqa: E402


def test_score_targets_matches_beam_scores():
    torch.manual_seed(0)
    model = XlitTransformer(12, 10, d_model=32, nhead=2, layers=1, ff=64).eval()
    with torch.no_grad():
        model.out.bias[EOS] += 1.5  # make beams finish so there is something to compare
    src = torch.tensor([[4, 5, 6, EOS], [7, 8, EOS, PAD]])
    hyps = beam_search(model, src, beam=3, max_len=8)
    checked = 0
    for b, h in enumerate(hyps):
        done = [(ids, s) for ids, s in h if ids and ids[-1] == EOS]
        if not done:
            continue
        rows = src[[b] * len(done)]
        got = score_targets(model, rows, [ids for ids, _ in done])
        for (_, want), g in zip(done, got, strict=True):
            assert g == pytest.approx(want, abs=1e-4)
            checked += 1
    assert checked > 0
