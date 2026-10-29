"""Training resumes from last.pth after a disconnect."""
import torch

from infravision.train.common import BestKeeper, Resumer


def _parts(out, resume):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    keeper = BestKeeper(out, "m", "val_psnr", meta={})
    return model, opt, keeper, Resumer(out, keeper, resume, model=model, opt=opt, missing=None)


def test_resume_continues_after_last_saved_epoch(tmp_path):
    model, opt, keeper, r = _parts(tmp_path, resume=True)
    assert r.start_epoch() == 1                     # nothing saved yet
    for epoch in (1, 2):
        model(torch.ones(1, 2)).sum().backward()
        opt.step()
        keeper.update(epoch, model.state_dict(), {"val_psnr": float(epoch)})
        r.end_epoch(epoch, 5)
    saved = model.weight.detach().clone()

    model2, _, keeper2, r2 = _parts(tmp_path, resume=True)     # "new Colab session"
    assert r2.start_epoch() == 3
    assert torch.equal(model2.weight, saved)
    assert [h["epoch"] for h in keeper2.history] == [1, 2]
    assert keeper2.best["epoch"] == 2
    r2.finish()
    assert (tmp_path / "DONE").exists()


def test_without_resume_starts_fresh(tmp_path):
    model, opt, keeper, r = _parts(tmp_path, resume=True)
    keeper.update(1, model.state_dict(), {"val_psnr": 1.0})
    r.end_epoch(1, 2)
    assert _parts(tmp_path, resume=False)[3].start_epoch() == 1
