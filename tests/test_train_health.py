"""Training health checks stop broken Kaggle runs early."""
import pytest

from infravision.train.common import LossWatch


def test_loss_watch_stops_after_nan_streak_only():
    w = LossWatch(limit=3)
    for loss in [1.0, float("nan"), float("inf"), 0.5, float("nan"), float("nan")]:
        w(loss)                                    # streaks of 2 are tolerated
    with pytest.raises(SystemExit, match="HEALTH CHECK FAILED"):
        w(float("nan"))
