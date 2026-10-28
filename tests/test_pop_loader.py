import pytest
import torch

from infravision.data.pop import (POPDetection, POPSRCrops, check_no_leakage, flight_of, pop_splits,
                                  summarize)


def test_flight_parsing():
    assert flight_of("F13_0145_50.101.JPG") == "F13"
    assert flight_of("POP/test/JPEGImages/T2_0001_50.804.JPG") == "T2"


def test_official_splits_are_by_flight(pop_root):
    data = pop_splits(pop_root)
    assert {r["flight"] for r in data["train"]} == {"F1", "F2"}
    assert {r["flight"] for r in data["val"]} == {"F13", "F14"}
    assert {r["flight"] for r in data["test"]} == {"T1", "T2"}
    s = summarize(data)
    assert s["train"]["images"] == 6 and s["train"]["people"] == 6


def test_resplit_is_seeded_and_never_mixes_flights(pop_root):
    a = pop_splits(pop_root, resplit=True, val_flights=1, seed=3)
    b = pop_splits(pop_root, resplit=True, val_flights=1, seed=3)
    assert [r["name"] for r in a["val"]] == [r["name"] for r in b["val"]]
    assert len({r["flight"] for r in a["val"]}) == 1
    assert {r["flight"] for r in a["test"]} == {"T1", "T2"}   # test never re-drawn
    check_no_leakage(a)


def test_leakage_detected():
    with pytest.raises(AssertionError):
        check_no_leakage({"train": [{"flight": "F1"}], "val": [{"flight": "F1"}]})


def test_detection_dataset_shapes(pop_root):
    ds = POPDetection("val", root=pop_root)
    img, boxes, name = ds[0]
    assert img.shape == (256, 320) and img.dtype == torch.uint8
    assert boxes.shape == (1, 4) and (boxes[:, 2:] > boxes[:, :2]).all()


def test_sr_crops_shapes(pop_root):
    recs = pop_splits(pop_root)["train"]
    ds = POPSRCrops(recs, size=64, scale=2, per_image=2, train=False)
    x, y = ds[0]
    assert len(ds) == 12
    assert x.shape == y.shape == (1, 64, 64)
    assert 0 <= x.min() and x.max() <= 1
    x2, _ = ds[0]
    assert torch.equal(x, x2)   # val crops are fixed
