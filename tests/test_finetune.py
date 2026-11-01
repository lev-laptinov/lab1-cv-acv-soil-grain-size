import numpy as np
import pandas as pd
import torch

from soilgsd.finetune import CurveModel, Trainer, augment, emd_loss
from soilgsd.metric import emd, is_valid_curve


def test_emd_loss_is_the_competition_metric():
    rng = np.random.default_rng(0)
    y = np.sort(rng.uniform(0, 100, (4, 11)), axis=1)
    p = np.sort(rng.uniform(0, 100, (4, 11)), axis=1)
    y[:, -1] = p[:, -1] = 100
    assert np.isclose(emd_loss(torch.tensor(p), torch.tensor(y)).item(), emd(y, p))


def test_model_output_is_always_a_valid_curve():
    model = CurveModel("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 3, 64, 64) * 5).numpy()
    assert is_valid_curve(out, atol=1e-3).all()


def test_augment_keeps_shape_range_and_content_scale():
    x = torch.rand(8, 3, 32, 32)
    a = augment(x.clone())
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    # Rotations and flips only permute pixels: the sorted values change by the colour jitter alone.
    assert torch.allclose(a.mean(dim=(1, 2, 3)), x.mean(dim=(1, 2, 3)), atol=0.15)


def test_lpft_trains_head_first_then_the_whole_model():
    exp = {"backbone": "resnet18", "pretrained": False, "batch_size": 4, "epochs": 2, "head_only_epochs": 1,
           "opt": "adamw", "lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 0}
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8)
    y = np.tile(np.linspace(5, 100, 11), (8, 1))
    trainer = Trainer(exp, torch.device("cpu"), seed=0)
    start = [p.detach().clone() for p in trainer.model.backbone.parameters()]
    log = trainer.fit(tiles, y, pd.Series(list("aabbccdd")))
    assert [row["epoch"] for row in log] == [0, 1]
    moved = [not torch.equal(a, b) for a, b in zip(start, trainer.model.backbone.parameters(), strict=True)]
    assert any(moved), "the backbone never trained after the linear-probe phase"


def _bag_exp(**kw):
    return {"backbone": "resnet18", "pretrained": False, "bag_samples": 2, "bag_tiles": 3, "epochs": 1,
            "opt": "adamw", "lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 0, **kw}


def test_bag_batch_takes_whole_bags_of_one_sample_each():
    sample_ids = pd.Series(["a"] * 6 + ["b"] * 4 + ["c"] * 8)
    photo_ids = pd.Series(["a1"] * 5 + ["a2"] + ["b1"] * 4 + ["c1"] * 4 + ["c2"] * 4)
    y = np.repeat(np.linspace(0, 100, 3)[:, None], 11, axis=1)[[0] * 6 + [1] * 4 + [2] * 8]
    trainer = Trainer(_bag_exp(), torch.device("cpu"), seed=0)
    bags = trainer._bags(sample_ids, photo_ids)
    # Each photo of a sample weighs the same: the single a2 tile is as likely as all five a1 tiles together.
    ix, w = bags["a"]
    assert np.isclose(w[ix == 5].sum(), 0.5)
    idx, yb = trainer._bag_batch(bags, y)
    assert idx.shape == (6,) and yb.shape == (2, 11)
    for bag, label in zip(idx.reshape(2, 3), yb, strict=True):
        assert sample_ids[bag].nunique() == 1 and len(set(bag)) == 3
        assert np.array_equal(y[bag[0]], label)


def test_bag_model_trains_and_predicts_one_valid_curve_per_sample():
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 255, (12, 32, 32, 3), dtype=np.uint8)
    meta = pd.DataFrame({"sample_id": list("aaaabbbbcccc"), "path": list("ppqqrrrrsstt")})
    y = np.tile(np.linspace(5, 100, 11), (12, 1))
    trainer = Trainer(_bag_exp(), torch.device("cpu"), seed=0)
    log = trainer.fit(tiles, y, meta["sample_id"], photo_ids=meta["path"])
    assert len(log) == 1 and np.isfinite(log[0]["train_emd"])
    pred = trainer.predict_samples(tiles, meta)
    assert list(pred.index) == ["a", "b", "c"]
    assert is_valid_curve(pred.to_numpy(), atol=1e-3).all()


def test_forward_bags_pools_features_not_curves():
    model = CurveModel("resnet18", pretrained=False).eval()
    x = torch.randn(6, 3, 32, 32)
    with torch.no_grad():
        pooled = model.forward_bags(x, 3)
        manual = model.curve(model.backbone(x).view(2, 3, -1).mean(1))
    assert torch.allclose(pooled, manual, atol=1e-5)


def test_hf_weights_load_by_name_or_by_position(tmp_path, monkeypatch):
    import huggingface_hub

    from soilgsd.finetune import load_hf_weights

    src = CurveModel("resnet18", pretrained=False).backbone
    sd = {k: v for k, v in src.state_dict().items() if not k.endswith("num_batches_tracked")}
    by_name = tmp_path / "a.pth"
    torch.save({**sd, "fc.weight": torch.zeros(3, 512)}, by_name)
    # Same tensors in the same order under foreign names (as efficientnet_pytorch files store them)
    by_pos = tmp_path / "b.pth"
    torch.save({f"_x{i}": v for i, v in enumerate(sd.values())}, by_pos)
    files = {"a.pth": by_name, "b.pth": by_pos}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, fname: str(files[fname]))
    for fname in files:
        dst = CurveModel("resnet18", pretrained=False).backbone
        load_hf_weights(dst, f"user/repo/{fname}")
        assert all(torch.equal(dst.state_dict()[k], v) for k, v in sd.items())


def _bag_data():
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 255, (12, 32, 32, 3), dtype=np.uint8)
    meta = pd.DataFrame({"sample_id": list("aaaabbbbcccc"), "path": list("ppqqrrrrsstt")})
    return tiles, meta, np.tile(np.linspace(5, 100, 11), (12, 1))


def test_groupnorm_option_replaces_every_batchnorm_and_still_trains():
    tiles, meta, y = _bag_data()
    trainer = Trainer(_bag_exp(bn="group"), torch.device("cpu"), seed=0)
    mods = list(trainer.model.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in mods)
    assert any(isinstance(m, torch.nn.GroupNorm) for m in mods)
    log = trainer.fit(tiles, y, meta["sample_id"], photo_ids=meta["path"])
    assert np.isfinite(log[0]["train_emd"])
    assert is_valid_curve(trainer.predict_samples(tiles, meta).to_numpy(), atol=1e-3).all()


def test_train_layers_keeps_the_rest_of_the_backbone_fixed():
    tiles, meta, y = _bag_data()
    trainer = Trainer(_bag_exp(train_layers=["layer4"]), torch.device("cpu"), seed=0)
    before = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}
    trainer.fit(tiles, y, meta["sample_id"], photo_ids=meta["path"])
    after = dict(trainer.model.named_parameters())
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and all(k.startswith(("backbone.layer4.", "head.")) for k in moved)
    assert any(k.startswith("head.") for k in moved) and any(k.startswith("backbone.layer4.") for k in moved)


def test_strong_colour_and_sharpness_augment_stay_in_range_and_keep_brightness():
    torch.manual_seed(0)
    x = torch.rand(16, 3, 32, 32) * 0.5 + 0.25
    a = augment(x.clone(), colour="strong", sharpness=True)
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.allclose(a.mean(dim=(1, 2, 3)), x.mean(dim=(1, 2, 3)), atol=0.25)


def test_sharpness_jitter_only_blurs_or_sharpens():
    from soilgsd.finetune import sharpness_jitter

    torch.manual_seed(0)
    x = torch.rand(64, 3, 32, 32)
    out = sharpness_jitter(x)
    detail = lambda t: (t[..., 1:, :] - t[..., :-1, :]).abs().mean(dim=(1, 2, 3))  # noqa: E731
    changed = ~torch.isclose(detail(out), detail(x))
    assert changed.any() and (~changed).any()  # some tiles untouched, some blurred or sharpened
    assert torch.allclose(out.mean(dim=(1, 2, 3)), x.mean(dim=(1, 2, 3)), atol=0.02)


def test_grayscale_option_applies_in_training_and_in_prediction():
    tiles, meta, y = _bag_data()
    trainer = Trainer(_bag_exp(grayscale=True), torch.device("cpu"), seed=0)
    for train in (True, False):
        x = trainer._input(tiles, train=train) * trainer.std + trainer.mean  # undo the normalisation
        assert torch.allclose(x[:, 0], x[:, 1], atol=1e-5) and torch.allclose(x[:, 1], x[:, 2], atol=1e-5)


def test_exclude_train_drops_a_sample_from_training_but_still_scores_it(monkeypatch):
    from soilgsd.finetune import cross_validate

    tiles, meta, _ = _bag_data()
    labels = pd.DataFrame({"sample_id": list("abc"), **{c: np.linspace(5, 100, 11)[i] for i, c in
                                                      enumerate(["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                                                                 "2", "6.3", "20", "63", "200"])}})
    folds = pd.DataFrame({"sample_id": list("abc"), "repeat": 0, "fold": [0, 1, 2]})
    seen = []
    orig = Trainer.fit
    monkeypatch.setattr(Trainer, "fit", lambda self, t, y, s, **k: seen.append(set(s)) or orig(self, t, y, s, **k))
    exp = _bag_exp(bag_samples=1, exclude_train=["a"])
    oof, _ = cross_validate(exp, tiles, meta, labels, folds, torch.device("cpu"), seed=0)
    assert all("a" not in s for s in seen)
    assert set(oof["sample_id"]) == {"a", "b", "c"}


def test_device_filters_train_on_one_phone_and_validate_on_another(monkeypatch):
    from soilgsd.finetune import cross_validate

    tiles, meta, _ = _bag_data()
    meta = meta.assign(device=["M", "M", "S", "S"] * 3)  # every sample has photos from both phones
    labels = pd.DataFrame({"sample_id": list("abc"), **{c: np.linspace(5, 100, 11)[i] for i, c in
                                                      enumerate(["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                                                                 "2", "6.3", "20", "63", "200"])}})
    folds = pd.DataFrame({"sample_id": list("abc"), "repeat": 0, "fold": [0, 1, 2]})
    seen_paths, val_paths = [], []
    orig_fit, orig_pred = Trainer.fit, Trainer.predict_samples
    monkeypatch.setattr(Trainer, "fit", lambda self, t, y, s, **k: seen_paths.append(set(k["photo_ids"]))
                        or orig_fit(self, t, y, s, **k))
    monkeypatch.setattr(Trainer, "predict_samples", lambda self, t, m: val_paths.append(set(m["path"]))
                        or orig_pred(self, t, m))
    exp = _bag_exp(bag_samples=1, train_devices=["M"], val_devices=["S"])
    oof, _ = cross_validate(exp, tiles, meta, labels, folds, torch.device("cpu"), seed=0)
    m_paths, s_paths = set(meta.path[meta.device == "M"]), set(meta.path[meta.device == "S"])
    assert all(p <= m_paths for p in seen_paths) and all(p <= s_paths for p in val_paths)
    assert set(oof["sample_id"]) == {"a", "b", "c"}
