"""Old name of the `porygon` package, kept so checkpoints saved before the rename still load:
SB3 zips and pool snapshots pickle references such as `gloria.model.MaskedPolicy` and
`gloria.train.thesis_lr`, which resolve to the same objects in `porygon`."""

import importlib
import sys

for _name in ("encoder", "model", "train"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"porygon.{_name}")
