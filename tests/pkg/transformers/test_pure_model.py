# python tests/model.py
from pathlib import Path

import pytest
import rich
from core.opt import CfgNode as CN
from core.opt import Opts
from pkg.datasets import DATASET_REGISTRY
from pkg.metrics import METRIC_REGISTRY
from pkg.models import MODEL_REGISTRY
from tqdm import tqdm


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "model_name",
    [
        "Segformer",
    ],
)
def test_load_model(model_name):
    cfg = CN(
        {
            "model": {
                "name": model_name,
                "args": {
                    "NUM_CLASS": 2,
                },
            }
        }
    )

    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "model_name, IMG_SIZE",
    [
        ("Segformer", 380),
        ("Mask2former", 380),
    ],
)
def test_pure_model(model_name, IMG_SIZE):
    cfg = CN(
        {
            "data": {
                "name": "SegDataset",
                "args": {
                    "SIZE": IMG_SIZE,  # REQUIRED
                    "train": {
                        "root_dir": "data",
                        "phase": "warmup",
                        "split": "valid",
                        "loader": {
                            "batch_size": 2,
                            "num_workers": 2,
                        },
                    },
                    "val": {
                        "root_dir": "data",
                        "phase": "warmup",
                        "split": "valid",
                        "loader": {
                            "batch_size": 2,
                            "num_workers": 2,
                            "shuffle": False,
                            "drop_last": False,
                        },
                    },
                },
            },
            "model": {
                "name": model_name,
                "args": {
                    "NUM_CLASS": 2,
                },
            },
            "metric": [],
        }
    )
    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.setup("fit")
    dl = model.train_dataloader()
    for batch in dl:
        output = model(batch)  # test fwd
        break


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "model_name, IMG_SIZE",
    [
        ("Segformer", 380),
        ("Mask2former", 380),
    ],
)
def test_pure_model_eval(model_name, IMG_SIZE, bs=2):
    cfg = CN(
        {
            "data": {
                "name": "SegDataset",
                "args": {
                    "SIZE": IMG_SIZE,  # REQUIRED
                    "train": {
                        "root_dir": "data",
                        "phase": "warmup",
                        "split": "valid",
                        "loader": {
                            "batch_size": bs,
                            "num_workers": 2,
                        },
                    },
                    "val": {
                        "root_dir": "data",
                        "phase": "warmup",
                        "split": "valid",
                        "loader": {
                            "batch_size": bs,
                            "num_workers": 2,
                            "shuffle": False,
                            "drop_last": False,
                        },
                    },
                },
            },
            "model": {
                "name": model_name,
                "args": {
                    "NUM_CLASS": 2,
                },
            },
            "metric": [
                {"name": "SMAPIoUMetricWrapper", "args": {"label_key": "labels"}}
            ],
        }
    )
    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.setup("fit")
    dl = model.train_dataloader()

    metrics = [
        METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
        if mcfg["args"]
        else METRIC_REGISTRY.get(mcfg["name"])()
        for mcfg in cfg["metric"]
    ]

    for i, batch in tqdm(enumerate(dl), total=5):
        out = model(batch)
        for metric in metrics:
            metric.update(out, batch)

        if (i % 5 == 0) and (i > 0):
            for metric in metrics:
                metric_dict = metric.value()
                # Log string
                log_string = ""
                for metric_name, score in metric_dict.items():
                    if isinstance(score, (int, float)):
                        log_string += metric_name + ": " + f"{score:.5f}" + " | "
                log_string += "\n"
                rich.print(log_string)

                # 4. Reset metric
                metric.reset()
            break


# if __name__ == "__main__":
#     test_pure_model_eval(model_name="Beit", bs=2)
