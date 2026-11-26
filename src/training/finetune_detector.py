"""Fine-tune a YOLO detector on a drone dataset.

Run:  python -m src.training.prepare_drone_dataset          (build data/drone_dataset.yaml first)
      python -m src.training.finetune_detector --base weights/yolov8n.pt --imgsz 640 --name fast
      python -m src.training.finetune_detector --base weights/yolov8s.pt --imgsz 1280 --name robust

The best checkpoint is copied to weights/drone_<name>.pt. Point config.yaml's
modes.fast.weights / modes.robust.weights at it to use it in the pipeline.
"""

import argparse
import shutil
from pathlib import Path
from typing import Any, Optional, Union


def finetune_detector(
    data_yaml: Union[str, Path] = "data/drone_dataset.yaml",
    base_weights: str = "weights/yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    name: str = "fast",
    device: Optional[str] = None,
    workers: int = 4,
    resume: bool = False,
) -> Path:
    """Train ``base_weights`` on ``data_yaml`` and return the path of the copied best weights."""
    from ultralytics import YOLO

    run_dir = Path("outputs/detector") / name
    if resume:
        # Continue an interrupted run with its saved epoch, optimizer state and settings.
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"No checkpoint to resume at {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        if not Path(data_yaml).is_file():
            raise FileNotFoundError(f"{data_yaml} not found. Run: python -m src.training.prepare_drone_dataset")
        model = YOLO(str(base_weights))
        kwargs: dict[str, Any] = dict(
            data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
            project=str(Path("outputs/detector").resolve()), name=name, exist_ok=True,
            patience=10, plots=True,
        )
        if device is not None:
            kwargs["device"] = device
        model.train(**kwargs)

    best = Path(model.trainer.best)
    dst = Path("weights") / f"drone_{name}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    print(f"Best weights -> {dst}")
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/drone_dataset.yaml")
    p.add_argument("--base", default="weights/yolov8n.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="-1 lets Ultralytics pick for the GPU memory")
    p.add_argument("--name", default="fast", help="run name; weights saved as weights/drone_<name>.pt")
    p.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="continue outputs/detector/<name>/weights/last.pt")
    a = p.parse_args()
    finetune_detector(a.data, a.base, a.epochs, a.imgsz, a.batch, a.name, a.device, a.workers, a.resume)


if __name__ == "__main__":
    main()
