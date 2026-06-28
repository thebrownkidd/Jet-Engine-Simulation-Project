from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


@dataclass(frozen=True)
class FigureSet:
    name: str
    files_by_fd: Dict[str, Path]


def collect_shared_figure_sets(base_dir: Path, fd_names: List[str]) -> List[FigureSet]:
    by_fd: Dict[str, Dict[str, Path]] = {}
    for fd in fd_names:
        fd_dir = base_dir / fd
        if not fd_dir.exists():
            by_fd[fd] = {}
            continue
        by_fd[fd] = {p.name: p for p in sorted(fd_dir.glob("*.png"))}

    if not by_fd:
        return []

    # Shared names are those available in every FD directory.
    shared_names = set(by_fd[fd_names[0]].keys())
    for fd in fd_names[1:]:
        shared_names &= set(by_fd[fd].keys())

    figure_sets: List[FigureSet] = []
    for name in sorted(shared_names):
        files = {fd: by_fd[fd][name] for fd in fd_names}
        figure_sets.append(FigureSet(name=name, files_by_fd=files))

    return figure_sets


def make_combined_figure(figure_set: FigureSet, output_path: Path, dpi: int = 300) -> None:
    fd_names = sorted(figure_set.files_by_fd.keys())

    # 2x2 layout for FD001..FD004; each panel keeps the original image pixels.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_flat = axes.ravel()

    for i, fd in enumerate(fd_names):
        img = mpimg.imread(figure_set.files_by_fd[fd])
        ax = axes_flat[i]
        ax.imshow(img)
        ax.set_title(fd, fontsize=12)
        ax.axis("off")

    for j in range(len(fd_names), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(figure_set.name.replace(".png", ""), fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def combine_rollout_baseline_freerun(
    figures_root: Path, output_dir: Path, fd_names: List[str], dpi: int = 300
) -> None:
    baseline_dir = figures_root / "research" / "rollout_baselines"
    files_by_fd: Dict[str, Path] = {}

    for fd in fd_names:
        candidate = baseline_dir / f"freerun_{fd}.png"
        if candidate.exists():
            files_by_fd[fd] = candidate

    if len(files_by_fd) != len(fd_names):
        print("Skipped baseline freerun combine: missing one or more freerun_FD00*.png files.")
        return

    figure_set = FigureSet(name="freerun_rollout_baselines.png", files_by_fd=files_by_fd)
    out_path = output_dir / "freerun_rollout_baselines_FD001_FD004_combined.png"
    make_combined_figure(figure_set, out_path, dpi=dpi)
    print(f"Wrote: {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    figures_root = project_root / "results" / "figures"
    output_dir = figures_root / "combined_fd"
    fd_names = ["FD001", "FD002", "FD003", "FD004"]

    figure_sets = collect_shared_figure_sets(figures_root, fd_names)
    if not figure_sets:
        print("No shared figure names found across FD001-FD004.")
        return

    for fs in figure_sets:
        out_name = fs.name.replace(".png", "_FD001_FD004_combined.png")
        out_path = output_dir / out_name
        make_combined_figure(fs, out_path, dpi=300)
        print(f"Wrote: {out_path}")

    combine_rollout_baseline_freerun(figures_root, output_dir, fd_names, dpi=300)

    print(f"Done. Generated {len(figure_sets)} combined figures in {output_dir}")


if __name__ == "__main__":
    main()
