# LOKOGAD — Low-Cost Rehabilitation Exoskeleton via Graph-Assisted Design

Code, results and paper source for

> **LOKOGAD: Design of a Low-Cost Rehabilitation Exoskeleton Based on a Modified Theo Jansen Mechanism
> for Human Bipedalism via the Graph-Assisted Design Framework**
> Gabriel A. Laurente Caldas, Axel T. Mora Zeña, Diego F. Pantoja Iturrizaga, Jimmy F. León Vásquez —
> Faculty of Biomedical Engineering, Universidad Nacional Mayor de San Marcos (UNMSM)
> PDF in [`paper/`](paper/)

The classic Theo Jansen linkage (built for quadrupedal walking) is reconfigured for human gait. The
mechanism is represented as a graph, its kinematics are solved exactly (circle–circle intersection), and its
bar lengths are optimized with NSGA-II against reference trajectories from the OpenSim Gait2392 model,
followed by a topological refinement stage. The final 12-bar, single-degree-of-freedom mechanism was
3D-printed (PLA/PETG) and driven by a single DC motor.


**Scope:** kinematics only. No force, torque or load criteria were optimized, the prototype was not tested
with patients or under load, and its trajectory was not measured.

## Results

| Metric | Value |
|---|---|
| Foot Procrustes error | 4.97 % |
| Knee Procrustes error | 10.79 % |
| Combined fitness | 0.11177 |
| Horizontal foot excursion (human scale) | 966.35 mm |
| Final mechanism | 8 pivots, 12 bars, 6 Z-layers |

## Repository layout

```
core/            shared modules: configuration and paths, linkage genome (graph), exact kinematic solver
pipeline/        fase1 dataset → fase2 GAT surrogate → fase3 NSGA-II → fase3 topology evolution → fase4 export
fabrication/     collision-free Z layers, print-ready STL export, motor-coupling bar, assembly diagram
visualization/   mechanism animations, comparison and gait-snapshot figures
preprocessing/   OpenSim trajectories → Fourier smoothing → reference tensors
opensim_extraction/  gait extraction from OpenSim Gait2392 and the reference tensors used by the optimizer
mechanism/       original Theo Jansen mechanism and URDF generator
snap_joint/      snap-fit joint design (STEP/STL)
video/           four-act presentation video (scripts and presentation_full.mp4)
data/, models/, results/   dataset, trained surrogate, optimization results and print-ready STL
paper/           paper (PDF)
```

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1. Reference gait | `opensim_extraction/extract_gait.py`, `preprocessing/01–03_*.py` | Joint trajectories from OpenSim Gait2392, Fourier smoothing, reference tensors |
| 2. Representation & sampling | `core/linkage_genome.py`, `pipeline/fase1_jansen_dataset.py` | Jansen mechanism as a graph (13 "Magic Numbers"); 10 000-sample tri-modal dataset |
| 2b. Surrogate (optional) | `pipeline/fase2_jansen_gat.py` | GAT surrogate trained on the dataset |
| 3. Optimization | `core/general_solver.py`, `pipeline/fase3_jansen_nsga.py` | Exact kinematic solver + NSGA-II (population 10 000, 500 generations) |
| 3b. Topological refinement | `pipeline/fase3_topology_evo.py` | Graph-level evolution (pivots/bars), yields the final 12-bar mechanism |
| 4. Fabrication | `fabrication/check_and_fix_collisions.py`, `fabrication/export_stl_print_ready*.py` | Swept collision test over 360 crank angles + greedy coloring of the conflict graph (→ `results/z_layers_fixed.json`); Shapely/trimesh STL export with 0.4 mm print tolerance |
| 5. Digital twin | `mechanism/generate_jansen_urdf.py` | URDF model with computed inertial properties |

Configuration (paths, constants) lives in `core/config_jansen.py`; all paths are relative to the repository.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric for your hardware (CUDA, ROCm or CPU) following their official
instructions. OpenSim is only needed to re-extract the reference gait (phase 1).

## Running

Run every command from the repository root.

```bash
bash run_jansen_pipeline.sh                       # phases 2 → 4 with the paper settings

# or step by step
python pipeline/fase1_jansen_dataset.py --n-samples 10000
python pipeline/fase3_jansen_nsga.py --pop-size 10000 --n-gen 500
python pipeline/fase3_topology_evo.py --n-gen 500
python pipeline/fase4_jansen_export.py --top-n 5
python fabrication/check_and_fix_collisions.py    # collision-free Z layers
python fabrication/export_stl_print_ready.py      # print-ready STL of the final mechanism
```

The reference gait tensors (`opensim_extraction/*_ref_tensor.pt`) are included, so phase 1 is not needed to
run the optimization. To re-extract the gait, clone
[opensim-models](https://github.com/opensim-org/opensim-models) into `opensim_extraction/` (see
`opensim_extraction/setup_opensim.sh`) and run `extract_gait.py`. The scripts in `preprocessing/` expect
`joint_trajectories_opensim.mat`, which is not included in this repository.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code released under the [MIT License](LICENSE). The paper in `paper/` is © its authors.
