# PLassKMC

Structure-based Monte Carlo benchmarking of protein–ligand association using experimental holo complexes.

**Notebook:** [PLassKMC.ipynb](PLassKMC.ipynb)

PLassKMC simulates ligand approach to a fixed protein, records encounter, pocket-entry, and productive-binding events, and compares the resulting apparent rates with reference association rates derived from experimental dissociation rates and equilibrium dissociation constants. It saves completed complexes individually so that a benchmark can continue after a Colab runtime interruption.

The implementation uses Metropolis Monte Carlo moves with an approximate diffusion-based time mapping. Its apparent rates require calibration and convergence assessment before quantitative kinetic interpretation.

## Contents

- [Requirements](#requirements)
- [Inputs and benchmark systems](#inputs-and-benchmark-systems)
- [Quick start](#quick-start)
- [Notebook workflow](#notebook-workflow)
- [Model and event definitions](#model-and-event-definitions)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Resuming and rerunning](#resuming-and-rerunning)
- [Trajectory visualization](#trajectory-visualization)
- [Troubleshooting and limitations](#troubleshooting-and-limitations)

## Requirements

### Recommended environment

Use **Google Colab with a Python 3 runtime**, internet access, and a Google Drive account for persistent storage. The current numerical code uses CPU-based NumPy/SciPy operations; selecting a GPU runtime does not automatically accelerate it.

The first setup cell installs:

```python
%pip -q install rdkit biopython gemmi scipy scikit-learn pandas numpy matplotlib requests tqdm pyarrow
```

The import/setup cell mounts Google Drive and creates the project directories. Allow the Drive authorization request when prompted. Package versions are not pinned in this notebook; record the installed versions when reporting reproducible results.

### Local Jupyter use

The setup cell falls back to `/mnt/data/KMC_experimental_holo_benchmark` outside Colab. Use a writable directory appropriate to your machine if `/mnt/data` is unavailable. The optional visualization cell imports `google.colab.files` directly, so remove or replace that import and the final `files.download()` call for local use. An IPython/Jupyter environment is needed for the notebook's `%pip` commands and `display()` calls.

## Inputs and benchmark systems

### No manual upload is required

The notebook embeds its experimental table in `TEST_SET`. It automatically downloads experimental structure mmCIF files and Chemical Component Dictionary (CCD) files from RCSB PDB, then caches them for reuse:

- Structure coordinates: `https://files.rcsb.org/download/<PDB_ID>.cif`
- CCD chemistry: `https://files.rcsb.org/ligands/download/<COMPONENT>.cif`
- CCD ideal structures for identity matching: `https://files.rcsb.org/ligands/download/<COMPONENT>_ideal.sdf`

The source worksheet mentioned in a code comment is not needed to execute the notebook because the values are embedded.

### Included systems

| Complex ID | PDB ID | Target in the embedded table | Current treatment |
| --- | --- | --- | --- |
| `KMC11_01` | `1KV1` | p38 alpha | Included |
| `KMC11_02` | `3QRI` | ABL1 | Included; default visualization example |
| `KMC11_03` | `3QRJ` | ABL1 | Included |
| `KMC11_04` | `4LY1` | HDAC2 | Included |
| `KMC11_05` | `3F3V` | Src kinase | Included |
| `KMC11_06` | `4YX4` | Carbonic anhydrase II | Included |
| `KMC11_07` | `3PWH` | A2A adenosine receptor | Included |
| `KMC11_08` | `5FCK` | Complement factor D | Included; source table marks connectivity matching |
| `KMC11_09` | `5A3O` | LecB lectin | Explicitly excluded from the simulation set |
| `KMC11_10` | `3LDP` | Glucose-regulated protein | Included |
| `KMC11_11` | `5LWM` | JAK3 | Included; source table marks connectivity matching |

The original table contains **11 entries**, but the current code attempts simulations for **10**. `5A3O` is excluded because its ligand is represented by multiple PDB chemical components, which the single-component preparation workflow does not assemble. Identification or preparation failures can further reduce the number of successful systems.

Native-ligand identification first prefers an exact InChIKey match, then allows a connectivity-block match when an exact match is unavailable. Candidate ligand instances must have a nearest protein-heavy-atom distance of at most 4.5 Å. Inspect `identified_native_ligands.csv` for the actual match used; connectivity matching does not establish identical stereochemistry.

### Experimental reference rates

Reference association rates are calculated as:

```text
KD_M = KD_nM × 10^-9
kon_exp = koff_s_1 / KD_M
```

The resulting `kon_exp` values have units of M^-1 s^-1. These are rates derived from the supplied kinetic and affinity values, rather than a separate column of directly measured association rates. Interpretation assumes compatible experimental conditions and the applicability of the simple binding relation.

## Quick start

1. Open `PLassKMC.ipynb` in Google Colab.
2. Run the package-installation and import/setup cells. Mount Google Drive.
3. In **Section 1 — Configuration**, set `FAST_MODE = True` for an initial workflow check. The uploaded notebook defaults to `False`.
4. Run **Sections 2–6** in order to create the embedded test table, identify native ligands, validate system preparation, and define the simulation functions.
5. Optionally run the sample-trajectory export cell after Section 6. Leave `SELECTED_CONDITION_ID = "baseline"` for the self-contained example.
6. Run **Section 7** to execute the checkpointed benchmark.
7. Run **Sections 8–10** to calculate comparison metrics, inspect per-complex errors, and package the output tables and figures.

For a larger run, use `FAST_MODE = False` and a **new project directory** so that completed smoke-test checkpoints are not reused. The full settings are substantially more expensive; runtime depends on the complex, trajectory count, and step limit.

## Notebook workflow

| Section | Purpose | Main result |
| --- | --- | --- |
| Setup cells | Install packages, import modules, mount Drive, create folders | Persistent project directory |
| 1. Configuration | Set sampling, geometry, moves, and resume controls | Simulation settings |
| 2. Embedded experimental test set | Build the 11-entry reference table | `experimental_test_set_11.csv` |
| 3. Download and ligand identification | Fetch structures/CCD data, match ligands, exclude `5A3O` | Identification and failure tables; `run_set` |
| 4. CCD ligand construction | Define chemistry and coordinate-mapping helpers | RDKit ligand preparation functions |
| 5. System preparation | Build protein/pocket representations and validate each system | `system_preparation_report.csv` |
| 6. Energy, moves, and states | Define scoring, Monte Carlo moves, events, and rate conversion | Simulation functions and `DT_NS` |
| Optional cell after Section 6 | Run one trajectory with coordinate-frame export | Multi-model PDB, frame CSV, summary JSON |
| 7. Persistent benchmark | Run trajectories and save per-complex checkpoints | Rate estimates and trajectory summaries |
| 8. Rate comparison | Compare each endpoint with experimental reference rates | Metrics and comparison plots |
| 9. Error analysis | Calculate and rank per-complex log-scale errors | Comparison table and residual plot |
| 10. Output packaging | Write run metadata and archive `outputs/` | `RUN_SUMMARY.json` and ZIP archive |

## Model and event definitions

### Structural representation

The first coordinate model of each experimental holo structure is used. The protein is fixed. Standard amino-acid heavy atoms provide steric exclusion, while residues within 7 Å of the native ligand define a coarse-grained pocket. Pocket residues contribute Cα and, where available, side-chain-centroid beads with simplified chemical properties.

The ligand retains its heavy atoms and CCD-derived bonding. Experimental atom names map its coordinates into the native pose. RDKit supplies Gasteiger charges, chemical features, and candidate rotatable bonds.

### Energy and sampling

The score combines protein-wide steric repulsion, a soft Lennard–Jones-like interaction, screened electrostatics, hydrophobic attraction, and simplified donor–acceptor attraction. These are approximate scoring terms, not an explicit-solvent molecular-mechanics force field.

Each step proposes translation, rotation, or torsion and accepts or rejects the move using a Metropolis criterion. The ligand starts with a random orientation on a sphere 28 Å from the protein's heavy-atom centroid. Proposals beyond the 38 Å outer boundary are reflected.

### Kinetic endpoints

| Endpoint | Default criterion |
| --- | --- |
| Encounter | Minimum ligand-heavy-atom to pocket-bead distance ≤ 6 Å |
| Pocket entry | Ligand heavy-atom centroid lies within 7 Å of the native ligand centroid |
| Productive binding | Pocket entry, native-pose RMSD ≤ 3 Å, and native-contact recovery ≥ 0.40, maintained for 5 consecutive evaluated steps |

Native contacts are ligand-atom/pocket-bead pairs within 4.5 Å in the reference pose. RMSD uses corresponding ligand atoms in the fixed protein coordinate frame, without fitting or symmetry correction. The centroid is an unweighted coordinate average.

First-event steps are recorded for encounter and entry. For productive binding, `bound_step` records the start of the qualifying persistent sequence. The flags are separate geometric criteria; the code does not enforce an explicit sequence of transitions between three discrete states. Persistence counts evaluated Monte Carlo steps, including steps with rejected proposals.

### Approximate rate conversion

The implementation maps each attempted step to:

```text
DT_NS = TRANSLATION_SIGMA_A**2 / (2 × LIGAND_DIFFUSION_A2_PER_NS)
```

The defaults give `DT_NS = 0.0036 ns`. Event fractions are converted to apparent second-order rates using:

```text
P = event_count / trajectory_count
T = MAX_STEPS × DT_NS × 10^-9 seconds
V = (4π/3) × START_RADIUS_A**3 × 10^-27 liters
C = 1 / (Avogadro_constant × V)
k_app = -ln(1 - P) / (C × T)
```

An all-event fraction is capped at `1 - 0.5 / trajectory_count`; a zero-event count gives a raw rate of zero. The rate calculation uses the configured observation window and event fraction, rather than fitting the recorded first-event times. The concentration convention uses the starting-sphere volume, not the outer-boundary volume.

## Configuration

### Sampling modes

| Setting | `FAST_MODE = True` | `FAST_MODE = False` (default) |
| --- | --- | --- |
| Trajectories per complex | 8 | 1,000 |
| Maximum steps per trajectory | 1,500 | 10,000 |

Fast mode checks execution only. Full mode still requires replicate seeds, convergence testing, uncertainty estimates, and calibration for quantitative use.

### Main parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `SEED` | `20260719` | Base benchmark random seed |
| `POCKET_RADIUS_A` | `7.0` | Native-ligand neighborhood for pocket selection |
| `START_RADIUS_A` / `OUTER_RADIUS_A` | `28.0` / `38.0` | Initial and confining radii |
| `TRANSLATION_SIGMA_A` | `0.60` | Translation proposal standard deviation |
| `ROTATION_SIGMA_DEG` | `10.0` | Rotation proposal angle scale |
| `TORSION_SIGMA_DEG` | `22.0` | Torsion proposal angle scale |
| `MOVE_PROBS` | `0.60 / 0.27 / 0.13` | Translation / rotation / torsion probabilities |
| `TEMPERATURE_K` | `298.15` | Temperature in acceptance calculations |
| `IONIC_STRENGTH_M` | `0.15` | Screening parameter |
| `DIELECTRIC` | `78.5` | Electrostatic dielectric constant |
| `LIGAND_DIFFUSION_A2_PER_NS` | `50.0` | Approximate time-mapping parameter |

Changing configuration requires rerunning the dependent preparation/definition cells. Use a fresh project directory when changing model or sampling settings; checkpoints are keyed by complex ID, not a configuration hash.

## Outputs

### Storage locations

In Colab, the default project root is:

```text
/content/drive/MyDrive/KMC_experimental_holo_benchmark/
```

| Folder or file | Contents |
| --- | --- |
| `structures/` | Cached experimental structure mmCIF files |
| `ccd/` | Cached CCD CIF and ideal SDF files |
| `checkpoints/` | One JSON status/result record per complex |
| `trajectories/` | One `<complex_id>_trajectories.csv` per completed complex |
| `sample_visualization_trajectories/` | Optional coordinate trajectory and associated metadata |
| `outputs/` | Combined results, diagnostics, metrics, and figures |
| `kmc_experimental_holo_outputs.zip` | Archive of files directly inside `outputs/` |

### Main output files

| File in `outputs/` | Description |
| --- | --- |
| `experimental_test_set_11.csv` | Original embedded reference table, including excluded `5A3O` |
| `identified_native_ligands.csv` | Selected component, chain, residue, and match type |
| `native_ligand_identification_failures.csv` | Ligand identification failures |
| `system_preparation_report.csv` | Preparation status and system sizes |
| `kmc_complex_results.csv` | Successful checkpoint summaries and apparent rates |
| `kmc_complex_failures.csv` | Failed benchmark checkpoints and errors |
| `kmc_complex_running.csv` | Checkpoints still marked `running` |
| `trajectory_level_results.csv` | Combined trajectory event summaries, if available |
| `benchmark_metrics.csv` | Endpoint-wise log-scale comparison metrics |
| `kmc_11_complex_results_with_logs.csv` | Results with log-transformed rates; filename retains “11” despite the exclusion |
| `per_complex_comparison.csv` | Per-complex residuals and absolute errors |
| `encounter_rate_comparison.png` | Encounter versus experimental reference rates |
| `entry_rate_comparison.png` | Pocket-entry versus experimental reference rates |
| `bound_rate_comparison.png` | Productive-binding versus experimental reference rates |
| `per_complex_bound_residuals.png` | Productive-binding log residuals |
| `RUN_SUMMARY.json` | Counts, sampling settings, approximate time step, and run caveat |

The benchmark trajectory CSVs contain event steps and summary statistics, not time-resolved atomic coordinates. Use the optional visualization cell for a coordinate trajectory. The ZIP excludes checkpoint, cached structure, trajectory-summary, and visualization folders; retain the full project directory when those are needed.

### Comparison metrics

Metrics include Pearson correlation, Spearman correlation, MAE and RMSE in log10 units, fraction within tenfold, and zero-event fraction. Correlations are calculated only for at least three successful entries.

For log-scale plots and metrics, zero-event rates receive a half-event-derived floor. This is a numerical plotting/analysis convention, not a measured binding rate. A productive-binding residual of `+1` indicates a tenfold overestimate; `-1` indicates a tenfold underestimate. Inspect event counts alongside correlations, especially for short runs.

## Resuming and rerunning

After a disconnection:

1. Reopen the notebook and rerun installation/import cells.
2. Mount the same Drive and retain the same project path and configuration.
3. Rerun Sections 1–6 to recreate the in-memory tables and functions.
4. Rerun Section 7, then the comparison and packaging sections.

| Control | Default | Behavior |
| --- | --- | --- |
| `RESUME_COMPLETED` | `True` | Skip complexes with status `ok` |
| `RETRY_FAILED` | `True` | Retry complexes with status `failed` |
| `SAVE_COMBINED_TABLES_AFTER_EACH_COMPLEX` | `True` | Rebuild combined tables after each attempted complex |
| `RESUME_DOWNLOADS` | `True` | Reuse nonempty cached downloads |

A checkpoint marked `running` is rerun from the beginning of that complex. **Recovery is per complex, not per trajectory or Monte Carlo step.** Partial in-memory trajectories from an interrupted complex are not resumed.

For clean comparisons between settings or seeds, change `DRIVE_ROOT` to a new directory in the setup cell. Disabling resume alone does not remove old records: combined tables are rebuilt from all checkpoints and trajectory CSVs in the selected directory, including files outside a newly restricted run set. Analysis cells also use the current configuration for rate floors, so do not mix incompatible runs.

## Trajectory visualization

The optional cell immediately after Section 6 runs one sample and writes:

- A multi-model `.pdb` file with a static protein and moving ligand.
- A `.frames.csv` file containing frame numbers, steps, energy, geometric state flags, RMSD, and native-contact recovery.
- A `.summary.json` file containing event steps and acceptance statistics.

Default settings are:

```python
SELECTED_COMPLEX_ID = "KMC11_02"  # 3QRI, ABL1
SELECTED_CONDITION_ID = "baseline"
SAMPLE_SEED = 20260728
SAMPLE_MAX_STEPS = 10000
SAVE_FRAME_EVERY = 10
SAVE_EVENT_FRAMES = True
```

The PDB is downloaded automatically in Colab and can be opened as a multi-model structure in a molecular viewer. Frame intervals are not necessarily uniform because event and final frames can be added between regularly sampled frames. Use the frame CSV to interpret step spacing.

The cell lists non-baseline condition names, but this notebook does **not** define `perturbation_panel` or run a full perturbation benchmark. Selecting a non-baseline condition requires a separately defined panel; the message referring to “Section 10” is not applicable because Section 10 only packages outputs. In addition, `flexibility_scale` is resolved into a local torsion scale but is not passed to the existing `torsion_move()` function. Do not interpret changing that setting as a verified flexibility perturbation in this version. Baseline visualization does not require the missing panel.

## Troubleshooting and limitations

### Downloads or identity matching fail

Check internet access and inspect the identification failure table. Confirm the requested PDB ID, CCD component identity, and ligand atom names. To refresh a suspected cached download, set `RESUME_DOWNLOADS = False` and rerun the relevant cells. Returning it to `True` restores cache reuse.

### Ligand preparation fails

Inspect `system_preparation_report.csv` for missing atom-name mappings, invalid chemistry, or an empty pocket. The preparation report does not itself remove failed systems from `run_set`; the benchmark retries preparation and records a failure if it still fails. Multi-component ligands are not assembled automatically.

### No productive binding is observed

Zero events can reflect a short observation window, limited sampling, model choices, or difficult geometry. Inspect encounter and entry counts, RMSD, and contact recovery before drawing conclusions. More sampling or altered thresholds change the calculation and should be documented in a separate run directory.

### Scope of the model

- The native holo pose defines the pocket, contacts, and RMSD endpoint; this is a native-pose-informed benchmark, not blind docking.
- Protein conformational change, explicit solvent, directional hydrogen bonding, metal coordination, and a membrane environment are not explicitly simulated. Protein sterics and simplified pocket interactions do not substitute for these effects.
- Torsion moves do not include a full intraligand energy model; the score primarily describes protein–ligand interactions.
- The shared approximate time step is assigned to translation, rotation, torsion, and rejected proposals. It is not a calibrated event-driven residence-time algorithm.
- `3QRI` and `3QRJ` use the same ligand identity and target, so the entries should not be treated as fully independent ligand–target pairs.
- Experimental sources, assay conditions, and measurement uncertainties should be documented separately before formal benchmark reporting; the embedded table contains values and identities but no per-row bibliographic provenance.

This README documents the supplied notebook's code and defaults. It does not report a new benchmark run or independently validate the predicted rates.
