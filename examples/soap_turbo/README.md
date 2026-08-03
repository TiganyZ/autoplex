# soap_turbo / turboGAP example

A complete, runnable example of fitting a GAP potential with the descriptor set
that [turboGAP](https://github.com/mcaroba/turbogap) evaluates natively —
[`soap_turbo`](https://github.com/libAtoms/soap_turbo) together with
`distance_2b` and `angle_3b` — and then checking that the fitted potential
survives conversion to turboGAP's own format.

The dataset is the 10-frame LiCl set that `autoplex` uses in its own tests. It is
far too small to give a useful potential; it is here so the whole pipeline runs
in well under a minute.

```
soap_turbo/
├── train/                        fit the potential
│   ├── gap_hyperparameters.yaml  which descriptors, and their hyperparameters
│   ├── training.yaml             dataset, species and run settings
│   ├── run_training.py
│   └── data/                     train.extxyz, test.extxyz
└── test/                         validate the potential
    ├── validation.yaml           tolerances and paths
    ├── run_validation.py
    └── data/                     structures to evaluate
```

## Running it

```bash
cd train && python run_training.py
cd ../test && python run_validation.py
```

Training writes `train/results/gap_file.xml` (plus its `.sparseX` files) and
reports the training and test errors. Validation reads that potential and does
two separate things:

1. **Scores it against the reference data** on the held-out test set. This says
   whether the fit is any good.
2. **Compares QUIP against turboGAP** on the same fitted potential. Both codes
   evaluate identical parameters, so any disagreement is a conversion problem
   rather than a fit-quality one.

Expected output, with the numbers from a run on this dataset:

```
Test set error : 2.196671e-05 eV/atom (tolerance 0.001)
Dropped 4 empty descriptor(s): angle_3b
Cl32Li32: dE = 2.954e-09 eV/atom, max|dF| = 4.982e-09 eV/Angstrom
Cl32Li32: dE = 4.442e-09 eV/atom, max|dF| = 4.986e-09 eV/Angstrom
QUIP and turboGAP agree on every validation structure.
```

Step 2 needs a turboGAP installation. Without one it is skipped with a warning
and the example still exits successfully, so step 1 remains useful on its own.
Point `autoplex` at turboGAP either through `validation.yaml`:

```yaml
turbogap_path: /path/to/turbogap
```

or through the environment:

```bash
export TURBOGAP_PATH=/path/to/turbogap
export PATH=$TURBOGAP_PATH/bin:$PATH
```

## Adapting it

Swap the contents of `train/data/` for your own `train.extxyz` and
`test.extxyz`, update `species_list` in both YAML files, and adjust the
descriptor hyperparameters in `gap_hyperparameters.yaml`. Every field not listed
in that file falls back to the defaults in `autoplex.settings.GAPSettings`.

The per-species `soap_turbo` entries (`alpha_max`, `atom_sigma_r`,
`atom_sigma_t`, the two scalings, `amplitude_scaling` and `central_weight`)
accept either one value, broadcast to every species, or a list with one value
per species in the order given by `species_list`.

## Constraints worth knowing

These are properties of turboGAP and QUIP, not of `autoplex`, and the defaults in
`gap_hyperparameters.yaml` already respect them. `autoplex` warns when they are
violated, but a fit can succeed and only disagree after conversion, so they are
easy to trip over.

| Setting | Constraint |
| --- | --- |
| `distance_2b` / `angle_3b` `cutoff_transition_width` | Must be `0.5`. turboGAP hardcodes this buffer and cannot read another value from the potential file. At any other value the potential fits and converts without complaint, then disagrees with QUIP. |
| `angle_3b` `sparse_method` | Cannot be `uniform`, which only supports one-dimensional descriptors. `angle_3b` is three-dimensional and the fit aborts. |
| `soap_turbo` `alpha_max` | Values above 7 are numerically unstable for the `poly3` basis; QUIP makes it an error above 10. Use `poly3gauss` if you need more radial basis functions. |
| `general` descriptor toggles | turboGAP cannot evaluate QUIP's `soap` or `distance_Nb`, so `two_body`, `three_body` and `soap` are off here. Leaving one on still gives a valid QUIP potential, but conversion is refused. |

Two further wrinkles are handled automatically during conversion, and explain
lines you will see in the output:

- **Empty descriptors are dropped.** `gap_fit` emits a descriptor for every
  species combination, including those with no instances in the training set. In
  this dataset 4 of the 6 `angle_3b` triplets are empty, because a rocksalt LiCl
  cell has no Li–Li–Li or Cl–Cl–Cl triplets inside the 3.25 Å cutoff. QUIP
  ignores them; turboGAP divides by the sparse-set size while reading the
  potential and dies with a floating-point exception.
- **Per-species `e0` values are carried across.** They live in the QUIP XML but
  not in the converted potential file. Without them every turboGAP energy is
  offset by the isolated-atom reference.
