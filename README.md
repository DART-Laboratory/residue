# _Residue_

## Installation

```bash
git clone https://github.com/DART-Laboratory/residue.git
cd residue
pip install -e .

# Needed to render provenance graphs
apt install graphviz
```

## Running

Put `residue run` in front of the entry point training script for your pipeline:

```bash
residue run <training-script>
```

writes `outputs/<timestamp>/residue.jsonl`, and the run's verification key to
`~/.residue-keys/<run>.txt`. To choose where the ledger lands, and to render a graph when the run
finishes:

```bash
residue run --out outputs/my-run --graph <training-script>
```

Then, once the run is done:

```bash
residue report outputs/my-run                            # whether the ledger has been tampered with, summary of recorded provenance 
residue report outputs/my-run --ref outputs/clean-run    # root-cause differential analysis results 
```

## Commands

| Command | What it does | Options |
|---|---|---|
| `residue run <script> [args…]` | Runs the script under capture. Args passed after the script name are untouched by Residue. | `-o/--out PATH` ledger file, or a directory to write `residue.jsonl` into (default `outputs/<stamp>/`) <br> `--keys-to PATH` where to write the run's key (default `~/.residue-keys/<run>.txt`) <br> `--graph` render a basic provenance graph when the run finishes |
| `residue report <target_run>` | The post-hoc report: ledger integrity, clean termination, root-cause diff, and a summary of every field captured. Accepts a run dir, its `residue/` dir, or the ledger itself. | `--ref RUN` clean reference run, enables the root-cause section <br> `--keys PATH` key file (found automatically by default) <br> `--literal` diff without run-id normalisation <br> `-o/--out PATH` report file (default `residue_report.txt` beside the ledger)<br> `--stdout` print recorded fields to console |
| `residue graph <ledger>` | Renders the ledger to a provenance graph via Graphviz. | `-o/--out PATH` output path (default `<ledger>.svg`) <br> `-f/--format FMT` dot output format: svg, png, or pdf (default svg) <br> `--full` draw every edge, including agent/intent fan-out, with no context collapse <br> `--fold` collapse the middle epochs of the training loop into one summary node <br> `--head N` epochs left expanded at the start under `--fold` (default 1) <br> `--tail N` epochs left expanded at the end under `--fold` (default 1) <br> `--from NODE_ID` render only one node's lineage, by id or unique prefix <br> `--direction` which way to slice from `--from`: `ancestors` what produced it (default), `descendants` what it reached, or `both` <br> `-v/--verbose` label every edge with its PROV relation, and add a few identifying fields to the labels of the nodes that have them <br> `--detail` node label contents: `curated` is type plus the identifying field (default), `all` dumps every field the node recorded. `all` replaces what `-v` adds to node labels, but not its edge labels, so `-v --detail all` shows the most <br> `--no-redundancy` drop entity-to-entity derivation edges implied by a path already drawn <br> `--rankdir` layout direction: `TB` upright portrait (default) or `LR` left-to-right <br> `--palette` what node colour encodes: `kind` is PROV kind (default), `bucket` is spine/trajectory/context, `rca` is causal-rank tiers in red on black-and-white, `bw` is print-safe greyscale <br> `--vs REFERENCE.jsonl` clean ledger to diff against; supplies the real buckets and rank tiers, without which `--palette bucket` falls back to node type and `rca` has nothing to paint <br> `--with-tail` outline trajectory divergences (weights, metrics) too, so the blast radius shows; fill stays reserved for ranked candidates <br> `--mono` draw all edges black; with `-v`, label each with its PROV relation <br> `--title TEXT` figure title (default `Run <run-dir name>`; pass `--title ""` to omit) <br> `--rca` print flag-path root-cause analysis, a backward trace over flagged nodes, to stdout |
| `residue verify <ledger>` | Checks the hash chain, and the MACs when the run's key is available. | `--keys PATH` key file (found automatically by default) |
| `residue observe <ledger>` | Projects the ledger to a flat `{field: value}` JSON summary of what was captured. | `-o/--out PATH` output file (default `observed.json` beside the ledger) |

Every command except `run` reads finished artifacts only, so they can be re-run over any ledger
that already exists, in any order, as many times as you like.

## What residue interposes on

Capture installs a fixed set of wrappers when `init()` runs. Each one replaces a library function
with a wrapper that calls the original and records the values crossing it — nothing is
reimplemented, and the observed pipeline is not modified. Where an interface has many concrete
implementations, residue patches the shared implementation once: a single wrapper on
`Metric.compute` sees every torchmetrics metric a pipeline computes, and one on
`binary_cross_entropy_with_logits` sees `BCEWithLogitsLoss` too.

Only PyTorch is required. Every other library below is patched if it is importable and skipped if
it is not.

| Boundary | Library | Interposed symbols | What the record establishes |
|---|---|---|---|
| Image read | Pillow | `Image.open` | which image files were read, and their contents |
| | scikit-image | `io.imread` | |
| | imageio | `imread`, `v2.imread` | |
| | OpenCV | `imread` | |
| | pydicom | `dcmread`, `filereader.dcmread` | |
| Image resize | Pillow · scikit-image · OpenCV | `Image.resize` · `transform.resize` · `resize` | the resize applied, and its parameters |
| Image write | Pillow · imageio · scikit-image · OpenCV | `Image.save` · `imwrite`, `v2.imwrite` · `io.imsave` · `imwrite` | the contents of the processed dataset left on disk |
| Table I/O | pandas | `read_csv`, `DataFrame.to_csv` | which label files were read and their contents; the name and contents of each split written to disk |
| Partitioning | scikit-learn · torch | `model_selection.train_test_split` · `utils.data.random_split` | split membership and the seed used |
| Sample read | *(the pipeline's own)* | `Dataset.__getitem__`, and the dataset's composed `transform` chain | per-sample contents as read, before augmentation; each transform in the chain, in order, with its construction parameters; normalization statistics |
| Batching | torch | `DataLoader.__iter__`, `BatchSampler.__iter__` | declared batch order — sampler class, shuffling, seed; realized batch order and sample composition, separating reordering from regrouping from substitution |
| Forward | torch | `register_module_forward_pre_hook`, `register_module_forward_hook` | forward passes per training step; the executed model structure — layers, buffer shapes, attached hooks; inputs the loader did not supply |
| Loss | torch | `nn.functional.{binary_cross_entropy_with_logits, binary_cross_entropy, cross_entropy, nll_loss, mse_loss, l1_loss, smooth_l1_loss, huber_loss, kl_div}` | loss per epoch; the computation actually optimized; labels the loader did not supply; per-sample losses taken outside a training step |
| Backward | torch | `Tensor.backward`, `autograd.grad` | backward passes per training step |
| Optimizer | torch | `optim.Optimizer.step` | applied hyperparameters — learning rate, weight decay, algorithm, batch size; training-step boundaries |
| Metrics | scikit-learn | `metrics.{f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score, log_loss}` | metric values, per split |
| | torchmetrics | `Metric.compute` | |
| Checkpoint | torch | `save` | contents of the weights written to disk |

