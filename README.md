# FoldQC

FoldQC is a PyMOL plugin for visualizing confidence metrics from predicted
protein structures. It supports Boltz (local) prediction folders, Boltz Lab and Boltz
API outputs, AlphaFold 3 local and server outputs, localfold exports, OpenFold3,
Chai-1 Discovery and Protenix prediction folders, AlphaFold Database (EMBL-EBI) monomer and
complex predictions, zip/tar archives containing supported prediction folders,
and single CIF/PDB files with pLDDT values stored in B-factors.

FoldQC can color structures by pLDDT, PAE, PDE, contact probability, chain
ipTM, and ensemble metrics. It also provides line plots, matrix plots, PAE/PDE
summary plots, binding site fingerprints, and ensemble summaries.

Development of FoldQC has included coding assistance from OpenAI's Codex.

## Installation

FoldQC requires Python 3.10 or newer in the PyMOL installation. Optional
packages are installed into that interpreter, using releases compatible with
its Python version.

1. Copy or clone this repository into a PyMOL plugin directory. Common user
   plugin locations include:

   ```text
   Windows:
   %APPDATA%\pymol\startup\FoldQC

   Linux/macOS:
   ~/.pymol/startup/FoldQC
   ```

   On Windows, `%APPDATA%` is usually `C:\Users\<username>\AppData\Roaming`.

   In addition, you can also install it through PyMOL's plugin manager, where you can also define custom plugin directories. The plugin manager is accessible from the menu: `Plugin -> Plugin Manager...`, then choose `Install New Plugin` or
   `Settings`.

2. Start PyMOL and open the plugin from:

   ```text
   Plugins -> FoldQC...
   ```

3. FoldQC checks optional dependencies only when a feature needs them. If
   Matplotlib, SciPy, or scikit-learn is missing, FoldQC offers to install the
   required package into PyMOL's Python environment and shows installation
   progress. The feature can usually be retried immediately after a successful
   installation; a PyMOL restart may be necessary if it is not yet available.

   FoldQC does not automatically use `pip --user`. If PyMOL's environment is
   not writable or automatic installation fails, the dialog provides copyable
   conda and pip user-installation commands. User-site packages can mix across
   conda environments, so prefer conda when it is available.

   As a manual alternative, run the standalone installer from the PyMOL command
   line:

   ```text
   run /path/to/FoldQC/tools/install_deps.py
   ```

   You can also select `install_deps.py` from the plugin's `tools` folder using
   `File -> Run Script...`.

## Usage

1. Open FoldQC from the PyMOL plugin menu.
2. Choose a prediction folder, archive, or single `.cif`/`.pdb` structure file,
   or select one of the last 10 successfully loaded predictions from the editable
   path history. FoldQC restores this history across sessions but starts with an
   empty path and never loads a saved prediction automatically. Alternatively,
   enter a UniProt accession or isoform ID in the second source row and click
   `AFDB-EBI` (or press Return). FoldQC queries monomer and complex entries,
   prompts when more than one prediction is available, and keeps a separate
   history of the last 10 successfully loaded accessions.
3. Select the model, target object, metric, and color palette. The palette,
   Reverse, Min, and Max controls affect PyMOL coloring only.
   When FoldQC must create a model object in PyMOL, it initially applies the
   familiar pLDDT quality-class coloring. If the named object already exists,
   FoldQC reuses it without overwriting its current colors.
   For predictions with multiple ranks, `Compare` opens a read-only
   table of provider summary scores for every rank. Selecting a row switches to
   that model. The table reads compact scalar summaries such as ranking score,
   pTM, ipTM, and complex pLDDT without loading every structure or PAE/PDE
   matrix; columns remain provider- and file-dependent. Prediction-level Boltz
   affinity values are intentionally excluded from this per-model table.
   Click a column heading to sort by that column, and click it again to reverse;
   an arrow indicates the current sort. Scores sort numerically, with missing
   values always last. The `Model` column sorts by original rank. Sorting keeps
   the selected model selected, and reopening the table restores original rank
   order.
4. Click the paint/color action to write the metric into B-factors and color the
   structure in PyMOL.
5. Use the statistics panel's threshold control and `Select ≥` / `Select ≤`
   buttons to create a persistent named PyMOL selection from the metric values
   that were just colored, independently of subsequent B-factor changes.
   For ensemble targets, each member is thresholded against its own values and
   combined into one selection. Reusing a metric/direction pair replaces its
   `foldqc_<metric>_ge` or
   `foldqc_<metric>_le` selection.
6. For selection-based metrics or site-focused plots, enter a PyMOL selection
   in the contextual Reference / Ligand-site field. The cutoff field is enabled
   for metrics that use it, such as contact-filtered PAE/PDE and PAE domain labels.
7. Use the `Plot` menu and ensemble actions for heatmaps, line plots,
   PAE/PDE summary plots, binding-site fingerprints, and multi-model summaries.

Plots use their own appearance defaults, independently of the PyMOL coloring
controls. PAE, PDE, and interaction probability matrices use `viridis`; the
chain pTM/ipTM matrix uses `viridis_r`. Matrix color scales start at zero and
use a data-derived upper bound, including for reference-restricted and ensemble
matrices. Line and PAE/PDE summary plots autoscale their y-axes. Adjust plot
palettes and scale limits using the embedded Matplotlib figure controls.

For an AlphaFold Database selection, FoldQC downloads the selected structure
and loads its optional PAE data when needed. pLDDT is read from the CIF B-factors.
Both monomers and complexes are treated as one-model predictions;
complexes expose pLDDT and PAE when present and show the EBI interface scores
ipTM, ipSAE, pDockQ2, and LIS in the confidence summary. These interface scores
range from 0 to 1, with higher values indicating greater confidence. ipTM
assesses relative chain placement; ipSAE focuses on reliably positioned
inter-chain residue pairs; pDockQ2 estimates interface quality from interface
pLDDT and PAE; and LIS summarizes confidence in local low-PAE interaction
regions. They are summary values rather than colorable token metrics.

`Load Ensemble` is enabled only for predictions containing at least two
models and only until that prediction's ensemble has been activated. Its
tooltip explains when a single-model prediction or an already-loaded ensemble
makes the action unavailable.
Choose the ensemble group in the PyMOL target selector to analyze all loaded
members, or choose an individual member object to analyze that model.
Automatic ensemble alignment also creates the named PyMOL selection
`foldqc_alignment_core`. It contains the rank-0 reference-object residues used
for the fit; choosing current coordinates leaves this selection empty.

`PAE summary` and `PDE summary` are (experimental) speciality line plots for multi-chain
targets. They plot the gap between each token's mean error to other chains and
its mean error within its own chain (`other - within`); PAE summary shows
separate row and column gap lines. If a Reference selection is entered, only
the displayed x-range is restricted to those tokens; the summary values are
still computed against the full complex. Ensemble targets show member means
with shaded standard deviations.

## Data availability

Available metrics depend on the selected model and target. FoldQC loads PAE,
PDE, and other optional data only when an analysis needs them. Missing values
remain unavailable; malformed data produce an error identifying the provider,
model, field, and source file.

Actions use the settings selected when they start. If the prediction, model,
ensemble, or relevant controls change while work is running, outdated results
are discarded. Cancelled actions do not apply their results. If coloring fails,
FoldQC restores the previous atom values, colors, and colorbar.

## CSV Export Schema

Use `Export CSV` to write token-level scalar values for the currently
selected metric, target, reference selection, and cutoff/threshold. The export
does not require the metric to be colored first. It is intended for downstream
analysis, so it includes biological token metadata and provenance, but omits
visualization-only state such as palettes, color scale limits, PyMOL object
names, and PyMOL selection strings.

Each exported row represents one FoldQC token: one polymer residue or one heavy
atom for HETATM ligands. For an ensemble group target, ensemble-level metrics
write one aggregate row per token, while single-model metrics write one row per
token per ensemble member. Full PAE, PDE, and interaction-probability matrices
are not exported by this workflow.

Base columns:

| Column | Meaning |
| --- | --- |
| `export_schema_version` | CSV schema version. Currently `2`. |
| `provider` | Prediction provider, such as `boltz`, `boltz_lab`, `boltz_api`, `alphafold3`, `af3_server`, `alphafold_db`, `openfold3`, `chai1`, `protenix`, or `structure_only`. |
| `prediction_name` | Prediction name. |
| `input_path` | Original selected folder, zip, CIF, or PDB path. |
| `structure_path` | Original structure file used for analysis. |
| `model_rank` | Ranked model number used for the row's values. |
| `model_label` | Model label shown in FoldQC. |
| `metric_key` | Stable metric identifier, such as `pde_contact` or `ensemble_plddt_mean`. |
| `metric_label` | User-facing label from the `Color by` control. |
| `value` | Token-level scalar metric value. Undefined values are written as `nan`. |
| `value_units` | Machine-readable unit/category: `plddt`, `angstrom`, `probability`, `iptm`, or `label`. |
| `value_semantics` | Interpretation hint: `higher_is_better`, `lower_is_better`, or `categorical_label`. |
| `reference_selection` | Reference text used by selection-based metrics. Blank when unused. |
| `cutoff_angstrom` | Cutoff/threshold used by contact-filtered PAE/PDE or PAE domain labels. Blank when unused. |
| `token_index` | Zero-based token index in original prediction token order. |
| `token_type` | `polymer_residue` or `ligand_atom`. |
| `chain_id` | Chain ID from the original structure file. |
| `res_num` | Numeric residue number from the original structure file. |
| `residue_id` | Complete residue label, including an insertion code when present (for example `42A`). |
| `insertion_code` | PDB/mmCIF insertion code; blank for ordinary residue identifiers. |
| `res_name` | Residue or ligand name. |
| `atom_name` | Atom name for ligand-atom tokens; blank for polymer residues. |
| `is_hetatm` | `true` for ligand/HETATM tokens, otherwise `false`. |
| `is_reference_token` | `true` when the token is part of the resolved reference selection. |
| `is_contact_token` | `true` for tokens in the contact shell for contact-filtered exports. |

Ensemble columns are included only for ensemble exports:

| Column | Meaning |
| --- | --- |
| `ensemble_group` | Active FoldQC ensemble group name. |
| `ensemble_member_rank` | Member rank for per-member rows; blank for aggregate ensemble rows. |
| `ensemble_member_label` | User-facing member label for per-member rows. |
| `ensemble_aligned` | `true` when the active ensemble was automatically aligned, `false` when current coordinates were used. |
| `aggregate_kind` | `single_model`, `ensemble_member`, `ensemble_mean`, `ensemble_std`, or `ensemble_rmsd`. |

Metric units and semantics:

| Metric keys | `value_units` | `value_semantics` |
| --- | --- | --- |
| `plddt_class`, `plddt`, `ensemble_plddt_mean` | `plddt` | `higher_is_better` |
| `ensemble_plddt_std` | `plddt` | `lower_is_better` |
| `pae_row_mean`, `pae_col_mean`, `pae_to_sel`, `pae_col_to_sel`, `pae_sym_sel`, `pae_sym_within_sel`, `pae_contact` | `angstrom` | `lower_is_better` |
| `pae_domain_complete`, `pae_domain_spectral` | `label` | `categorical_label` |
| `pde_mean`, `pde_chain_mean`, `pde_to_sel`, `pde_within_sel`, `pde_contact`, `ensemble_rmsd` | `angstrom` | `lower_is_better` |
| `contact_prob_mean`, `contact_prob_to_sel` | `probability` | `higher_is_better` |
| `chain_iptm` | `iptm` | `higher_is_better` |
| `chain_ptm` | `ptm` | `higher_is_better` |

## Color by Metric Reference

FoldQC colors token-level values. A token is one polymer residue or one heavy
atom for HETATM ligands. Continuous metrics are written into PyMOL B-factors and
colored with the selected palette; undefined values are colored grey. Selection
metrics use the PyMOL expression in the Reference field, mapped back to FoldQC
tokens.

Residues are matched by chain, residue number, insertion code, and residue name
across selections, plots, ensembles, and CSV export. Ligand hydrogen atoms
(including deuterium and tritium) are excluded from analysis.

PAE and PDE are error metrics, so lower values usually mean higher confidence.
pLDDT, interaction probability, and ipTM are confidence metrics, so higher
values usually mean higher confidence.

The `Color by` control is grouped by metric family: pLDDT, PAE, PDE,
Interaction probability, Chain/interface, and Ensemble. `[Advanced]` marks
specialized options that need more care to interpret, and `[Experimental]`
marks heuristic workflows that may depend on optional scientific packages.
These markers do not change the calculations.

The compact preview below the metric controls summarizes what the current
selection will color and calls out missing requirements such as a reference
selection or unloaded ensemble.

FoldQC uses the same pLDDT values for coloring, plots, exports, alignment, and
ensemble consensus. Boltz uses its token NPZ when present. All atom-level
sources, including provider confidence arrays and structure B-factors, average
finite polymer atom values by residue in prediction token order; ligand heavy
atoms remain individual tokens. OpenFold3 uses its full-confidence per-atom
pLDDT when available.
Standalone structures, Chai, Boltz Lab, and Boltz API use structure B-factors.
Missing provider arrays fall back to structure B-factors; malformed explicit
arrays are reported as errors. This follows AlphaFold 3's documented
[`atom_plddts` per-atom semantics](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md#metrics-in-confidences-json).

PAE, PDE, and interaction-probability availability can differ between models.
Ensemble coloring and ordinary metric plots require every targeted member to
provide the selected family. Binding-site fingerprints and ensemble-site
summaries instead aggregate the members that
provide each family and leave unavailable member values missing.

| Color by option | How it is calculated | How to interpret it |
| --- | --- | --- |
| pLDDT - classes | Applies the AlphaFold four-class coloring to pLDDT: very high >=90, high 70-90, low 50-70, very low <50. | Quick local-confidence overview. Blue/light blue regions are more reliable; yellow/orange regions should be treated cautiously. |
| pLDDT - continuous | Colors pLDDT as a continuous value. | Higher values indicate higher local model confidence. |
| PAE - row mean | For token `i`, computes `mean(PAE[i, :])` over all other tokens. | Average uncertainty of the rest of the model when aligned on token `i`. Lower values indicate a better anchored token; higher values often mark flexible or poorly positioned regions. |
| PAE - column mean | For token `j`, computes `mean(PAE[:, j])` over all alignment frames. | Average uncertainty in token `j`'s position from the perspective of all other tokens. Lower values indicate globally consistent placement. |
| PAE - row mean to selection | For each token `i`, computes `mean(PAE[i, reference_tokens])`. | Directional confidence of each token relative to the reference selection, such as a ligand or partner chain. Lower values indicate more confident placement relative to the reference. |
| PAE - column mean to selection | For each token `j`, computes `mean(PAE[reference_tokens, j])`. Matrix plots for this metric show reference tokens on rows and all tokens on columns. | Opposite PAE direction from row mean to selection. Useful when a ligand or other reference has asymmetric PAE relative to the surrounding structure. Lower values indicate more confident placement from the reference frame to the token. |
| PAE - symmetric mean to selection | Averages both PAE directions between each token and the reference: token-to-reference and reference-to-token. Reference tokens are scored against non-reference tokens. | Less sensitive to PAE directionality than "mean to selection". Lower values indicate more mutually confident relative placement. |
| PAE - symmetric mean within selection | For selected tokens only, computes mean symmetric PAE against the same selected token set; all non-selected tokens are undefined. | Internal relative-placement confidence within a chosen ligand, residue set, or chain. Lower values indicate the selected region is predicted as a more coherent arrangement. |
| PAE - contact-filtered to selection | Finds non-reference polymer residues with any atom within the contact cutoff of the reference selection, then computes symmetric mean PAE to the reference only for those contact tokens. | Binding-site-focused relative-placement confidence for AF3-style outputs that lack PDE. Lower values indicate more reliable predicted contacts; grey tokens were outside the contact shell or undefined. |
| PAE - domain labels (complete linkage) | Builds a hierarchy from symmetric PAE distances, then cuts it at the cutoff/threshold field. Requires SciPy. | Tokens with the same label are grouped only when all pairwise symmetric PAE distances within the cluster fit under the selected threshold. Labels are categorical and use discrete colors; neighboring label numbers do not imply ordered confidence. Ensemble members are clustered independently. |
| PAE - domain labels (spectral clustering) | Converts symmetric PAE to a continuous affinity matrix, estimates the cluster count from the normalized-Laplacian eigengap capped at 12, then applies spectral clustering. Requires SciPy and scikit-learn. | Heuristic soft-domain grouping for models where rigid bodies are not cleanly separated by strict complete-linkage clustering. Labels are categorical and use discrete colors. Ensemble members are clustered independently. |
| PDE - mean | For token `i`, computes `mean(PDE[i, :])` over all tokens. | Average predicted distance error for token `i` relative to the whole model. Lower values indicate more reliable pairwise distances. |
| PDE - mean to selection | For each token `i`, computes `mean(PDE[i, reference_tokens])`. | Distance-error confidence between each token and the reference selection. Lower values indicate more reliable distances to the selected ligand, residue set, or chain. |
| PDE - within-chain mean | For each token, computes the mean PDE only against tokens in the same chain. | Intra-chain distance confidence. Lower values indicate a more internally well-defined chain region. |
| PDE - within-selection mean | For selected tokens only, computes mean PDE against the same selected token set; all non-selected tokens are undefined. | Internal distance confidence within a chosen region. Lower values indicate the selected region is predicted as a more coherent local arrangement. |
| PDE - contact-filtered to selection | Finds non-reference polymer residues with any atom within the contact cutoff of the reference selection, then computes mean PDE to the reference only for those contact tokens. | Binding-site-focused distance confidence. Lower values indicate more reliable predicted contacts; grey tokens were outside the contact shell or undefined. |
| Interaction probability - mean | For each token, computes the mean predicted interaction/contact probability against all other tokens. | Higher values indicate a token is predicted to participate broadly in contacts or interactions. |
| Interaction probability - mean to selection | For each non-reference token, computes mean predicted interaction/contact probability against the reference tokens; reference tokens are set to undefined/grey. | Higher values indicate stronger predicted interaction with the selected ligand, chain, or residue set, without letting trivial within-reference contacts dominate the color scale. |
| Ensemble RMSD | After ensemble setup, computes per-token coordinate RMSD across ensemble members. By default, models are aligned on a high-confidence polymer core; expert mode can instead use current PyMOL coordinates. | Lower values indicate agreement across models; higher values indicate conformational variability or uncertain placement. |
| Ensemble pLDDT mean | After ensemble setup, computes the per-token mean pLDDT across loaded models. | Higher values indicate consistently high local confidence across the ensemble. |
| Ensemble pLDDT std | After ensemble setup, computes the per-token standard deviation of pLDDT across loaded models. | Lower values indicate stable confidence estimates across models; higher values mark positions where model confidence varies between predictions. |
| Chain ipTM | Assigns each token its chain's ipTM, falling back to per-chain pTM when ipTM is unavailable. Matrix plots show pairwise chain ipTM. | Higher values indicate higher confidence in chain-level placement or interactions. Off-diagonal cells describe confidence for a chain pair; diagonal cells are chain-restricted pTM/self scores, not interfaces. |
| Chain pTM | Assigns each token its chain's main-diagonal score from the normalized pairwise chain ipTM matrix. If no matrix is available, uses the provider's per-chain pTM vector. | Higher values indicate higher confidence in the chain's internal fold, subject to provider-specific length scaling, especially for [small ligands and short chains](#small-ligands-and-short-chains). Missing scores remain undefined (grey); interface scores and global pTM are not used as substitutes. |

[localfold](https://github.com/sokrypton/localfold) exports use the AlphaFold
Server layout with `*_model_<rank>.pdb` structures and matching
`*_summary_confidences_<rank>.json` / `*_full_data_<rank>.json` files. FoldQC
reads these through its AlphaFold 3 Server provider (`af3_server` in CSV
exports), independently of the weights used, including AF3 and Boltz-2.
Atom pLDDTs are averaged per residue; PAE, interaction probabilities,
and chain confidence use the same loading rules as Server outputs. Native
Boltz-only data such as PDE or affinity are not implied by Boltz-2 weights.
Select the extracted folder or its ZIP archive. If both CIF and PDB exist for
the same model basename, FoldQC uses CIF.

OpenFold3 output roots contain one directory per query and one `seed_<n>`
directory per model seed. When a selected folder or archive contains multiple
queries, FoldQC lists each query as a separate prediction candidate, including
queries below an additional wrapper/output directory. Within a query, all seed
and diffusion-sample structures are ranked by `sample_ranking_score`. Full
confidence JSON files provide per-atom pLDDT, PAE, and PDE lazily; when a full
file is absent, pLDDT falls back to the structure B-factors and PAE/PDE remain
unavailable. Aggregated confidence supplies average pLDDT, pTM, ipTM, gPDE,
disorder, clashes, chain pTM, pair-chain ipTM, and bespoke ipTM. Bespoke ipTM is
shown for finite chain pairs in the confidence summary and is not currently a
color or plot metric.

Chai-1 Discovery folders provide structure B-factor pLDDT, PAE matrices, and
score JSON/NPZ files with global pTM/ipTM, per-chain pTM, pairwise chain ipTM,
and clash flags. PDE metrics are available for Chai only when matching
`pde*.npy` files are present. Interaction probability metrics are not exposed
for Chai-1 Discovery folders.

Protenix folders provide summary confidence JSONs, global pTM/ipTM/gPDE,
per-chain pTM/ipTM, pairwise chain ipTM, and clash flags. When a matching
`*_full_data_sample_*.json` contains atom pLDDT, those values are used instead of
structure B-factors. PAE, PDE,
and interaction probability metrics are available only when their arrays are
present in the full-data file.

### Interpreting Pairwise Chain ipTM

Chain pTM and Chain ipTM both support Line and Matrix plots, but not Distribution
plots. Reference selections restrict only line-plot ranges. Both metrics show the
same full pairwise chain ipTM matrix, including off-diagonal interface scores;
Chain pTM coloring uses its diagonal self-scores. Normalization fills missing or
zero diagonal cells from provider per-chain pTM when available. Ensemble line
plots show member means and standard deviations; matrix plots show member means
with standard-deviation annotations. Per-chain pTM without a matrix supports
coloring and line plots, but Matrix reports unavailable pairwise data.

The pairwise chain ipTM matrix lists chains in their order of first appearance
in the original structure. Across all providers, FoldQC uses the same plot axes:

- rows are alignment/reference chains;
- columns are scored chains whose placement is evaluated relative to the row;
- `matrix[i, j]` therefore means alignment chain `i` → scored chain `j`.

AlphaFold 3 `chain_pair_iptm` is documented as the ipTM restricted to the two
selected chains and is symmetric. OpenFold3's pairwise ipTM and bespoke ipTM
are also symmetric. For these matrices, swapping row and column does not change
the displayed off-diagonal value. The bespoke matrix has no diagonal self-score and is
presented separately from the colorable chain-ipTM matrix.

When a normalized matrix is asymmetric, report both directions when diagnosing
an interface. For a single conservative interface summary, use
`min(matrix[i, j], matrix[j, i])`; for a permissive summary, use the mean or
maximum, but state which aggregation was used. Do not infer direction from a
provider's raw field name alone; use the normalized FoldQC plot axes described
above.

#### Small ligands and short chains

**A very low Chain pTM does not by itself imply a poorly predicted ligand or
binding pose.** pTM uses a length-dependent distance scale that becomes extremely
strict for small chains. AlphaFold 3 explicitly warns that chains with fewer than
20 tokens can have pTM below 0.05 even when PAE and pLDDT indicate good confidence
([AF3 confidence documentation](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md#confidence-metrics)).
A small-molecule ligand contributes one token per heavy atom; a polymer normally
contributes one per residue, so this also matters for short peptides.

The public implementations differ in which token count sets that scale:

| Provider | Length scaling relevant to FoldQC's Chain pTM |
|---|---|
| AlphaFold 3 | Uses the individual chain's token count; small ligands can receive very low pTM. |
| [Chai-1](https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/ranking/ptm.py) | Uses the scored chain's token count. The effect applies to diagonal self-scores and to off-diagonal entries in a small ligand's column. |
| [Protenix](https://github.com/bytedance/Protenix/blob/main/protenix/model/sample_confidence.py) | Uses the selected chain's token count for chain pTM, so the same small-chain effect can occur. |
| [OpenFold3](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/metrics/confidence.py) | Uses the selected chain's token count for chain pTM, so the same small-chain effect can occur. |
| Public [Boltz-1](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/confidence_utils.py) / [Boltz-2](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/confidence_utils.py) | Uses the whole complex's token count even for diagonal chain self-scores. A small ligand in a large complex therefore avoids this particular short-chain penalty. Hosted Boltz Lab/API versions may differ. |

FoldQC preserves these provider scores; normalizing chain order and filling
missing diagonal entries does not harmonize their length scaling. Do not compare
absolute Chain pTM values across providers as if they used the same definition.
The caveat also applies to diagonal cells in either chain metric's Matrix plot
and to Chain ipTM when it falls back to per-chain pTM. Chai-1's asymmetric
off-diagonal scores require care too: inspect both directions rather than
treating a low ligand-column score alone as evidence of poor placement.

For a small ligand, inspect pLDDT together with protein-ligand PAE around the
binding pocket and interface confidence interpreted using the provider's
definition. Low intra-ligand PAE supports a coherent internal arrangement but
does not establish correct placement in the protein. A pTM value of 0.04 is a
score, not a 4% probability that the prediction is correct.

## Analyze interfaces

**Interfaces** opens one non-modal **Analyze interfaces** report for the current model,
selected ensemble member, or already-loaded ensemble. The action is independent
of **Color by** and requires at least two protein chains in a loaded target.
It does not load additional prediction ranks. Small molecules, nucleic acids,
and modified residues represented as individual HETATM atom tokens are excluded;
the report identifies excluded modified residues.

The report uses **original prediction coordinates and representative-atom
confidence**. PyMOL alignment, coordinate transformations, and B-factor coloring
do not alter these inputs.
The report reads Cα or Cβ atom confidence where the reference requires it,
including Cα for glycine, rather than substituting residue-averaged pLDDT.
Fractional confidence is converted to the published 0–100 scale. Missing or
ambiguous atoms/confidence produce **N/A with a reason**, not a zero score.

**Chain pairs** lists all unordered protein-chain pairs, including pairs with
no contacts. Click a score for its definition, parameters, direction-specific
values, confidence source, support counts, contributors, and citation. **Chain
versus rest** presents original Zhu pDockQ2 for each chain against all other
protein chains, including dimers.

The **Cβ pairs / Cα pairs** columns count geometric residue pairs. Details also
show the number of **unique contacting residues on A and B/rest**, the atom
convention, and the inclusive 8 Å cutoff. One residue may contact several partner
residues, so pair counts can exceed either residue count. Chain-pair diagnostics
use Cβ (Cα for glycine); chain-versus-rest diagnostics use Cα and count each
partner residue once across the rest chains. These counts depend only on original
geometry and remain available when PAE or pLDDT is missing. Missing or ambiguous
required coordinates give N/A with a reason; valid interfaces without contacts
give zero counts. Ensemble rows have no averaged contact diagnostic.

**Contacts CSV** exports every geometric contact for the selected model and
chain pair or chain-versus-rest row, including when selected through the heatmap.
It uses that row's atom convention, independently of the metric or contributor
side selected in the details controls, and applies no PAE filter. This export
requires a current snapshot with available geometry; aggregate rows cannot be
exported as contact lists. The existing **Export CSV** continues to export the
complete report, including stale snapshots. Contact exports run in the background
and can be canceled; failed or canceled exports preserve an existing output file.

Header tooltips and score details identify each metric's implementation and
aggregation. CSV exports retain full precision. The report presents Kim LIS;
the strict-cutoff Dunbrack LIS variant is also included in the CSV export.

Details show named literature references, marking the ipSAE, LIS, and
FlyPredictome cLIS/iLIS sources as bioRxiv preprints, and link the pinned code
separately to identify implementation variants. pDockQ also links the
[Bryant et al. author correction](https://doi.org/10.1038/s41467-022-29480-5)
that specifies the corrected sigmoid coefficients used by FoldQC. Both pDockQ2
variants cite Zhu et al. (2023), with their different source implementations
identified separately. Provider pair ipTM cites the provider output rather than
assigning it a generic metric-paper reference. The CSV `citation` field retains
the primary source.

| Implementation | Calculation and scope |
|---|---|
| ipSAE — Dunbrack | For each alignment residue, retain opposite-chain residues with PAE strictly below the preset (10 Å by default, or 15 Å). Set `d0 = max(1, 1.24*cbrt(max(n,26)-15)-1.8)` from the number `n` of qualifying residues. Average `1/(1+(PAE/d0)^2)` over that set, then take the maximum over alignment residues. Retain both chain directions and display their maximum. No geometric distance filter. |
| LIS — Kim | Mean `1-PAE/12` over PAE ≤12 Å in each direction; average the two directional means. Cells exactly at the cutoff contribute zero but remain in the denominator. |
| LIS — Dunbrack | Same transformation, with a strict PAE <12 Å cutoff; average the two directional means, matching the reference pair summary. A direction with no qualifying pairs contributes zero. |
| cLIS — Kim | Kim LIS restricted to Cβ–Cβ contacts ≤8 Å (Cα for glycine), with a mean over directions. |
| iLIS — Kim | `sqrt(LIS_Kim * cLIS_Kim)`, calculated after directional averaging. |
| pDockQ — Bryant | Corrected sigmoid `0.724/(1+exp(-0.052*(x-152.611)))+0.018`, where `x = mean_interface_pLDDT * log10(contact_pair_count)`. Cβ contacts ≤8 Å (Cα for glycine); interface residues from both chains counted once each. Requires no PAE. |
| pDockQ2 — Dunbrack | Pairwise Cβ contacts ≤8 Å (Cα for glycine), unique interface-residue pLDDT pooled across both chains, multiplied by directional mean `1/(1+(PAE/10)^2)`. Apply `1.31/(1+exp(-0.075*(x-84.733)))+0.005`; display the larger direction. No contacts returns zero. |
| pDockQ2 — Zhu original | Chain versus all other protein chains, using Cα contacts ≤8 Å. Scored-chain Cα pLDDT is weighted by contact multiplicity; multiply by the mean transformed directional PAE over these contacts. Original sigmoid constants are `L=1.31034849`, `x0=84.7326239`, `k=0.0747157696`, `b=0.00501886443`. With no contacts the sigmoid baseline (~0.00735) remains. |
| Pair ipTM — provider | Existing normalized off-diagonal pair matrix only, with both directions retained and their maximum displayed. No diagonal pTM, global ipTM, or PAE-based reconstruction is substituted. |

The numerical definitions are pinned to [Dunbrack/IPSAE
6174cf9](https://github.com/DunbrackLab/IPSAE/blob/6174cf9e71cb1bd660cc805856a18c4871a6dec3/ipsae.py),
[Kim/AFM-LIS
9921567](https://github.com/flyark/AFM-LIS/blob/9921567e186e0caa14cdbcd6bc51b9b02b36e36d/lis.py),
and [Zhu/AFM-Benchmark
5866c6d](https://gitlab.com/ElofssonLab/afm-benchmark/-/blob/5866c6d45abc802217a618290f76390999a83461/src/pdockq2.py).

The **Heatmap** shows one model or the ensemble mean, with the original chain
order, a fixed reversed-viridis 0–1 scale (low values bright, high values dark),
an undefined diagonal, and grey N/A cells.
Directional views have alignment chains in rows and scored chains in columns.
Clicking a cell selects the corresponding table row without modifying PyMOL.
For incompatible ensemble residue identities there is no single heatmap value.

**Select** creates or replaces the FoldQC-owned
`foldqc_interface` selection for both sides, A, or B/rest. It selects the chosen
metric's actual low-PAE or geometric/contact-filtered contributors, without
changing colors, B-factors, camera, or object visibility. Selection is disabled
for ensemble aggregate rows. Clear selections in the viewer when needed. Closing
the report with the operating system's window controls removes only the selection
created by this report.

Ensembles retain member rows plus means, population standard deviations,
minima, maxima, and contributing-model counts. Statistics combine per-model
scores only for matching chain/residue identity sets; mismatched members remain
separate. Missing PAE does not prevent geometry-only pDockQ. If available data
cannot be loaded or are malformed, the refresh reports an error and preserves
the previous results.

The report is a snapshot. Change the preset and press **Analyze target** to
refresh. The header records the preset used by the displayed results. Changing
the main target does not automatically recalculate; stale
snapshots remain readable/exportable but cannot select residues on replacement
models or generate contact-list exports. Failed/canceled refreshes leave the previous report intact. The window
closes with the plugin.

These are confidence/quality estimates, **not binding probabilities or affinity
measurements**. The [ipSAE manuscript](https://doi.org/10.1101/2025.02.10.637595)
discusses cutoff and short-interface sensitivity. The
[iLIS manuscript](https://doi.org/10.64898/2026.04.14.718529) calibrates 0.223
for top-model scores at a 10% false-positive rate using 774 positive PPIs
(yeast/fly/human Y2H reference sets and ELM interactions) and 6,004 controls
(random reference sets, GFP and Wingless controls). It is not a universal
threshold, and ensemble means need different
calibration. The [pDockQ2 paper](https://doi.org/10.1093/bioinformatics/btad424)
estimates protein-interface quality, not interaction strength. FoldQC applies
no automatic binder classification. Imported AFDB scalar summaries remain
separate from locally calculated, explicitly scoped results.

### Interface CSV schema

**Export CSV** exports the complete snapshot, regardless of tab, sort order, or
selected row. This is a separate schema (`interface_schema_version=2`); the
existing token CSV schema remains unchanged. Each row contains one model/scope/
metric/direction or one ensemble statistic.

| Columns | Meaning |
|---|---|
| `provider`, `prediction_name`, `input_path`, `target_name` | Prediction provenance and captured analysis target. |
| `model_rank`, `model_label`, `structure_path` | Member provenance; empty for ensemble aggregates. |
| `scope`, `chain_a`, `partner_chains` | Chain pair or chain-versus-rest; partner chain IDs are a JSON array. |
| `metric_key`, `metric_label`, `value_units`, `value_semantics` | Stable implementation-specific metric identity; dimensionless `score` (or native `iptm`), higher is better. |
| `implementation`, `implementation_revision`, `citation`, `parameters` | Source revision, scientific citation, and JSON parameter mapping. |
| `direction`, `aggregation`, `confidence_source` | Summary or A→B/B→A direction; metric aggregation or ensemble statistic; confidence input provenance. |
| `value`, `status`, `reason` | Numeric value or empty field; `ok`/`unavailable`; explanatory text including special no-contact behavior. |
| `counts`, `interface_plddt` | JSON support/contact counts and representative-atom interface confidence. LIA/cLIA combined counts sum directional qualifying cells, so a pair qualifying both ways contributes twice. |
| `contact_diagnostics` | JSON geometry-only diagnostic for the model/scope, repeated on its metric/direction rows: `scope`, `chain_a`, `partners`, `convention` (`cb_ca_gly` or `ca`), `cutoff_angstrom`, `comparison` (`<=`), `pair_count`, `residues_a`, `residues_b`, `status`, and `reason`. Unavailable counts are JSON `null`; aggregate rows leave this field empty. Added in v2; existing score/support fields retain their meaning. |
| `contributors_a`, `contributors_b` | JSON arrays of token index, chain ID, lossless residue ID (including insertion code), and residue name. |
| `n_models`, `member_ranks`, `residue_identities` | Coverage and JSON member ranks; aggregate identity groups. Ensemble rows provide mean, population standard deviation, minimum, maximum, and directional means where defined. |

No palette, color range, or PyMOL selection expression is exported. Coloring and
per-residue plots are not part of this interface window release.

### Contact CSV schema

**Contacts CSV** uses its own `contact_schema_version=1`, with one row per
geometric contact, ordered by canonical token index A then B. It contains:

| Columns | Meaning |
|---|---|
| `provider`, `prediction_name`, `input_path`, `target_name`, `model_rank`, `model_label`, `structure_path` | Captured prediction, target, and model provenance. Distances use original prediction coordinates. |
| `scope`, `chain_a`, `partner_chains` | Selected interface scope and partners (JSON chain-ID array). For chain versus rest, each endpoint B retains its actual partner chain. |
| `convention`, `cutoff_angstrom`, `comparison` | `cb_ca_gly` for Cβ/Cα-glycine chain pairs or `ca` for Cα chain versus rest; distance ≤8 Å. |
| `token_index_a/b`, `chain_id_a/b`, `residue_number_a/b`, `insertion_code_a/b`, `res_name_a/b`, `atom_name_a/b` | Separate columns for both endpoints, preserving zero-based token indices, residue numbering/insertion codes, and actual `CA`/`CB` representative names. |
| `distance_angstrom` | Representative-atom distance in Å, exported at full precision. |

A valid zero-contact interface writes a header-only file. Unavailable geometry
cannot be exported. Contact lists are generated on demand from the current
snapshot's canonical structure; stale snapshots retain their compact counts but
must be analyzed again before exporting a pair list.
