# FoldQC

FoldQC is a PyMOL plugin for visualizing confidence metrics from predicted
protein structures. It supports Boltz (local) prediction folders, Boltz Lab and Boltz
API outputs, AlphaFold 3 local and server outputs, Chai-1 Discovery and
Protenix prediction folders, zip/tar archives containing one supported
prediction folder, and single CIF/PDB files with pLDDT values stored in
B-factors.

FoldQC can color structures by pLDDT, PAE, PDE, contact probability, chain
ipTM, and ensemble metrics. It also provides line plots, matrix plots, PAE/PDE
summary plots, binding site fingerprints, and ensemble summaries.

PyMOL commands and selection syntax are isolated in `mol_viewer.py`; loading,
analysis, plotting, and GUI modules consume its viewer-neutral function facade.

Development of FoldQC has included coding assistance from OpenAI's Codex.

## Installation

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
   empty path and never loads a saved prediction automatically.
3. Select the model, target object, metric, and color palette.
   When FoldQC must create a model object in PyMOL, it initially applies the
   familiar pLDDT quality-class coloring. If the named object already exists,
   FoldQC reuses it without overwriting its current colors.
   For predictions with multiple ranks, `Compare` opens a read-only
   table of provider summary scores for every rank. Selecting a row switches to
   that model. The table reads compact scalar summaries such as ranking score,
   pTM, ipTM, and complex pLDDT without loading every structure or lazy PAE/PDE
   matrix; columns remain provider- and file-dependent. Prediction-level Boltz
   affinity values are intentionally excluded from this per-model table.
4. Click the paint/color action to write the metric into B-factors and color the
   structure in PyMOL.
5. Use the statistics panel's threshold control and `Select ≥` / `Select ≤`
   buttons to create a persistent named PyMOL selection from the metric values
   that were just colored. FoldQC uses its stored token map and metric array,
   not the object's current B-factors. For ensemble targets, each member is
   thresholded against its own values and combined into one selection. Reusing
   a metric/direction pair replaces its stable `foldqc_<metric>_ge` or
   `foldqc_<metric>_le` selection.
6. For selection-based metrics or site-focused plots, enter a PyMOL selection
   in the contextual Reference / Ligand-site field. The cutoff field is enabled
   for metrics that use it, such as contact-filtered PAE/PDE and PAE domain labels.
7. Use the `Plot` menu and ensemble actions for heatmaps, line plots,
   PAE/PDE summary plots, binding-site fingerprints, and multi-model summaries.

`Load Ensemble` is enabled only for predictions containing at least two
models and only until that prediction's ensemble has been activated. Its
tooltip explains when a single-model prediction or an already-loaded ensemble
makes the action unavailable.
The active ensemble group is shown in bold italic text in the PyMOL target
selector, distinguishing analyses that use ensemble data from member-object
analyses.
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

## Data and Confidence Contracts

FoldQC's metric and plot behavior is registry-derived. Each metric specification
declares its required data family, viewer context, supported plots, optional
dependencies, matrix presentation, and CSV units/semantics. The GUI resolves
that specification once when an action starts, then loads only the required
per-model capabilities. Availability therefore follows the selected rank and
target rather than a prediction-wide union.

Coloring, plotting, and CSV export capture the selected metric, target,
reference, cutoff, and action in one immutable analysis request. Lazy provider
data is loaded against the canonical model versions for that request. If the
prediction, model, ensemble, or relevant controls change while work is running,
the stale result is discarded instead of applying the old action to a new
target.

The same captured request also owns palette/range, plot, or export options, so
completion never rereads the dialog controls. Loading progress is modeless and
shown only after a short delay. Closing the plugin or explicitly cancelling a
workflow abandons its result safely; user cancellation and stale results stay
silent, while genuine provider or viewer failures are presented once.

Transient interaction is centralized: workflow notices, confirmations,
candidate/alignment choices, progress, statistics, and plot windows are
presented by the Qt presentation adapter. Ordinary control state
and previews are rendered separately from a typed context view state. Native
open/save dialogs remain in the main dialog because their paths are captured
before a workflow is submitted.

The dialog itself is only the Qt composition root. Concrete lifecycle,
acquisition, analysis, metric, coloring, plot, and export services receive
their collaborators explicitly and do not retain the dialog, its widgets, or a
service-locator object. A Qt-independent operation coordinator owns the one
serialized background-operation lease, progress lifetime, cancellation, and
stale-generation checks. This makes the asynchronous boundary both testable
with fake ports and visible to static type checking.

FoldQC painting is transactional. It resolves token-to-atom mappings before
mutation and snapshots the affected atoms' B-factors and color indices before
viewer updates are suspended. The managed colorbar is retained as a recreatable
typed specification: FoldQC deletes and recreates its ramp under the stable
managed name and never renames PyMOL ramp objects. If a multi-object paint or
colorbar operation fails, the previous atom values, colors, and managed
colorbar are restored.

Provider confidence JSON is normalized when parsed. Live model state contains
one immutable provider-neutral confidence object with these recognized fields
when available: `ranking_score`, `confidence_score`, `ptm`, `iptm`,
`ligand_iptm`, `protein_iptm`, `complex_plddt`, `complex_iplddt`, `complex_pde`,
`complex_ipde`, `fraction_disordered`, `gpde`, `has_clash`, per-chain pTM/ipTM,
pairwise chain ipTM, and affinity value/probability. Provider aliases such as
`aggregate_score`, `structure_confidence`, `disorder`, and the various chain
field spellings are accepted only at the provider boundary.
AF3's numeric `has_clash` values are accepted only when they are exactly `0`
or `1` and are converted immediately to the canonical boolean.
Chai-1 score outputs do not provide `fraction_disordered`, so FoldQC omits that
field from Chai summaries and comparisons.

Unknown provider JSON fields are deliberately discarded. Missing recognized
values remain unavailable; malformed recognized values report the provider,
model, field, and source file. Chain vectors and matrices are read-only
`float32` arrays indexed by canonical first-appearance `TokenMap.chain_order`.
Missing chain cells use `NaN`; missing or zero pair-matrix diagonal cells are
filled from per-chain pTM when that value exists.

For contributors, `uv run mypy` checks the Phase 5 service boundary with a
stable strict target. Qt, PyMOL, Matplotlib, providers, and older numerical
modules outside that target are intentionally not yet a repository-wide typing
baseline.

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
| `provider` | Prediction provider, such as `boltz`, `boltz_lab`, `boltz_api`, `alphafold3`, `af3_server`, `chai1`, `protenix`, or `structure_only`. |
| `prediction_name` | Provider-scanned prediction name. |
| `input_path` | Original selected folder, zip, CIF, or PDB path. |
| `structure_path` | Structure file used for token mapping in this session. |
| `model_rank` | Ranked model number used for the row's values. |
| `model_label` | User-facing model label from the prediction scan. |
| `metric_key` | Stable internal metric key, such as `pde_contact` or `ensemble_plddt_mean`. |
| `metric_label` | User-facing label from the `Color by` control. |
| `value` | Token-level scalar metric value. Undefined values are written as `nan`. |
| `value_units` | Machine-readable unit/category: `plddt`, `angstrom`, `probability`, `iptm`, or `label`. |
| `value_semantics` | Interpretation hint: `higher_is_better`, `lower_is_better`, or `categorical_label`. |
| `reference_selection` | Reference text used by selection-based metrics. Blank when unused. |
| `cutoff_angstrom` | Cutoff/threshold used by contact-filtered PAE/PDE or PAE domain labels. Blank when unused. |
| `token_index` | Zero-based token index in original prediction token order. |
| `token_type` | `polymer_residue` or `ligand_atom`. |
| `chain_id` | Chain ID from the structure file token map. |
| `res_num` | Numeric residue number from the structure file token map. |
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

## Color by Metric Reference

FoldQC colors token-level values. A token is one polymer residue or one heavy
atom for HETATM ligands. Continuous metrics are written into PyMOL B-factors and
colored with the selected palette; undefined values are colored grey. Selection
metrics use the PyMOL expression in the Reference field, mapped back to FoldQC
tokens.

Residue identity is lossless across CIF/PDB parsing, PyMOL selections, plots,
ensembles, and CSV export: chain, numeric residue number, insertion code, and
residue name all participate in matching. Ligand H, D, and T atoms are excluded
from tokenization, while their source atom positions remain represented in
provider atom-array alignment.

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

Each provider resolves one canonical token-level pLDDT array when it loads a
model. Boltz uses its token NPZ when present. All atom-level sources, including
provider confidence arrays and structure B-factors, average finite polymer atom
values by residue in prediction token order; ligand heavy atoms remain individual
tokens. Standalone structures, Chai, Boltz Lab, and Boltz API use structure
B-factors. Missing provider arrays fall back to structure B-factors; malformed
explicit arrays are reported as errors. Coloring, plots, exports, alignment, and
ensemble consensus all consume this same canonical array. This follows AlphaFold
3's documented
[`atom_plddts` per-atom semantics](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md#metrics-in-confidences-json).

PAE, PDE, and interaction-probability availability is tracked per ranked model,
not as a prediction-wide union. Strict ensemble coloring and ordinary metric
plots require every targeted member to provide the selected family. Binding-site
fingerprints and ensemble-site summaries instead aggregate the members that
provide each family and leave unavailable member values missing. Loaded arrays
are centrally validated against the model's canonical token count, normalized,
and made read-only before entering model state.

| Color by option | How it is calculated | How to interpret it |
| --- | --- | --- |
| pLDDT - classes | Uses the provider-selected canonical pLDDT array. Applies the AlphaFold four-class coloring: very high >=90, high 70-90, low 50-70, very low <50. | Quick local-confidence overview. Blue/light blue regions are more reliable; yellow/orange regions should be treated cautiously. |
| pLDDT - continuous | Uses the provider-selected canonical pLDDT array and colors local confidence as a continuous value. | Higher values indicate higher local model confidence. |
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
| Chain ipTM | Uses FoldQC's normalized per-chain ipTM (falling back to per-chain pTM) and assigns each token the score at that chain's position in canonical `TokenMap.chain_order`. Matrix plots use the normalized pairwise chain ipTM matrix. | Higher values indicate higher confidence in chain-level placement or interactions. Off-diagonal cells describe confidence for a chain pair; diagonal cells are chain-restricted pTM/self scores, not interfaces. |

Chai-1 Discovery folders provide structure B-factor pLDDT, PAE matrices, and
score JSON/NPZ files with global pTM/ipTM, per-chain pTM, pairwise chain ipTM,
and clash flags. PDE metrics are available for Chai only when matching
`pde*.npy` files are present. Interaction probability metrics are not exposed
for Chai-1 Discovery folders.

Protenix folders provide summary confidence JSONs, global pTM/ipTM/gPDE,
per-chain pTM/ipTM, pairwise chain ipTM, and clash flags. When a matching
`*_full_data_sample_*.json` contains atom pLDDT, those values are averaged into
the canonical token array; otherwise structure B-factors are used. PAE, PDE,
and interaction probability metrics are available only when their arrays are
present in the full-data file.

### Interpreting Pairwise Chain ipTM

The pairwise chain ipTM matrix uses the unique first-appearance chain order from
the canonical structure token map, mapped onto the displayed PyMOL chain IDs.
Provider JSON indices are converted to this order while loading; row `i`,
column `j` then represents those two canonical chains.

Do not assume every provider uses the matrix as a simple "reference chain" by
"placed chain" table. For AlphaFold 3 `chain_pair_iptm`, the off-diagonal
element `(i, j)` is documented as the ipTM restricted to tokens from chains `i`
and `j`; in the public AlphaFold 3 implementation this chain-pair score is
written symmetrically. In that case `(A, X)` and `(X, A)` should be the same
apart from rounding or output-version differences.

Boltz-style, Chai-1 Discovery, and Protenix `pair_chains_iptm` outputs can be
asymmetric. For those matrices, read `[i][j]` as the provider's directional
score for chain `i` against chain `j`. This is analogous to PAE directionality:
the row can be treated as the alignment/reference side and the column as the
side whose placement is being evaluated relative to it. Because the two
directions can differ, report both directions when diagnosing an interface. For
a single conservative interface summary, use `min([i][j], [j][i])`; for a
permissive summary, use the mean or maximum, but state which aggregation you
used.
