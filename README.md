# FoldQC

FoldQC is a PyMOL plugin for visualizing confidence metrics from predicted
protein structures. It supports Boltz prediction folders, AlphaFold 3 local and
server outputs, Chai-1 Discovery and Protenix prediction folders, and single
CIF/PDB files with pLDDT values stored in B-factors.

FoldQC can color structures by pLDDT, PAE, PDE, contact probability, chain
ipTM, and ensemble metrics. It also provides line plots, matrix plots, PAE/PDE
summary plots, binding site fingerprints, and ensemble summaries.

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

2. Start PyMOL.

3. Install missing Python dependencies from the PyMOL command line:

   ```text
   run /path/to/FoldQC/tools/install_deps.py
   ```

   You can also run the script from the menu: `File -> Run Script...` and select `install_deps.py` from the plugin `tools` folder.

4. Restart PyMOL.

5. Open the plugin from:

   ```text
   Plugins -> FoldQC...
   ```

## Usage

1. Open FoldQC from the PyMOL plugin menu.
2. Choose a prediction folder or a single `.cif`/`.pdb` structure file.
3. Select the model, target object, metric, and color palette.
4. Click the paint/color action to write the metric into B-factors and color the
   structure in PyMOL.
5. For selection-based metrics or site-focused plots, enter a PyMOL selection
   in the contextual Reference / Ligand-site field. The cutoff field is enabled
   for metrics that use it, such as contact-filtered PAE/PDE and PAE domain labels.
6. Use the `Plot` dropdown and ensemble actions for heatmaps, line plots,
   PAE/PDE summary plots, binding-site fingerprints, and multi-model summaries.

`PAE summary` and `PDE summary` are (experimental) speciality line plots for multi-chain
targets. They plot the gap between each token's mean error to other chains and
its mean error within its own chain (`other - within`); PAE summary shows
separate row and column gap lines. If a Reference selection is entered, only
the displayed x-range is restricted to those tokens; the summary values are
still computed against the full complex. Ensemble targets show member means
with shaded standard deviations.

## CSV Export Schema

Use `Export CSV...` to write token-level scalar values for the currently
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
| `export_schema_version` | CSV schema version. Currently `1`. |
| `provider` | Prediction provider, such as `boltz`, `alphafold3`, `af3_server`, `chai1`, `protenix`, or `structure_only`. |
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
| `res_num` | Residue number from the structure file token map. |
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

| Color by option | How it is calculated | How to interpret it |
| --- | --- | --- |
| pLDDT - quality classes | Reads pLDDT from structure B-factors when available, otherwise falls back to provider prediction arrays or JSON data. Applies the AlphaFold four-class coloring: very high >=90, high 70-90, low 50-70, very low <50. | Quick local-confidence overview. Blue/light blue regions are more reliable; yellow/orange regions should be treated cautiously. |
| pLDDT - continuous | Reads pLDDT from structure B-factors when available, otherwise falls back to provider prediction arrays or JSON data. Colors local confidence as a continuous value. | Higher values indicate higher local model confidence. The structure file remains the preferred source, including single CIF/PDB inputs. |
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
| Chain ipTM (from JSON) | Reads per-chain ipTM from the confidence JSON (`chains_iptm`, `chain_iptm`, or `chains_ptm`) and assigns each token the score for its chain. Matrix plots show the provider's pairwise chain matrix (`pair_chains_iptm` or `chain_pair_iptm`) with rows as chain `i` and columns as chain `j`. | Higher values indicate higher confidence in chain-level placement or interactions. Off-diagonal cells describe confidence for a chain pair; diagonal cells are chain-restricted pTM/self scores, not interfaces. |

Chai-1 Discovery folders provide structure B-factor pLDDT, PAE matrices, and
score JSON/NPZ files with global pTM/ipTM, per-chain pTM, pairwise chain ipTM,
and clash flags. PDE metrics are available for Chai only when matching
`pde*.npy` files are present. Interaction probability metrics are not exposed
for Chai-1 Discovery folders.

Protenix folders provide structure B-factor pLDDT, summary confidence JSONs,
global pTM/ipTM/gPDE, per-chain pTM/ipTM, pairwise chain ipTM, and clash flags.
PAE, PDE, interaction probability, and provider atom-pLDDT metrics are available
for Protenix only when matching `*_full_data_sample_*.json` files are present.

### Interpreting Pairwise Chain ipTM

The pairwise chain ipTM matrix uses the chain order from the prediction output,
mapped onto the displayed PyMOL chain IDs. Row `i`, column `j` means the JSON
cell `[i][j]` for those two chains.

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
