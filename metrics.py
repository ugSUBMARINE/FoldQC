"""Immutable metric and plot registries for FoldQC.

This module is independent of Qt and molecular viewers.  It is the single
source for user-facing metric metadata, data/context requirements, plot
routing, optional dependencies, and export semantics.  Numerical callables
remain in :mod:`compute` and :mod:`properties`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import Literal

from .dependencies import DependencyKey, required_dependency_keys
from .loader_models import DataCapability

MetricTier = Literal["normal", "advanced", "experimental"]
DataRequirement = Literal["plddt", "pae", "pde", "contact_probs", "confidence"]
PlotType = Literal[
    "line",
    "distribution",
    "matrix",
    "pae_summary",
    "pde_summary",
    "binding_site_fingerprint",
    "ensemble_site_summary",
]
MatrixSource = Literal["pae", "pde", "contact_probs", "chain_iptm"]
ValueUnit = Literal["plddt", "angstrom", "probability", "iptm", "label"]
ValueSemantics = Literal["higher_is_better", "lower_is_better", "categorical_label"]
AggregateKind = Literal[
    "single_model", "ensemble_mean", "ensemble_std", "ensemble_rmsd"
]
DEFAULT_METRIC_KEY = "plddt_class"
_PREVIEW_FIELDS = frozenset({"target_text", "ref_sel", "cutoff"})
_DATA_CAPABILITIES = frozenset({"plddt", "pae", "pde", "contact_probs"})
_DATA_REQUIREMENTS = _DATA_CAPABILITIES | {"confidence"}
_METRIC_TIERS = frozenset({"normal", "advanced", "experimental"})
_PLOT_TYPES = frozenset(
    {
        "line",
        "distribution",
        "matrix",
        "pae_summary",
        "pde_summary",
        "binding_site_fingerprint",
        "ensemble_site_summary",
    }
)
_MATRIX_SOURCES = frozenset({"pae", "pde", "contact_probs", "chain_iptm"})
_VALUE_UNITS = frozenset({"plddt", "angstrom", "probability", "iptm", "label"})
_VALUE_SEMANTICS = frozenset(
    {"higher_is_better", "lower_is_better", "categorical_label"}
)
_AGGREGATE_KINDS = frozenset(
    {"single_model", "ensemble_mean", "ensemble_std", "ensemble_rmsd"}
)


@dataclass(frozen=True)
class MatrixSpec:
    """Matrix data and presentation metadata for one metric family."""

    source: MatrixSource
    title: str
    colorbar_label: str


@dataclass(frozen=True)
class MetricSpec:
    """Complete declarative contract for one user-visible metric."""

    key: str
    label: str
    group: str
    requirements: frozenset[DataRequirement]
    value_unit: ValueUnit
    value_semantics: ValueSemantics
    line_ylabel: str
    preview_template: str
    details_template: str
    tier: MetricTier = "normal"
    needs_reference: bool = False
    needs_contact_shell: bool = False
    needs_cutoff: bool = False
    ensemble_level: bool = False
    plot_modes: frozenset[PlotType] = frozenset({"line", "distribution"})
    reference_scoped_plots: frozenset[PlotType] = frozenset()
    matrix: MatrixSpec | None = None
    aggregate_kind: AggregateKind = "single_model"
    dependency_keys: tuple[DependencyKey, ...] = ()

    @property
    def load_capabilities(self) -> frozenset[DataCapability]:
        return frozenset(
            requirement
            for requirement in self.requirements
            if requirement in _DATA_CAPABILITIES
        )

    @property
    def needs_confidence(self) -> bool:
        return "confidence" in self.requirements

    @property
    def is_domain_label(self) -> bool:
        return self.value_semantics == "categorical_label"


@dataclass(frozen=True)
class PlotSpec:
    """Declarative contract for one plot-menu action."""

    key: PlotType
    label: str
    requires_metric: bool = True
    dependency_keys: tuple[DependencyKey, ...] = ("matplotlib",)


class MetricRegistry(Sequence[MetricSpec]):
    """Ordered, immutable, validated collection of metric specifications."""

    def __init__(self, specs: Sequence[MetricSpec]) -> None:
        self._specs = tuple(specs)
        self._by_key = MappingProxyType(_validate_metric_specs(self._specs))

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index):
        return self._specs[index]

    def __iter__(self) -> Iterator[MetricSpec]:
        return iter(self._specs)

    def find(self, key: str | None) -> MetricSpec | None:
        if key is None:
            return None
        return self._by_key.get(str(key))

    def require(self, key: str) -> MetricSpec:
        spec = self.find(key)
        if spec is None:
            raise KeyError(f"Unknown metric: {key}")
        return spec

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self._specs)


class PlotRegistry(Sequence[PlotSpec]):
    """Ordered, immutable, validated collection of plot specifications."""

    def __init__(self, specs: Sequence[PlotSpec]) -> None:
        self._specs = tuple(specs)
        keys = [spec.key for spec in self._specs]
        if len(keys) != len(set(keys)):
            raise ValueError("Plot keys must be unique.")
        for spec in self._specs:
            if spec.key not in _PLOT_TYPES:
                raise ValueError(f"Unknown plot key: {spec.key!r}.")
            required_dependency_keys(spec.dependency_keys)
        self._by_key = MappingProxyType({spec.key: spec for spec in self._specs})

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index):
        return self._specs[index]

    def __iter__(self) -> Iterator[PlotSpec]:
        return iter(self._specs)

    def find(self, key: str | None) -> PlotSpec | None:
        if key is None:
            return None
        return self._by_key.get(str(key))  # type: ignore[arg-type]

    def require(self, key: str) -> PlotSpec:
        spec = self.find(key)
        if spec is None:
            raise KeyError(f"Unknown plot type: {key}")
        return spec


def _validate_metric_specs(specs: tuple[MetricSpec, ...]) -> dict[str, MetricSpec]:
    by_key: dict[str, MetricSpec] = {}
    for spec in specs:
        if not spec.key:
            raise ValueError("Metric keys must not be empty.")
        if spec.key in by_key:
            raise ValueError(f"Metric keys must be unique: {spec.key!r}.")
        unknown_requirements = set(spec.requirements) - _DATA_REQUIREMENTS
        if unknown_requirements:
            raise ValueError(
                f"Metric {spec.key!r} has unknown data requirements: "
                f"{sorted(unknown_requirements)!r}."
            )
        if spec.tier not in _METRIC_TIERS:
            raise ValueError(f"Metric {spec.key!r} has unknown tier {spec.tier!r}.")
        unknown_plots = set(spec.plot_modes) - _PLOT_TYPES
        if unknown_plots:
            raise ValueError(
                f"Metric {spec.key!r} has unknown plot modes: {sorted(unknown_plots)!r}."
            )
        if spec.value_unit not in _VALUE_UNITS:
            raise ValueError(
                f"Metric {spec.key!r} has unknown value unit {spec.value_unit!r}."
            )
        if spec.value_semantics not in _VALUE_SEMANTICS:
            raise ValueError(
                f"Metric {spec.key!r} has unknown semantics {spec.value_semantics!r}."
            )
        if spec.aggregate_kind not in _AGGREGATE_KINDS:
            raise ValueError(
                f"Metric {spec.key!r} has unknown aggregate kind "
                f"{spec.aggregate_kind!r}."
            )
        if spec.matrix is not None and spec.matrix.source not in _MATRIX_SOURCES:
            raise ValueError(
                f"Metric {spec.key!r} has unknown matrix source {spec.matrix.source!r}."
            )
        required_dependency_keys(spec.dependency_keys)
        if not spec.preview_template:
            raise ValueError(f"Metric {spec.key!r} requires a preview template.")
        fields = {
            name
            for _literal, name, _format_spec, _conversion in Formatter().parse(
                spec.preview_template
            )
            if name is not None
        }
        unknown_fields = fields - _PREVIEW_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Metric {spec.key!r} has unknown preview fields: "
                f"{sorted(unknown_fields)!r}."
            )
        required_preview_fields = set()
        if spec.needs_reference:
            required_preview_fields.add("ref_sel")
        if spec.needs_cutoff:
            required_preview_fields.add("cutoff")
        missing_preview_fields = required_preview_fields - fields
        if missing_preview_fields:
            raise ValueError(
                f"Metric {spec.key!r} preview is missing required fields: "
                f"{sorted(missing_preview_fields)!r}."
            )
        if not spec.details_template:
            raise ValueError(f"Metric {spec.key!r} requires a details template.")
        details_fields = {
            name
            for _literal, name, _format_spec, _conversion in Formatter().parse(
                spec.details_template
            )
            if name is not None
        }
        unknown_details_fields = details_fields - _PREVIEW_FIELDS
        if unknown_details_fields:
            raise ValueError(
                f"Metric {spec.key!r} has unknown details fields: "
                f"{sorted(unknown_details_fields)!r}."
            )
        missing_details_fields = required_preview_fields - details_fields
        if missing_details_fields:
            raise ValueError(
                f"Metric {spec.key!r} details are missing required fields: "
                f"{sorted(missing_details_fields)!r}."
            )
        if spec.needs_contact_shell and not (
            spec.needs_reference and spec.needs_cutoff
        ):
            raise ValueError(
                f"Metric {spec.key!r} contact shells require reference and cutoff."
            )
        if (spec.matrix is None) != ("matrix" not in spec.plot_modes):
            raise ValueError(
                f"Metric {spec.key!r} matrix metadata and plot mode disagree."
            )
        if spec.matrix is not None:
            matrix_requirement = (
                "confidence"
                if spec.matrix.source == "chain_iptm"
                else spec.matrix.source
            )
            if matrix_requirement not in spec.requirements:
                raise ValueError(
                    f"Metric {spec.key!r} matrix source {spec.matrix.source!r} "
                    "is not one of its data requirements."
                )
        if spec.ensemble_level and spec.aggregate_kind == "single_model":
            raise ValueError(
                f"Ensemble metric {spec.key!r} requires an ensemble aggregate kind."
            )
        if not spec.ensemble_level and spec.aggregate_kind != "single_model":
            raise ValueError(
                f"Single-model metric {spec.key!r} cannot declare an ensemble aggregate."
            )
        if not spec.reference_scoped_plots <= spec.plot_modes:
            raise ValueError(
                f"Metric {spec.key!r} reference-scoped plots must be supported plots."
            )
        by_key[spec.key] = spec
    return by_key


PAE_MATRIX = MatrixSpec("pae", "Predicted Aligned Error (Å)", "PAE (Å)")
PDE_MATRIX = MatrixSpec("pde", "Predicted Distance Error (Å)", "PDE (Å)")
CONTACT_MATRIX = MatrixSpec("contact_probs", "Interaction probability", "Probability")
CHAIN_IPTM_MATRIX = MatrixSpec("chain_iptm", "Pairwise chain ipTM", "ipTM")

_LINE_DISTRIBUTION = frozenset({"line", "distribution"})
_LINE_DISTRIBUTION_MATRIX = frozenset({"line", "distribution", "matrix"})
_DISTRIBUTION_MATRIX = frozenset({"distribution", "matrix"})

METRICS = MetricRegistry(
    (
        MetricSpec(
            "plddt_class",
            "pLDDT — classes",
            "pLDDT",
            frozenset({"plddt"}),
            "plddt",
            "higher_is_better",
            "pLDDT",
            "Applies AlphaFold pLDDT confidence classes to {target_text}.",
            "Uses the canonical token-level pLDDT values for {target_text} and "
            "applies the four AlphaFold confidence classes: very high (at least "
            "90), high (70 to below 90), low (50 to below 70), and very low "
            "(below 50). Blue and light-blue regions have higher predicted local "
            "accuracy; yellow and orange regions should be interpreted more "
            "cautiously.",
            reference_scoped_plots=_LINE_DISTRIBUTION,
        ),
        MetricSpec(
            "plddt",
            "pLDDT — continuous",
            "pLDDT",
            frozenset({"plddt"}),
            "plddt",
            "higher_is_better",
            "pLDDT",
            "Colors {target_text} by continuous local confidence (pLDDT).",
            "Colors {target_text} with its canonical token-level pLDDT values on "
            "a continuous scale. Higher values indicate greater confidence in "
            "the local structure around a residue or ligand atom; low values can "
            "mark flexible, disordered, or inaccurately modeled regions.",
            reference_scoped_plots=_LINE_DISTRIBUTION,
        ),
        MetricSpec(
            "pae_row_mean",
            "PAE — row mean",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            "Colors each token in {target_text} by how well the rest of the model is positioned when aligned on that token.",
            "For every token i in {target_text}, calculates the mean of PAE[i, :] "
            "over the other tokens. This is the average uncertainty in the rest "
            "of the model when the predicted and true structures are aligned on "
            "token i. Lower values identify better-anchored tokens; higher values "
            "often identify flexible or poorly positioned regions.",
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_col_mean",
            "PAE — column mean",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            "Colors each token in {target_text} by its average positional uncertainty across all alignment frames.",
            "For every token j in {target_text}, calculates the mean of PAE[:, j] "
            "over all alignment frames. This estimates how uncertain token j's "
            "position is from the perspective of all other tokens. Lower values "
            "indicate more globally consistent placement. Because PAE is "
            "directional, this can differ from the row mean.",
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_to_sel",
            "PAE — row mean to selection",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            'Colors each token in {target_text} by directional PAE from that token to "{ref_sel}".',
            "For every token i in {target_text}, calculates mean PAE[i, reference] "
            'using the tokens selected by "{ref_sel}". It measures how confidently '
            "the reference is positioned when alignment is based on token i. "
            "Lower values indicate more confident relative placement. PAE is "
            "directional, so this is not interchangeable with the column-mean "
            "variant.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_col_to_sel",
            "PAE — column mean to selection",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            'Colors each token in {target_text} by directional PAE from "{ref_sel}" to that token.',
            "For every token j in {target_text}, calculates mean PAE[reference, j] "
            'using the tokens selected by "{ref_sel}". It measures how confidently '
            "token j is positioned in alignment frames based on the reference. "
            "Lower values indicate more confident placement from the reference "
            "frame. This is the opposite PAE direction from row mean to selection.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_sym_sel",
            "PAE — symmetric mean to selection",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            'Colors {target_text} by bidirectional mean PAE between each token and "{ref_sel}".',
            "Averages both PAE directions between every token in {target_text} and "
            'the tokens selected by "{ref_sel}". Reference tokens are scored '
            "against non-reference tokens. Lower values indicate more mutually "
            "confident relative placement and make the result less sensitive to "
            "PAE directionality than either one-way selection mean.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_sym_within_sel",
            "PAE — symmetric mean within selection",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            'Colors only tokens in "{ref_sel}" within {target_text} by their internal symmetric PAE.',
            'For only the tokens selected by "{ref_sel}" in {target_text}, '
            "calculates mean symmetric PAE against the same selected token set. "
            "Lower values indicate that the selected ligand, residue set, or "
            "chain is predicted as a more coherent internal arrangement. "
            "Non-selected tokens are undefined and remain grey.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_contact",
            "PAE — contact-filtered to selection",
            "PAE",
            frozenset({"pae"}),
            "angstrom",
            "lower_is_better",
            "PAE (Å)",
            'Colors polymer binding-site residues in {target_text} within {cutoff} of "{ref_sel}" by mean PAE to the reference.',
            "Finds non-reference polymer residues in {target_text} with any atom "
            'within {cutoff} of "{ref_sel}", then calculates their symmetric mean '
            "PAE to the reference tokens. This focuses the result on predicted "
            "binding-site contacts, including for outputs without PDE. Lower "
            "values indicate more reliable relative placement; residues outside "
            "the contact shell or without a value remain grey.",
            tier="advanced",
            needs_reference=True,
            needs_contact_shell=True,
            needs_cutoff=True,
            plot_modes=_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=PAE_MATRIX,
        ),
        MetricSpec(
            "pae_domain_complete",
            "PAE — domain labels (complete linkage)",
            "PAE",
            frozenset({"pae"}),
            "label",
            "categorical_label",
            "Domain label",
            "Colors {target_text} with categorical rigid-domain labels by grouping tokens whose pairwise symmetric PAE stays within the {cutoff} threshold.",
            "Builds a hierarchy from pairwise symmetric PAE distances in "
            "{target_text}, then cuts it at the {cutoff} threshold using complete "
            "linkage. Tokens share a domain only when all pairwise distances "
            "within that cluster satisfy the threshold. The labels are categorical: "
            "their numbers and colors do not rank confidence. Ensemble members "
            "are clustered independently. Requires SciPy.",
            tier="experimental",
            needs_cutoff=True,
            plot_modes=_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_DISTRIBUTION_MATRIX,
            matrix=PAE_MATRIX,
            dependency_keys=("scipy",),
        ),
        MetricSpec(
            "pae_domain_spectral",
            "PAE — domain labels (spectral clustering)",
            "PAE",
            frozenset({"pae"}),
            "label",
            "categorical_label",
            "Domain label",
            "Colors {target_text} with categorical heuristic PAE domain labels by spectral clustering of a symmetric PAE affinity graph using the {cutoff} threshold as a scale.",
            "Converts symmetric PAE for {target_text} into an affinity graph using "
            "{cutoff} as its scale, estimates up to 12 clusters from the normalized-"
            "Laplacian eigengap, and applies spectral clustering. This heuristic can "
            "find softer domain boundaries than complete linkage. Labels are "
            "categorical rather than ordered confidence values, and ensemble "
            "members are clustered independently. Requires SciPy and scikit-learn.",
            tier="experimental",
            needs_cutoff=True,
            plot_modes=_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_DISTRIBUTION_MATRIX,
            matrix=PAE_MATRIX,
            dependency_keys=("scipy", "sklearn"),
        ),
        MetricSpec(
            "pde_mean",
            "PDE — mean",
            "PDE",
            frozenset({"pde"}),
            "angstrom",
            "lower_is_better",
            "Distance / error (Å)",
            "Colors each token in {target_text} by its average predicted distance error to all other tokens.",
            "For every token i in {target_text}, calculates mean PDE[i, :] over all "
            "tokens. This summarizes the predicted error in pairwise distances "
            "between token i and the whole model. Lower values indicate that its "
            "distances to the surrounding structure are predicted more reliably.",
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=PDE_MATRIX,
        ),
        MetricSpec(
            "pde_chain_mean",
            "PDE — within-chain mean",
            "PDE",
            frozenset({"pde"}),
            "angstrom",
            "lower_is_better",
            "Distance / error (Å)",
            "Colors each token in {target_text} by predicted distance error within its own chain.",
            "For every token in {target_text}, calculates mean PDE only against "
            "tokens belonging to the same chain. It isolates intra-chain distance "
            "confidence from inter-chain placement. Lower values identify chain "
            "regions with more reliably predicted internal geometry.",
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=PDE_MATRIX,
        ),
        MetricSpec(
            "pde_to_sel",
            "PDE — mean to selection",
            "PDE",
            frozenset({"pde"}),
            "angstrom",
            "lower_is_better",
            "Distance / error (Å)",
            'Colors each token in {target_text} by predicted distance error to "{ref_sel}".',
            "For every token i in {target_text}, calculates mean PDE[i, reference] "
            'using the tokens selected by "{ref_sel}". It summarizes predicted '
            "distance error between each token and the chosen ligand, residue set, "
            "or chain. Lower values indicate more reliable distances to the "
            "reference.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=PDE_MATRIX,
        ),
        MetricSpec(
            "pde_within_sel",
            "PDE — within-selection mean",
            "PDE",
            frozenset({"pde"}),
            "angstrom",
            "lower_is_better",
            "Distance / error (Å)",
            'Colors only tokens in "{ref_sel}" within {target_text} by their internal predicted distance error.',
            'For only the tokens selected by "{ref_sel}" in {target_text}, '
            "calculates mean PDE against the same selected token set. Lower values "
            "indicate that the selected region has more reliably predicted internal "
            "distances. Non-selected tokens are undefined and remain grey.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=PDE_MATRIX,
        ),
        MetricSpec(
            "pde_contact",
            "PDE — contact-filtered to selection",
            "PDE",
            frozenset({"pde"}),
            "angstrom",
            "lower_is_better",
            "Distance / error (Å)",
            'Colors polymer binding-site residues in {target_text} within {cutoff} of "{ref_sel}" by mean PDE to the reference.',
            "Finds non-reference polymer residues in {target_text} with any atom "
            'within {cutoff} of "{ref_sel}", then calculates their mean PDE to the '
            "reference tokens. This isolates predicted distance confidence in the "
            "binding-site contact shell. Lower values indicate more reliable "
            "contacts; residues outside the shell or without a value remain grey.",
            tier="advanced",
            needs_reference=True,
            needs_contact_shell=True,
            needs_cutoff=True,
            plot_modes=_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=PDE_MATRIX,
        ),
        MetricSpec(
            "contact_prob_mean",
            "Interaction probability — mean",
            "Interaction probability",
            frozenset({"contact_probs"}),
            "probability",
            "higher_is_better",
            "Interaction probability",
            "Colors each token in {target_text} by its average predicted interaction probability across the model.",
            "For every token in {target_text}, calculates the mean predicted "
            "interaction or contact probability against all other tokens. Higher "
            "values indicate broader predicted participation in contacts or "
            "interactions; they do not by themselves distinguish a specific "
            "binding partner or prove physical binding.",
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=_LINE_DISTRIBUTION_MATRIX,
            matrix=CONTACT_MATRIX,
        ),
        MetricSpec(
            "contact_prob_to_sel",
            "Interaction probability — mean to selection",
            "Interaction probability",
            frozenset({"contact_probs"}),
            "probability",
            "higher_is_better",
            "Interaction probability",
            'Colors each token in {target_text} by predicted interaction probability with "{ref_sel}".',
            "For every non-reference token in {target_text}, calculates mean "
            "predicted interaction or contact probability with the tokens selected "
            'by "{ref_sel}". Higher values indicate a stronger predicted interaction '
            "with that ligand, chain, or residue set. Reference tokens are undefined "
            "and remain grey so within-reference contacts do not dominate the scale.",
            tier="advanced",
            needs_reference=True,
            plot_modes=_LINE_DISTRIBUTION_MATRIX,
            reference_scoped_plots=frozenset({"matrix"}),
            matrix=CONTACT_MATRIX,
        ),
        MetricSpec(
            "ensemble_rmsd",
            "Ensemble RMSD, aligned",
            "Ensemble",
            frozenset(),
            "angstrom",
            "lower_is_better",
            "Distance / error (Å)",
            "Colors {target_text} by per-token coordinate variation in the loaded ensemble after alignment.",
            "After ensemble setup and alignment, calculates coordinate RMSD for "
            "each token across the loaded models and colors {target_text} with the "
            "shared result. Lower values indicate structural agreement between "
            "models; higher values indicate conformational variability or uncertain "
            "placement. The default setup aligns models on a high-confidence polymer "
            "core, while expert mode can use current viewer coordinates.",
            ensemble_level=True,
            reference_scoped_plots=_LINE_DISTRIBUTION,
            aggregate_kind="ensemble_rmsd",
        ),
        MetricSpec(
            "ensemble_plddt_mean",
            "Ensemble pLDDT mean",
            "Ensemble",
            frozenset({"plddt"}),
            "plddt",
            "higher_is_better",
            "pLDDT",
            "Colors {target_text} by the mean pLDDT at each token across models in the loaded ensemble.",
            "Calculates the arithmetic mean of canonical pLDDT at every token across "
            "the loaded ensemble and colors {target_text} with that aggregate. Higher "
            "values indicate local structure that is consistently assigned high "
            "confidence across models. A mean can conceal member-to-member variation, "
            "which is shown by the ensemble pLDDT standard deviation.",
            ensemble_level=True,
            reference_scoped_plots=_LINE_DISTRIBUTION,
            aggregate_kind="ensemble_mean",
        ),
        MetricSpec(
            "ensemble_plddt_std",
            "Ensemble pLDDT std",
            "Ensemble",
            frozenset({"plddt"}),
            "plddt",
            "lower_is_better",
            "pLDDT",
            "Colors {target_text} by how much pLDDT varies at each token across models in the loaded ensemble.",
            "Calculates the standard deviation of canonical pLDDT at every token "
            "across the loaded ensemble and colors {target_text} with that aggregate. "
            "Lower values indicate stable confidence estimates; higher values mark "
            "positions where confidence varies between models. This measures "
            "variation in predicted confidence, not coordinate variation.",
            ensemble_level=True,
            reference_scoped_plots=_LINE_DISTRIBUTION,
            aggregate_kind="ensemble_std",
        ),
        MetricSpec(
            "chain_iptm",
            "Chain ipTM",
            "Chain/interface",
            frozenset({"confidence"}),
            "iptm",
            "higher_is_better",
            "ipTM",
            "Colors chains in {target_text} by chain-level ipTM; use Plot > Matrix for pairwise chain ipTM.",
            "Assigns every token in each chain of {target_text} that chain's normalized "
            "ipTM score, falling back to per-chain pTM when needed. Higher values "
            "indicate greater confidence in chain-level placement or interactions. "
            "The matrix plot shows pairwise chain ipTM: off-diagonal cells describe "
            "chain pairs, while diagonal cells are chain-restricted pTM or self "
            "scores and should not be interpreted as interfaces.",
            plot_modes=frozenset({"line", "matrix"}),
            reference_scoped_plots=frozenset({"line"}),
            matrix=CHAIN_IPTM_MATRIX,
        ),
    )
)

PLOTS = PlotRegistry(
    (
        PlotSpec("line", "Line"),
        PlotSpec("distribution", "Distribution"),
        PlotSpec("matrix", "Matrix"),
        PlotSpec("pae_summary", "PAE summary", requires_metric=False),
        PlotSpec("pde_summary", "PDE summary", requires_metric=False),
        PlotSpec(
            "binding_site_fingerprint",
            "Binding-site fingerprint",
            requires_metric=False,
        ),
        PlotSpec(
            "ensemble_site_summary", "Ensemble site summary", requires_metric=False
        ),
    )
)

PLDDT_CLASS_STATS: tuple[tuple[str, float | None, float | None], ...] = (
    ("Very high (>=90)", 90.0, None),
    ("High (70-90)", 70.0, 90.0),
    ("Low (50-70)", 50.0, 70.0),
    ("Very low (<50)", None, 50.0),
)
PLDDT_CLASS_PLOT_LABELS = MappingProxyType(
    {
        "Very high (>=90)": "very high",
        "High (70-90)": "high",
        "Low (50-70)": "low",
        "Very low (<50)": "very low",
    }
)


def metric_label(key: str) -> str:
    spec = METRICS.find(key)
    return spec.label if spec is not None else key


def property_combo_label(spec: MetricSpec) -> str:
    label = spec.label
    if spec.tier == "advanced":
        label = f"{label} [Advanced]"
    elif spec.tier == "experimental":
        label = f"{label} [Experimental]"
    return f"  {label}"
