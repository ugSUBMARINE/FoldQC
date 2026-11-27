# FoldQC examples: reproducing the figures

Boltz-2 predictions to load in [FoldQC](https://github.com/ugSUBMARINE/FoldQC), the PyMOL plugin described in the application note. Each case folder can be opened directly in FoldQC. The folder covers Case studies 1 and 2 (Figs 1-4); for Case study 3 (Fig. 5) the structure is fetched from the AlphaFold Database, see [Fig. 5](#fig-5-atgl-abhd5-case-study-3).

## Contents

```
case1-cln6/        CLN6 with R,S-GPG (ligand CCD code CU7) and divalent ions
case2-atgl_fa/     ATGL residues 1-254 with a saturated fatty acid bonded to Ser47
  fac08/   C8
  fac16/  C16
  fac24/   C24
```

### Case study 1: CLN6

| folder | ions | used for |
|---|---|---|
| `cln6_RS-GPG` | none | Fig. 2 (`model_0`), Fig. 3 (all 25 samples) |
| `cln6_RS-GPG_ca` | 1 Ca2+ | Fig. 2 (`model_4`) |
| `cln6_RS-GPG_2ca` | 2 Ca2+ | Fig. 1 (`model_4`), Fig. 2 (`model_4`), Fig. 3 (all 25 samples) |
| `ctrl_1mg` | 1 Mg2+ | `model_19`, ion-identity control (text of Case study 1) |
| `ctrl_2mg` | 2 Mg2+ | `model_0`, ion-identity control (text of Case study 1) |

### Case study 2: ATGL and fatty-acid chain length

These are the models shown in Fig. 4.

| folder to open | fatty acid | model shown in Fig. 4 |
|---|---|---|
| `fac08/` | C8:0 | `model_15` |
| `fac16/` | C16:0 | `model_19` |
| `fac24/` | C24:0 | `model_9` |

Open the `facNN` folder itself and don't rename it: FoldQC recognises a Boltz prediction folder by its name matching the file names.

## Files in each set

| file | content |
|---|---|
| `<set>_model_N.pdb` (CLN6) or `.cif` (ATGL) | structure |
| `plddt_<set>_model_N.npz` | pLDDT per token |
| `pae_<set>_model_N.npz` | predicted aligned error (token x token) |
| `pde_<set>_model_N.npz` | predicted distance error (token x token) |
| `confidence_<set>_model_N.json` | summary scores (confidence score, pTM, ipTM and others) |

## Opening a set in FoldQC

1. Install FoldQC (version 0.5.0) through the PyMOL plugin manager and open its dialog.
2. Under *Prediction source*, press **Folder** and choose one set, for example `case1-cln6/cln6_RS-GPG_2ca`.
3. Choose the **Model** and the **PyMOL target**.
4. Choose a metric in **Color by**, set the reference selection, palette and range, and press **Apply Coloring**.

All figures use the Viridis palette with **dark = confident**: tick **rev** for pLDDT (high is good) and leave it unticked for PAE and PDE (low is good). In reference selections, replace `<object>` by the name of the loaded PyMOL object.

## Reproducing the figures

| figure | set and model | Color by | reference selection | range (A or pLDDT 0-1) | rev |
|---|---|---|---|---|---|
| Fig. 1 | `cln6_RS-GPG_2ca`, `model_4` | PDE - mean to selection | `<object> and organic` | 0.3 - 1.5 | off |
| Fig. 2, 0 Ca | `cln6_RS-GPG`, `model_0` | pLDDT - continuous | - | 0.4 - 1.0 | on |
| Fig. 2, 1 Ca | `cln6_RS-GPG_ca`, `model_4` | pLDDT - continuous | - | 0.4 - 1.0 | on |
| Fig. 2, 2 Ca | `cln6_RS-GPG_2ca`, `model_4` | pLDDT - continuous | - | 0.4 - 1.0 | on |
| Fig. 4, top row | `fac08` `model_15`, `fac16` `model_19`, `fac24` `model_9` | PDE - mean to selection | `<object> and organic` | 0.3 - 1.5 | off |
| Fig. 4, bottom row | the same three models | pLDDT - continuous | - | 0.6 - 1.0 | on |

The viewing angles are given under [Camera views](#camera-views). The representation, labels and grey background residues of the printed Figs 2 and 4 are not restored by these steps.

**Fig. 3 (binding-site fingerprint).** Load `cln6_RS-GPG`, press **Load Ensemble**, select the ensemble as PyMOL target, set the reference selection to `organic` (cutoff 5 A) and choose **Plot > Binding-site fingerprint**; repeat for `cln6_RS-GPG_2ca`. In the printed figure both panels show the same residues (the site found for the 2 Ca2+ ensemble) on the same axes. This was done by script, so a plot made in the dialog can list a different residue set for the 0 Ca2+ ensemble. 

## Camera views

Paste these into the PyMOL command line after loading and colouring. The views were read from the figure sessions (Fig. 1 has no session; its view is the one produced by the figure script). A view only gives the printed orientation if the coordinates are in the same frame, so where models were superposed, apply the `cmd.transform_selection` commands **once** per loaded object first. Object names are the ones FoldQC creates (for CLN6, the hyphen in `RS-GPG` becomes an underscore). The `ray` size sets the framing; the printed panels were cropped afterwards.

### Fig. 1 (model view, right panel)

```pymol
bg_color white
hide everything
show cartoon, cln6_RS_GPG_2ca_model_4
show sticks, cln6_RS_GPG_2ca_model_4 and organic
show spheres, cln6_RS_GPG_2ca_model_4 and resn CA and not polymer
set_view (\
    0.2966, -0.1049, 0.9492,\
    0.2549, 0.9666, 0.0272,\
    -0.9204, 0.2339, 0.3134,\
    0.0000, 0.0000, -212.6017,\
    0.6459, 0.0167, 0.1644,\
    167.6168, 257.5866, -20.0000 )
ray 1200, 1000
```

### Fig. 2

The 0 Ca2+ and 1 Ca2+ models are superposed on the 2 Ca2+ model.

```pymol
cmd.transform_selection("cln6_RS_GPG_model_0", [\
    -0.12862572, -0.78404900, -0.60722532, 2.00838918,\
    0.35831397, 0.53420268, -0.76566219, 1.21357208,\
    0.92469807, -0.31606117, 0.21222351, -1.23991313,\
    0, 0, 0, 1 ], homogenous=1)
cmd.transform_selection("cln6_RS_GPG_ca_model_4", [\
    0.46157296, 0.87701855, -0.13337493, 0.13863063,\
    -0.79007875, 0.47478565, 0.38775527, -0.18202999,\
    0.40339306, -0.07360065, 0.91206194, -3.28761875,\
    0, 0, 0, 1 ], homogenous=1)
set_view (\
    0.8218, -0.0533, -0.5672,\
    -0.0179, 0.9927, -0.1192,\
    0.5694, 0.1081, 0.8149,\
    -0.0000, -0.0000, -58.6694,\
    -1.5584, -3.5527, 10.2370,\
    43.3863, 73.9523, -20.0000 )
```

Show one model at a time and render each with the same view:

```pymol
disable all
enable cln6_RS_GPG_model_0
ray 1400, 1000
png fig2_0ca.png
disable all
enable cln6_RS_GPG_ca_model_4
ray 1400, 1000
png fig2_1ca.png
disable all
enable cln6_RS_GPG_2ca_model_4
ray 1400, 1000
png fig2_2ca.png
```

### Fig. 4

The C8 and C24 models are superposed on the C16 model.

```pymol
cmd.transform_selection("fac08_model_15", [\
    0.19300727, -0.89452381, -0.40320633, -0.04879734,\
    -0.71304218, -0.41015957, 0.56862991, -1.35299894,\
    -0.67403193, 0.17775341, -0.71699699, 2.29500724,\
    0, 0, 0, 1 ], homogenous=1)
cmd.transform_selection("fac24_model_9", [\
    0.25169490, 0.24292167, 0.93682375, -0.09274436,\
    -0.38279663, -0.86406212, 0.32689967, -1.10003805,\
    0.88888493, -0.44089195, -0.12449043, 1.48383355,\
    0, 0, 0, 1 ], homogenous=1)
set_view (\
    0.9413, 0.3309, -0.0667,\
    -0.0760, 0.0153, -0.9970,\
    -0.3289, 0.9435, 0.0395,\
    0.0000, 0.0000, -90.2929,\
    -2.7795, 4.3263, -3.7757,\
    71.1876, 109.3981, -20.0000 )
```

Show one model at a time and render each with the same view, once coloured by PDE (top row) and once by pLDDT (bottom row):

```pymol
disable all
enable fac08_model_15
ray 1500, 1300
png fig4_c8.png
disable all
enable fac16_model_19
ray 1500, 1300
png fig4_c16.png
disable all
enable fac24_model_9
ray 1500, 1300
png fig4_c24.png
```

## Fig. 5: ATGL-ABHD5 (Case study 3)

No files are included: FoldQC fetches the complex from the [AlphaFold Database](https://alphafold.ebi.ac.uk/). The figure uses entry **AF-0000000211755036** (ATGL, UniProt Q96AD5, chain A; ABHD5, UniProt Q8WTS1, chain B), version v1. A download on 2026-11-27 was identical to the files used for the figure; a later database release could differ. An internet connection is needed. Use FoldQC 0.5.0.

### Step 1: fetch the complex

In the second source row of the FoldQC dialog, enter `Q96AD5` and press **AFDB-EBI**. In the list that opens, choose **Complex Q96AD5 + Q8WTS1 (AF-0000000211755036)**. The PyMOL object is named `AF_0000000211755036`.

### Step 2: colour by pLDDT (panel a)

| setting | value |
|---|---|
| Color by | pLDDT - continuous (structure B-factors) |
| Palette | Viridis, **rev** ticked |
| Min / Max | 0.3 / 1.0 |

Press **Apply Coloring**, then keep a copy of this colouring before the next step changes it:

```pymol
create fig5a, AF_0000000211755036
```

### Step 3: iLIS contact residues (panel b)

1. Press **Interfaces** (keep the ipSAE preset, PAE < 10 A).
2. On the **Chain pairs** tab, click the row for chains A and B. It should show iLIS 0.465 and 74 contact pairs.
3. Set **Details for** to `iLIS`, **Residue view** to `Contact-filtered support` and the side to `Both sides`.
4. Press **Color residues**. Contributor residues turn orange (31 in ATGL, 36 in ABHD5), the others grey.

### Step 4: representation and view

Run these in this order (`util.performance(0)` changes some earlier settings). The residues highlighted in blue (ATGL N209, I212, N215) and pink (ABHD5 R297, G326, D332; mouse R299, G328, D334) are those reported as crucial for the interaction (Kulminskaya et al., *J. Lipid Res.* 2024, PMID 38135254).

```pymol
bg_color white
set ray_opaque_background, 1
set antialias, 2
set depth_cue, 0
set valence, 0
util.performance(0)
set ray_transparency_oblique, 1.0
set antialias, 2
set ambient, 1.0
set direct, -0.6
set reflect, -0.6
set shininess, 0
set specular, 0
set specular_intensity, 0
set ray_shadow, off
hide everything, fig5a
show ribbon, fig5a
set ribbon_sampling, 10, fig5a
set ribbon_radius, 0.3, fig5a
set ribbon_transparency, 0.4, fig5a
hide everything, AF_0000000211755036
show ribbon, AF_0000000211755036
set ribbon_sampling, 10, AF_0000000211755036
set ribbon_radius, 0.3, AF_0000000211755036
set ribbon_transparency, 0.4, AF_0000000211755036
set_color okorange, [0.9, 0.62, 0.0]
set_color okblue, [0.0, 0.447, 0.698]
set_color okpink, [0.8, 0.475, 0.655]
create contributors, AF_0000000211755036 and polymer and byres b>0.5
hide everything, contributors
show ribbon, contributors
color okorange, contributors
set ribbon_radius, 0.5, contributors
set ribbon_transparency, 0.0, contributors
set ribbon_sampling, 10, contributors
select atgl_key, AF_0000000211755036 and chain A and resi 209+212+215
select abhd5_key, AF_0000000211755036 and chain B and resi 297+326+332
set stick_radius, 0.45, atgl_key or abhd5_key
color okblue, atgl_key
color okpink, abhd5_key
set valence, 0
```

Main view, panel (b) and then panel (a):

```pymol
set_view (\
    0.0436, 0.3394, 0.9396,\
    -0.9653, 0.2566, -0.0478,\
    -0.2574, -0.9050, 0.3388,\
    0.0000, 0.0000, -366.8329,\
    -0.4854, 1.8335, 4.3689,\
    289.2139, 444.4520, -20.0000 )
disable all
enable AF_0000000211755036
enable contributors
ray 1600, 1400
png fig5b.png
disable all
enable fig5a
ray 1600, 1400
png fig5a.png
```

Interface close-up (inset of panel b):

```pymol
disable all
enable AF_0000000211755036
enable contributors
show sticks, atgl_key or abhd5_key
hide sticks, (atgl_key or abhd5_key) and name C+O+N
show spheres, abhd5_key and resi 326 and name CA
set sphere_scale, 0.6, abhd5_key and resi 326 and name CA
hide ribbon, AF_0000000211755036 and not byres (AF_0000000211755036 within 11 of (atgl_key or abhd5_key))
hide ribbon, contributors and not byres (contributors within 11 of (atgl_key or abhd5_key))
deselect
set_view (\
    -0.1964, -0.7854, 0.5870,\
    0.1300, 0.5726, 0.8095,\
    -0.9719, 0.2353, -0.0104,\
    0.0000, 0.0000, -92.5622,\
    -0.0962, 2.6082, -1.8432,\
    72.9767, 112.1477, -20.0000 )
ray 1400, 1400
png fig5b_closeup.png
```
