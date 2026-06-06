# Submission Readiness Checklist

## Completed in this package

- Manuscript draft v3 with citation keys and no bare `[Ref]` placeholders.
- Literature matrix with 41 candidate references and manuscript-use notes.
- BibTeX seed file for the candidate reference set.
- Active figure manifest with 21 active figures and 11 legacy figure stems.
- Figure files for 12 main figures and 9 supplementary figures; active formats complete: True.
- Supplementary Methods draft explaining preprocessing, symmetry reconstruction, boundary fitting, dynamics, stability, error budget, sensitivity and supplementary figures.
- Nomenclature table and equation inventory from the reproducible analysis script.
- Processed reproducibility package prepared under `analysis_outputs/reproducibility_package/` and zipped as `analysis_outputs/reproducibility_package.zip`.
- Reviewer-risk response notes covering finite process-matrix scope, molten-region-only export, overfitting, overparameterization, theory depth, identifiability and material sensitivity.

## Remaining manual tasks before submission

- Target journal set to Applied Mathematical Modelling; before submission, check abstract length, graphical rules, reference style and data-availability wording against the current author guide.
- Verify every seed reference against the publisher page or database export, especially older books and classic papers.
- Add author names, affiliations, ORCID identifiers, acknowledgments and funding statements.
- Decide whether Flow3D raw CSV files can be shared publicly; if not, submit the processed reproducibility package as Supplementary Data and optionally archive it on Zenodo, Mendeley Data or GitHub.
- Confirm whether Flow3D software settings, mesh resolution and export filters can be described in sufficient detail for reproducibility.
- Check all TIFF files against the target journal's DPI, color mode and physical width requirements.
- Remove or ignore legacy figure stems during final layout; use only files marked `active` in `active_figure_manifest.csv`.
- Add final reference manager output, journal-specific BibTeX or CSL formatting.
- Confirm all material constants with the simulation setup notes, especially absorptivity, beam radius, latent heat and temperature-dependent property tables.
- Decide whether the manuscript needs experimental validation language removed or softened, since the current evidence is CFD-informed rather than experiment-validated.

## Current recommended submission position

For Applied Mathematical Modelling, the paper should be presented as CFD-informed observed free-boundary identification and engineering mathematical modeling for L-DED. The central defensible claim is that the molten-region point-cloud sequences admit compact superellipsoid boundary descriptors and parsimonious stable condition-wise baseline dynamics, with external CFD holdout support, while remaining short of experimental validation or a universal process map.
