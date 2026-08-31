# Q2 final quantitative figure freeze

Status: **FROZEN FOR MANUSCRIPT**

Canonical final rendering:
- workflow: `Q2 publication figures`
- run ID: `33368316998`
- head SHA: `d376c6bf1eec1f821c7411c31d0770856acde87b`
- artifact: `q2-publication-figures-final`
- artifact ID: `9749130021`
- artifact digest: `sha256:ae7e748e0b4e9959ff7c1753632ee804a87812ed2d3d4d78b7f07ed91776939a`

## Final quantitative displays

- Fig. 1: quantitative panels b–d frozen; panel a remains experimental-setup placeholder.
- Fig. 2: frozen representation analysis. Panel c reports only directly reproducible absolute Pearson correlations, with no condition-number annotation.
- Fig. 3: methodology artwork placeholder, not part of this quantitative freeze.
- Fig. 4: frozen conventional SOC prediction/error display.
- Fig. 5: frozen electrical-OOD optical-complementarity display.
- Fig. 6: frozen five-seed strict T4 generalization display.
- Fig. 7: frozen wavelength-noise and 95% UQ display.

## Fig. 2 final correlation values

Direct descriptive statistics from the released trajectories:
- raw W1/W2, 1C: |r| = 0.738;
- raw W1/W2, 2C: |r| = 0.655;
- decoupled T/F, 1C: |r| = 0.982;
- decoupled T/F, 2C: |r| = 0.985.

These values replace earlier development-stage approximate condition-number language in the main manuscript. The paper does not require an exact covariance-condition-number claim.

## Final QA

The strict workflow gate passed completely:
- static/source preflight completed;
- all figures exported as editable SVG, PDF, 600-dpi TIFF and PNG preview;
- minimum PDF text size: Fig.1 7.0 pt, Fig.2 6.0 pt, Fig.4 7.0 pt, Fig.5 6.3 pt, Fig.6 6.0 pt, Fig.7 7.0 pt;
- below-5-pt text count = 0 for all figures;
- collision audit = 0 FAIL / 0 WARN for every quantitative figure;
- figure/source-data contract passed.

## Freeze rule

Do not change quantitative panels for stylistic experimentation. Reopen only for a factual error, explicit target-journal formatting requirement, or reviewer request.

Remaining main-figure artwork is limited to Fig. 1(a) and Fig. 3.