from pathlib import Path

src = Path('docs/Q2_MANUSCRIPT_EN_V1.md')
out = Path('docs/Q2_MANUSCRIPT_EN_V2.md')
text = src.read_text(encoding='utf-8')

replacements = [
    (
        'Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods and data-driven methods that learn nonlinear mappings from measured signals to SOC.',
        'Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods [8] and data-driven methods that learn nonlinear mappings from measured signals to SOC.'
    ),
    (
        'Fiber Bragg grating (FBG) sensors are especially attractive because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature.',
        'Fiber Bragg grating (FBG) sensors are especially attractive because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature [12,13].'
    ),
    (
        'The framework first determines which optical coordinate system is more suitable for cross-condition prediction and then combines voltage, current, and the retained native FBG wavelengths in a compact causal temporal model.',
        'The framework first determines which optical coordinate system is more suitable for cross-condition prediction and then combines voltage, current, and the retained native FBG wavelengths in a compact causal temporal model based on dilated temporal convolution [9].'
    ),
    (
        'The frozen estimator produces calibration predictions \\(\\hat{y}_i\\) and nonconformity scores',
        'Following split-conformal regression [10,11], the frozen estimator produces calibration predictions \\(\\hat{y}_i\\) and nonconformity scores'
    ),
    (
        'Figure 7(c) shows a representative calibrated interval. These results demonstrate that a simple post-hoc conformal layer can add uncertainty information without requiring a second probabilistic neural network.',
        'Figure 7(c) shows a representative calibrated interval. These empirical coverage results pertain to the blocked mixed-condition calibration/test regime; no formal 95% coverage guarantee is claimed here for arbitrary cross-rate or unseen-profile distribution shift. These results demonstrate that a simple post-hoc conformal layer can add uncertainty information without requiring a second probabilistic neural network.'
    ),
    (
        'The present study focuses on operating-condition transfer within a fixed dual-FBG sensing configuration. Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, and sensor-specific sensitivity may vary and can alter optical signal level and dynamics.',
        'The primary quantitative dataset in this study contains one physical cell instrumented with a fixed dual-FBG sensing configuration; the present analysis therefore focuses on operating-condition transfer within that sensing system. Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, and sensor-specific sensitivity may vary and can alter optical signal level and dynamics.'
    ),
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError(f'Expected manuscript text not found:\n{old}')
    text = text.replace(old, new, 1)

marker = '> Reference-manager note: add classic SOC observer/filtering references, the original TCN reference, foundational conformal-prediction references, FBG sensing-mechanism references, and the original SiC-18 data/paper citation before journal submission.'
if marker not in text:
    raise RuntimeError('Reference-manager note marker not found')

additional_refs = '''[8] G. L. Plett, “Extended Kalman filtering for battery management systems of LiPB-based HEV battery packs: Part 3. State and parameter estimation,” *Journal of Power Sources* 134(2) (2004) 277–292. https://doi.org/10.1016/j.jpowsour.2004.02.033.\n\n[9] S. Bai, J. Z. Kolter, V. Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” arXiv:1803.01271 (2018). https://doi.org/10.48550/arXiv.1803.01271.\n\n[10] J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman, “Distribution-Free Predictive Inference for Regression,” *Journal of the American Statistical Association* 113(523) (2018) 1094–1111. https://doi.org/10.1080/01621459.2017.1307116.\n\n[11] G. Shafer, V. Vovk, “A Tutorial on Conformal Prediction,” *Journal of Machine Learning Research* 9 (2008) 371–421.\n\n[12] C.-J. Bae, A. Manandhar, P. Kiesel, A. Raghavan, “Monitoring the Strain Evolution of Lithium-Ion Battery Electrodes using an Optical Fiber Bragg Grating Sensor,” *Energy Technology* 4(7) (2016) 851–855. https://doi.org/10.1002/ente.201500514.\n\n[13] A. Fortier, M. Tsao, N. D. Williard, Y. Xing, M. G. Pecht, “Preliminary Study on Integration of Fiber Optic Bragg Grating Sensors in Li-Ion Batteries and In Situ Strain and Temperature Monitoring of Battery Cells,” *Energies* 10(7) (2017) 838. https://doi.org/10.3390/en10070838.\n\n'''
text = text.replace(marker, additional_refs + marker, 1)

out.write_text(text, encoding='utf-8')
print(f'Wrote {out} ({len(text)} characters)')
