from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "paper" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

front = (DOCS / "Q2_ENGLISH_FRONT_INTRO_V1.md").read_text(encoding="utf-8")
abstract_final = (DOCS / "Q2_ABSTRACT_EN_FINAL.md").read_text(encoding="utf-8")
sec23 = (DOCS / "Q2_ENGLISH_SECTIONS2_3_V1.md").read_text(encoding="utf-8")
sec46 = (DOCS / "Q2_ENGLISH_SECTIONS4_6_V2.md").read_text(encoding="utf-8")
captions = (DOCS / "Q2_MAIN_FIGURE_CAPTIONS_EN.md").read_text(encoding="utf-8")
references = (DOCS / "Q2_REFERENCES_EN_CORE.md").read_text(encoding="utf-8")

# Internal recent-reference key remains a drafting resource and is not placed mid-manuscript.
front_main = front.split("## Recent-reference key used in this draft", 1)[0].rstrip()

# Replace the longer drafting abstract with the final 250-word submission abstract.
abstract_body = abstract_final.split("\n", 2)[-1].strip()
front_main = re.sub(
    r"## Abstract\n\n.*?\n\n## Keywords",
    f"## Abstract\n\n{abstract_body}\n\n## Keywords",
    front_main,
    count=1,
    flags=re.S,
)

sec23_main = re.sub(r"^# English manuscript V1 — Sections 2–3\s*", "", sec23).strip()
sec46_main = re.sub(r"^# English manuscript V2 — Sections 4–6\s*", "", sec46).strip()
captions_main = captions.split("\n", 1)[1].strip()
references_main = references.strip()

# Reviewer-risk editorial patches. These alter wording/citations only, never numerical evidence.
front_patches = [
    (
        "Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods and data-driven methods that learn nonlinear mappings from measured signals to SOC.",
        "Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods [8] and data-driven methods that learn nonlinear mappings from measured signals to SOC.",
    ),
    (
        "Fiber Bragg grating (FBG) sensors are especially attractive for such applications because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature.",
        "Fiber Bragg grating (FBG) sensors are especially attractive for such applications because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature [12,13].",
    ),
]
for old, new in front_patches:
    if old not in front_main:
        raise SystemExit(f"Expected front-matter patch target not found: {old}")
    front_main = front_main.replace(old, new, 1)

sec23_patches = [
    (
        "A representation-aware dual-FBG temporal convolutional network, denoted RA-FBG-TCN, is developed for SOC estimation under dynamic loads and changing operating conditions.",
        "A representation-aware dual-FBG temporal convolutional network, denoted RA-FBG-TCN, is developed for SOC estimation under dynamic loads and changing operating conditions. The temporal backbone follows the causal/dilated convolutional sequence-modeling principle established for TCNs [9].",
    ),
    (
        "A point estimate alone does not communicate prediction reliability. To obtain an uncertainty interval without introducing an additional probabilistic neural network, residual split conformal prediction is applied after the point estimator has been trained and frozen.",
        "A point estimate alone does not communicate prediction reliability. To obtain an uncertainty interval without introducing an additional probabilistic neural network, residual split conformal prediction [10,11] is applied after the point estimator has been trained and frozen.",
    ),
]
for old, new in sec23_patches:
    if old not in sec23_main:
        raise SystemExit(f"Expected Section 2/3 patch target not found: {old}")
    sec23_main = sec23_main.replace(old, new, 1)

sec46_patches = [
    (
        "Figure 7(c) illustrates the point estimate and calibrated interval over a representative test segment. These results show that a simple post-hoc conformal layer can supplement the deterministic point estimator with an uncertainty interval whose empirical coverage is aligned with the nominal level, without requiring a second probabilistic neural network.",
        "Figure 7(c) illustrates the point estimate and calibrated interval over a representative test segment. These empirical coverage results pertain to the blocked mixed-condition calibration/test regime; no formal 95% coverage guarantee is claimed here for arbitrary cross-rate or unseen-profile distribution shift. The result nevertheless shows that a simple post-hoc conformal layer can supplement the deterministic point estimator with an uncertainty interval whose empirical coverage is aligned with the nominal level, without requiring a second probabilistic neural network.",
    ),
    (
        "The present study focuses on operating-condition transfer within a fixed dual-FBG sensing configuration. The conclusions therefore apply most directly when sensor installation and calibration remain consistent while discharge rate and driving profile change.",
        "The primary quantitative dataset in this study contains one physical cell instrumented with a fixed dual-FBG sensing configuration. The conclusions therefore apply most directly to operating-condition transfer in which sensor installation and calibration remain consistent while discharge rate and driving profile change.",
    ),
]
for old, new in sec46_patches:
    if old not in sec46_main:
        raise SystemExit(f"Expected Section 4–6 patch target not found: {old}")
    sec46_main = sec46_main.replace(old, new, 1)

manuscript = "\n\n".join([
    front_main,
    sec23_main,
    sec46_main,
    "# Figure captions",
    captions_main,
    references_main,
]) + "\n"

# Canonical quantitative claims that must remain synchronized with frozen evidence.
required = [
    "0.482%", "0.593%", "0.999614",
    "2.151%", "1.632%", "48.52%",
    "1.795%", "0.806%", "1.301%",
    "0.961–1.677%", "95.04%", "2.075%",
    "0.738", "0.655", "0.982", "0.985",
]
missing = [x for x in required if x not in manuscript]
if missing:
    raise SystemExit(f"Missing canonical manuscript claims: {missing}")

# Claims explicitly excluded from the submission narrative.
banned = [
    "first FBG-based SOC",
    "outperforms all baselines under all conditions",
    "universal cross-cell generalization",
    "untouched independent holdout",
    "condition number of 107",
    "condition number of 119",
]
found = [x for x in banned if x.lower() in manuscript.lower()]
if found:
    raise SystemExit(f"Banned/obsolete claims found: {found}")

# Basic structure contract.
for heading in [
    "# 1. Introduction", "# 2. Dataset and electrical–optical signal analysis",
    "# 3. Methodology", "# 4. Experiments and results",
    "# 5. Discussion", "# 6. Conclusion", "# Figure captions", "# References",
]:
    if heading not in manuscript:
        raise SystemExit(f"Missing section heading: {heading}")

# Reviewer-risk safeguards.
for phrase in [
    "one physical cell instrumented with a fixed dual-FBG",
    "no formal 95% coverage guarantee",
    "[8] and data-driven methods",
    "TCNs [9]",
    "split conformal prediction [10,11]",
]:
    if phrase not in manuscript:
        raise SystemExit(f"Reviewer-risk safeguard missing: {phrase}")

# Only intended artwork placeholders are allowed.
placeholder_lines = [line for line in manuscript.splitlines() if "placeholder" in line.lower()]
if any(("Fig. 1" not in line and "Fig. 3" not in line and "final artwork" not in line.lower()) for line in placeholder_lines):
    raise SystemExit(f"Unexpected placeholder text: {placeholder_lines}")

# Abstract length contract: keep the submission abstract compact.
abstract_match = re.search(r"## Abstract\n\n(.*?)\n\n## Keywords", manuscript, re.S)
if abstract_match is None:
    raise SystemExit("Abstract section not found")
abstract_words = len(re.findall(r"\b[\w’'-]+\b", abstract_match.group(1)))
if abstract_words > 250:
    raise SystemExit(f"Abstract too long: {abstract_words} words")

out_path = OUT / "Q2_ENGLISH_MANUSCRIPT_V2.md"
out_path.write_text(manuscript, encoding="utf-8")

words = len(re.findall(r"\b[\w’'-]+\b", manuscript))
report = [
    "# English manuscript V2 assembly report",
    "",
    f"- output: `{out_path.relative_to(ROOT)}`",
    f"- approximate word count including tables/captions/references: **{words:,}**",
    f"- final abstract length: **{abstract_words} words**",
    f"- required canonical numerical claims: **{len(required)}/{len(required)} present**",
    "- banned/obsolete claim check: **PASS**",
    "- reviewer-risk safeguards: **PASS**",
    "- section/reference structure check: **PASS**",
    "- intended artwork placeholders only: **PASS**",
    "",
    "The final journal bibliography should still be regenerated through the reference manager; the assembled V2 includes the verified core references needed for the current argument.",
]
(OUT / "Q2_ENGLISH_MANUSCRIPT_V2_ASSEMBLY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
