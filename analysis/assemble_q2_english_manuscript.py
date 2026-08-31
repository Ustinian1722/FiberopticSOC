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
front_main = re.sub(r"^# English manuscript V1 — Front matter and Introduction\s*", "", front_main)
front_main = re.sub(r"^## Title\s*\n\n\*\*(.*?)\*\*", r"# \1", front_main, count=1, flags=re.S)

# Replace the drafting abstract with the final submission abstract.
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
references_main = re.sub(r"\n> Drafting note:.*$", "", references_main, flags=re.S).rstrip()

# Reviewer-risk editorial patches retained for the older Section 1/2/3 source fragments.
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
        "This study uses the publicly available SiC-18 dataset collected from a SiOx/C pouch-type lithium-ion cell instrumented with two embedded fiber Bragg grating (FBG) sensors.",
        "This study uses the publicly available SiC-18 dataset previously reported for an instrumented SiOx/C pouch-type lithium-ion cell [5]. The cell contains two embedded fiber Bragg grating (FBG) sensors.",
    ),
    (
        "For the dual-FBG system considered here, the two sensing channels have different thermal and mechanical sensitivities. In the released dataset, the two wavelength responses and the decoupled temperature T and deformation-force-related quantity F approximately satisfy",
        "For the dual-FBG system considered here, the two sensing channels have different thermal and mechanical sensitivities. The coefficients below are adopted directly from the published SiC-18 sensing relation [5] and retain the variable definitions and units used in the released data. The two wavelength responses and the decoupled temperature T and deformation-force-related quantity F approximately satisfy",
    ),
    (
        "A representation-aware dual-FBG temporal convolutional network, denoted RA-FBG-TCN, is developed for SOC estimation under dynamic loads and changing operating conditions.",
        "A representation-aware dual-FBG temporal convolutional network, denoted RA-FBG-TCN, is developed for SOC estimation under dynamic loads and changing operating conditions. The temporal backbone follows the causal/dilated convolutional sequence-modeling principle established for TCNs [9].",
    ),
    (
        "A point estimate alone does not communicate prediction reliability. To obtain an uncertainty interval without introducing an additional probabilistic neural network, residual split conformal prediction is applied after the point estimator has been trained and frozen.",
        "A point estimate alone does not communicate prediction reliability. To obtain an uncertainty interval without introducing an additional probabilistic neural network, residual split conformal prediction [10,11] is applied after the point estimator has been trained and frozen.",
    ),
    (
        "Figure 2(d) and Table 4 report the main results.",
        "Figure 2(d) reports the development-stage representation results.",
    ),
]
for old, new in sec23_patches:
    if old not in sec23_main:
        raise SystemExit(f"Expected Section 2/3 patch target not found: {old}")
    sec23_main = sec23_main.replace(old, new, 1)

back_matter = """# Data availability

The SiC-18 dataset analyzed in this study is publicly available through Mendeley Data (DOI: 10.17632/ft6rtwt8vm.1), as reported with the source study [5].

# Code availability

The analysis code, frozen experimental workflows, source-data tables, and reproducible figure-generation pipeline used in this study are maintained at https://github.com/Ustinian1722/FiberopticSOC. The submission branch preserves the numerical provenance of the results reported in the manuscript.
""".strip()

manuscript = "\n\n".join([
    front_main,
    sec23_main,
    sec46_main,
    back_matter,
    "# Figure captions",
    captions_main,
    references_main,
]) + "\n"

required = [
    # conventional benchmark
    "0.482%", "0.593%", "0.999614",
    # new strict backbone benchmark
    "0.864%", "1.163%", "0.565%",
    # new formal input ablation
    "2.162%", "1.795%", "1.770%", "0.464%",
    # OOD diagnostic retained as mechanism analysis
    "2.151%", "1.632%", "48.52%",
    # final five-seed T4
    "0.806%", "1.301%", "0.961–1.677%",
    # uncertainty
    "95.04%", "2.075%",
    # representation statistics
    "0.738", "0.655", "0.982", "0.985",
]
missing = [x for x in required if x not in manuscript]
if missing:
    raise SystemExit(f"Missing canonical manuscript claims: {missing}")

banned = [
    "first FBG-based SOC",
    "outperforms all baselines under all conditions",
    "universal cross-cell generalization",
    "untouched independent holdout",
    "condition number of 107",
    "condition number of 119",
    "native W1/W2 universally outperform",
]
found = [x for x in banned if x.lower() in manuscript.lower()]
if found:
    raise SystemExit(f"Banned/obsolete claims found: {found}")

for heading in [
    "# Representation-aware dual-FBG optical sensing for robust battery state-of-charge estimation under operating-condition shifts",
    "# 1. Introduction", "# 2. Dataset and electrical–optical signal analysis",
    "# 3. Methodology", "# 4. Experiments and results",
    "# 5. Discussion", "# 6. Conclusion", "# Data availability", "# Code availability",
    "# Figure captions", "# References",
]:
    if heading not in manuscript:
        raise SystemExit(f"Missing section heading: {heading}")

for phrase in [
    "one physical cell instrumented with a fixed dual-FBG",
    "no formal 95% coverage guarantee",
    "[8] and data-driven methods",
    "TCNs [9]",
    "split conformal prediction [10,11]",
    "SiC-18 dataset previously reported",
    "0.635% SOC [5]",
    "10.17632/ft6rtwt8vm.1",
    "ranks third rather than first",
    "statistically comparable",
]:
    if phrase not in manuscript:
        raise SystemExit(f"Reviewer-risk safeguard missing: {phrase}")

if "English manuscript V1" in manuscript or "Drafting note:" in manuscript:
    raise SystemExit("Internal drafting metadata leaked into assembled manuscript")

placeholder_lines = [line for line in manuscript.splitlines() if "placeholder" in line.lower()]
if any(("Fig. 1" not in line and "Fig. 3" not in line and "final artwork" not in line.lower()) for line in placeholder_lines):
    raise SystemExit(f"Unexpected placeholder text: {placeholder_lines}")

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
    "- clean front matter / no drafting metadata: **PASS**",
    "- data/code availability sections: **PASS**",
    "- section/reference structure check: **PASS**",
    "- intended artwork placeholders only: **PASS**",
    "",
    "The final journal bibliography should still be regenerated through the reference manager; the assembled V2 includes the verified core references needed for the current argument.",
]
(OUT / "Q2_ENGLISH_MANUSCRIPT_V2_ASSEMBLY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))