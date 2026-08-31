from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "paper" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

front = (DOCS / "Q2_ENGLISH_FRONT_INTRO_V1.md").read_text(encoding="utf-8")
sec23 = (DOCS / "Q2_ENGLISH_SECTIONS2_3_V1.md").read_text(encoding="utf-8")
sec46 = (DOCS / "Q2_ENGLISH_SECTIONS4_6_V1.md").read_text(encoding="utf-8")
captions = (DOCS / "Q2_MAIN_FIGURE_CAPTIONS_EN.md").read_text(encoding="utf-8")

# Internal recent-reference key remains a drafting resource and is not placed mid-manuscript.
front_main = front.split("## Recent-reference key used in this draft", 1)[0].rstrip()
sec23_main = re.sub(r"^# English manuscript V1 — Sections 2–3\s*", "", sec23).strip()
sec46_main = re.sub(r"^# English manuscript V1 — Sections 4–6\s*", "", sec46).strip()

manuscript = "\n\n".join([
    front_main,
    sec23_main,
    sec46_main,
    "# Figure captions",
    captions.split("\n", 1)[1].strip(),
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
    "cross-cell generalization",
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
    "# 5. Discussion", "# 6. Conclusion", "# Figure captions",
]:
    if heading not in manuscript:
        raise SystemExit(f"Missing section heading: {heading}")

# Only intended artwork placeholders are allowed.
placeholder_lines = [line for line in manuscript.splitlines() if "placeholder" in line.lower()]
if any(("Fig. 1" not in line and "Fig. 3" not in line and "final artwork" not in line.lower()) for line in placeholder_lines):
    raise SystemExit(f"Unexpected placeholder text: {placeholder_lines}")

out_path = OUT / "Q2_ENGLISH_MANUSCRIPT_V1.md"
out_path.write_text(manuscript, encoding="utf-8")

words = len(re.findall(r"\b[\w’'-]+\b", manuscript))
report = [
    "# English manuscript assembly report",
    "",
    f"- output: `{out_path.relative_to(ROOT)}`",
    f"- approximate word count including tables/captions: **{words:,}**",
    f"- required canonical numerical claims: **{len(required)}/{len(required)} present**",
    "- banned/obsolete claim check: **PASS**",
    "- section structure check: **PASS**",
    "- intended artwork placeholders only: **PASS**",
    "",
    "The reference list is intentionally not auto-generated here. The seven verified recent references remain in the drafting files and will be merged with classic SOC/TCN/FBG/conformal references through the final reference manager.",
]
(OUT / "Q2_ENGLISH_MANUSCRIPT_ASSEMBLY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
