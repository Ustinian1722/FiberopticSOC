# Q2 external E3 WLTP audit decision

Status: **FROZEN — SUPERVISED E3 SOC MODELING NOT PURSUED**

Source: Hebenbrock et al. external dataset, Zenodo `10.5281/zenodo.15388590`, `validation_WLTP_cycling.csv`.

Canonical structure audit: run `33361861005`, artifact `q2-external-wltp-audit`, head SHA `6293255b3e32b24fc2021aa01e6538527580ac83`.

Earlier runs `33361356093`, `33361523536`, and `33361692208` were audit/interface-development runs only. No WLTP model result was generated. The first failed before downloading target data due a freeze-key mismatch; later runs progressively exposed the raw phase structure.

## Facts established without SOC modeling

The source publication states that WLTP class 3 was converted to a current profile, scaled up to about 3C, repeatedly applied to P1/P2, with a 2.5 h break after each cycle until the lower cut-off voltage was reached.

The raw-data audit confirms:

- positive current is discharge, as documented by the dataset authors;
- P1/P2 S5 is available through the WLTP file;
- merged sampling has median contiguous `dt` about 0.985 s;
- current reaches approximately +/-30 A, corresponding to about 2.75-2.78C using the frozen initial capacities;
- the smallest idle-current threshold that consistently recovers the published ~2.5 h breaks is `|I| <= 0.1 A`;
- P1 contains 30 such long rests and P2 contains 25;
- the recovered long rests are approximately 9014-9017 s, directly matching the stated 2.5 h protocol;
- ordinary dynamic active intervals between rests are approximately 1780 s and typically have positive net discharge around 1.54-1.56 Ah.

## Critical raw-data structure

The file is not a single clean full-charge-to-cutoff discharge trajectory with only dynamic cycles and rests.

After several normal WLTP blocks, the raw series contains long active phases (~15-19 ks) with:

- both positive and negative dynamic current;
- large negative net Ah;
- voltage reaching the lower region and later returning toward 4.2 V.

These phases are consistent with a terminal dynamic portion followed by recharge, and in some cases the next post-charge dynamic portion occurs before the next 2.5 h rest boundary. Thus a single 'between-rests' interval can contain multiple physical stages.

This explains why naive first-active-to-last-active signed integration produced impossible SOC values: it mixed WLTP discharge and subsequent recharge processes.

## Why supervised E3 is stopped

No direct SOC label is supplied in the WLTP CSV. Producing a high-confidence SOC target for every dynamic sample would require an additional heuristic stage parser inside the mixed active phases to determine exact:

- terminal dynamic-cycle end;
- recharge start/end;
- full-charge reset;
- next WLTP start.

Although such a parser could be engineered from current/voltage patterns, it would add substantial label-construction freedom after E1/E2 results and would be difficult to defend as an independent external confirmation.

Therefore **no supervised constant-current -> WLTP SOC experiment is performed in this paper**. This is a research-integrity decision, not a model failure.

The WLTP data can still be used descriptively to establish that the external FBG system experiences repeated dynamic currents approaching ~3C and that the raw external dataset contains nontrivial recharge/rest structure.

## External-validation conclusion after E1/E2/E3 audit

- E1: uncalibrated S5rel is not cell invariant across physical cells.
- E2: with sensor identity fixed, S5rel improves mean same-cell 0.2/0.5C -> 1C transfer error in 3/4 cells but fails the strict Q95 robust-benefit gate.
- E3: dynamic WLTP exists and is physically relevant, but the public raw file does not expose a sufficiently unambiguous supervised SOC target without additional phase-label engineering; supervised E3 is therefore intentionally omitted.

No E3 outcome is eligible to reopen SiC-18 architecture, representation, UQ, or formal T4 decisions.