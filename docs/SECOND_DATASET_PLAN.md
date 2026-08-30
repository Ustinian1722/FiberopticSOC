# Q1 second-dataset validation plan

Date: 2026-08-31

## Purpose

SiC-18 is a single physical silicon-based pouch cell. The second dataset is not required to reproduce the exact dual-FBG sensing chain. Its role is to test whether the paper's core **electrical-base + bounded mechanical correction** principle remains useful with another cell population and another mechanical sensing mechanism.

## Preferred route: multi-cell sodium-ion pressure dataset

Preferred because it offers:

- 12 physical sodium-ion cells;
- mechanical pressure sensing rather than FBG-derived deformation force;
- CLTC-P and WLTC dynamic operating conditions;
- multiple life stages;
- an opportunity for both cross-cell and cross-condition evaluation.

The paper should not force FBG-specific T/F concepts onto this dataset. The shared abstraction is:

`electrical history -> electrical SOC base`

`mechanical history -> bounded SOC correction`

This allows a genuine cross-chemistry, cross-sensor validation of the fusion principle.

### Minimum adapter fields

Each sample/cycle representation should expose:

- `cell_id`
- `profile`
- `life_stage` if available
- `time` or sample order
- `voltage`
- `current`
- `mechanical` (pressure or a leakage-safe relative pressure representation)
- `temperature` if available
- `soc_target`

Forbidden metadata such as cycle ID/life stage/SOH must not enter the SOC estimator unless an explicit experiment is designed for them. They may be used only for split definitions/reporting.

### Recommended sodium validation tasks

1. same-cell cross-life SOC estimation;
2. same-life-stage cross-cell generalization;
3. cross-profile generalization where supported;
4. ablation: VI vs VI+mechanical vs bounded fusion;
5. per-cell and per-life-stage statistics.

The sodium experiment does not need to use raw optical/multi-view consistency because it has no dual-FBG wavelength pair.

## Public fallback: Warwick internal gas-pressure dataset

Dataset: *Dataset of accumulated internal gas pressure and temperature during lithium-ion battery operation and ageing*.

- Mendeley DOI: `10.17632/pn5ct66rn5.1`
- Data in Brief 59 (2025) 111420
- three instrumented LG-Chem INR21700-M50 cells
- embedded internal gas-pressure sensors
- 100 ageing cycles with RPTs every 20 cycles
- cell voltage, discharge capacity, temperatures, pressure-related measurements and characterization data
- 10 Hz data acquisition reported in the data article

This is a useful fallback because it moves beyond SiC-18's single cell and uses a different internal mechanical sensing mechanism. Before using it as an SOC benchmark, the exact availability of continuous current/voltage/pressure trajectories and leakage-safe SOC labels must be audited.

## Cross-dataset model rule

Do not claim that FBG deformation force, external fixture pressure and internal gas pressure are physically identical. They are different mechanical observables.

The transferable hypothesis is narrower:

> mechanical state contains SOC-relevant information complementary to electrical measurements, and a bounded correction architecture can exploit that information without requiring the mechanical modality to dominate the estimator.

## Go/no-go criteria

The second dataset strengthens the Q1 story only if at least one of the following is demonstrated without leakage:

- bounded fusion improves mean error over the electrical backbone;
- bounded fusion reduces worst-case / high-load / cross-condition error;
- bounded fusion reduces variance across cells or operating conditions;
- the gate/correction naturally becomes small where the mechanical modality is uninformative.

If none occurs, report the negative result internally and do not use the dataset merely to increase dataset count.
