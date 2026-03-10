# Blue Dot Modifications to audfprint

This documents all changes made to the upstream [audfprint](https://github.com/dpwe/audfprint) library (Dan Ellis, Columbia, 2014) for the Blue Dot fingerprinting system.

## Change Log

### Mid-Cancel Channel Extraction (2025)

**Files:** `audfprint_analyze.py`, `audio_read.py`, `audfprint.py`

Added `--mid-cancel` flag to extract the side channel (L−R) from stereo audio before fingerprinting. This removes center-panned speech from podcasts, dramatically improving music detection when speech is mixed over music.

### Decoupled threshcount: Gate vs Output Threshold (2025-06-26)

**Files:** `audfprint_match.py`, `audfprint.py`

#### Problem

In upstream audfprint, the `Matcher.threshcount` value (set via `--min-count`) was used for **two different purposes**:

1. **Stage 1 — Candidate gate** (`_best_count_ids`, line ~143): Filter track IDs by raw hash hit count > threshcount. Also in `_approx_match_counts` (line ~291): check if the **peak single-bin** count > threshcount before computing the windowed count.

2. **Stage 2 — Output filter** (`_approx_match_counts`, line ~295): The **windowed count** (sum of ±`window` bins around the peak) is what gets reported as the match's landmark count.

The critical issue: audfprint was designed for "is this 5-second clip my song?" queries where `--min-count 5` is typical. In that regime, coupling the gate and output thresholds is fine — if you need 5 matching hashes total, requiring 5 in a single ~23ms bin is reasonable.

But for **podcast scanning** with `--min-count 40`, matching hashes from music-under-speech are distributed across multiple adjacent time bins (±2 bins = ±46ms of time skew). A real match might have **59 total hashes in the ±2 window** but only **26 in the peak single bin**. The coupled threshold rejects it at the gate before the windowed count is ever computed.

#### Solution

Added `Matcher.threshcount_gate` attribute:
- **`threshcount`** (unchanged): Applied to the **windowed count** for final output filtering. "Does this match have enough evidence to report?"
- **`threshcount_gate`** (new): Applied to the **peak single-bin count** for candidate selection. "Is this candidate worth computing the full windowed count for?" Defaults to `None` (= use `threshcount`, preserving original behavior).

The `_gate_thresh` property returns the effective gate threshold.

#### Usage

```python
matcher = Matcher()
matcher.threshcount = 40       # Final output: need 40 windowed landmarks
matcher.threshcount_gate = 10  # Gate: allow candidates with 10+ in peak bin
```

CLI:
```bash
python audfprint.py match -d db.pklz --min-count 40 --min-count-gate 10 audio.wav
```

#### Impact

For the test episode "Loket voor het Leven #3" with `--min-count 40`:
- **Before**: 1 match (only Mencher and Polk, peak bin=67)
- **After**: 7 matches (all with windowed count ≥ 40, 0 false positives added)

Recovered matches: Bedhead (59 lm), Silky (56 lm), Arlan Vale (50 lm), Bedhead-intro (45 lm), Bright Edges (45 lm), Bedhead-outro (40 lm).

#### Why gate=10 is safe

Analysis of all 15 prefiltered segments from the test episode showed:
- **9/9 true matches** have peak single-bin count > 10
- **0 false positives** pass the gate at 10 (none have peak-bin > 10 in the density=50 midcancel DB)
- The false positive noise floor has peak-bin counts of 4-7

#### Quality DB needs gate=20

The quality_midcancel DB (density=100, fanout=8) produces ~2.7× more hashes per
track than the midcancel DB. This proportionally raises the false positive noise
floor — tracks that had ~15–18 windowed landmarks in midcancel can reach 40–57 in
quality. At gate=10, the quality DB produces 139 false positives; at gate=20, only 5.

| Gate | min_count=40 True+ | min_count=40 False+ | Unique tracks |
|------|---------------------|---------------------|---------------|
| 10   | 19                  | 36                  | 8/9           |
| 15   | 16                  | 9                   | 8/9           |
| 20   | 16                  | 4                   | 8/9           |
| 25   | 15                  | 2                   | 8/9           |

Gate=20 is the sweet spot: keeps 8/9 true tracks, cuts FP to just 4 Cobalt Blue
hits (all sub-second at 28:37, caught by downstream confidence/duration filtering).
See `scripts/test_quality_gate.py` for the full analysis.

#### Code Locations

| Location | Old behavior | New behavior |
|----------|-------------|--------------|
| `Matcher.__init__` | `self.threshcount = 5` | Added `self.threshcount_gate = None` + `_gate_thresh` property |
| `_best_count_ids` L~143 | `rawcounts > self.threshcount` | `rawcounts > self._gate_thresh` |
| `_approx_match_counts` L~291 | `filtered_bincounts[mode] <= self.threshcount` | `filtered_bincounts[mode] <= self._gate_thresh` |
| `_approx_match_counts` L~295+ | (no windowed count check) | Added `if count < self.threshcount: continue` |
| `_exact_match_counts` L~223 | `find_modes(..., threshold=self.threshcount)` | `find_modes(..., threshold=self._gate_thresh)` |
| `_exact_match_counts` L~228 | `filtcount >= self.threshcount` | unchanged (exact count = windowed) |
