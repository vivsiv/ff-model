# Plan: nflverse processor + gold/training-set builder

## Context

PFR scraping is dead (bot-blocked), so we switched to `nflreadpy`/nflverse as the
data source (`src/data/nflverse.py`, class `NflverseDataScraper`, bronze layer only
so far). This plan covers the next two layers:

1. A per-source silver processor for nflverse (bronze -> silver).
2. A cross-cutting gold/training-set builder (silver -> gold) that fixes a
   real modeling bug: the current gold table conflates players with long,
   established careers and players with 1 short (possibly outlier) season,
   because features are built from fixed 2/3-year rolling windows with no
   signal for how many seasons back that average is drawn from. Concrete
   example: Christian McCaffrey (long track record) vs. Brian Thomas Jr.
   (1 outlier rookie year) look equally "established" to the model today.

**Decision (this session):** nflverse is a full replacement for PFR as the
gold-table source, not a parallel source. `PfrProcessor`/multi-source support
is explicitly out of scope until something concrete makes it relevant again
(YAGNI) — don't build extensibility for it speculatively.

Sequence-model work (LSTM/Transformer over variable-length career sequences)
is deferred to a later session. This plan is tabular-only. Two things from
that discussion still apply here and are captured below: player-level
group leakage in train/test splitting, and keeping a raw `player` column in
gold output so that grouping is possible downstream.

## Mental model

- **Bronze**: raw per-source dumps, one file per year per stat type. Already
  exists for nflverse (`NflverseDataScraper`).
- **Silver**: per-source, cleaned/standardized, still one row per player-season
  (or team-season). One processor class per source.
- **Gold**: cross-year feature engineering + the fantasy-target join. This is
  where the career-length fix lives. Not tied to any one source.

## 1. `NflverseProcessor` (bronze -> silver) — IMPLEMENTED

Location: `src/processing/nflverse.py`.

nflverse's bronze data is much simpler to consolidate than PFR's ever was:
columns are already named, and one row already covers passing/rushing/receiving/
kicking for a player-season (no need to stitch together separate per-category
tables, no positional column-count repair hacks).

Actual shape (revised twice from the original sketch — see notes below):

```python
FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

class NflverseProcessor:
    def __init__(self, data_dir: str = "../data/nflv"):
        self.bronze_dir = os.path.join(data_dir, "bronze")
        self.silver_dir = os.path.join(data_dir, "silver")

    def _load_bronze(self, filename: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Load a single bronze file (e.g. "player_stats.csv") and filter to
        season.between(start_year, end_year)."""

    def build_player_stats(self, start_year: int, end_year: int, positions=FANTASY_POSITIONS) -> pd.DataFrame:
        """Load bronze player stats, filter to fantasy-relevant positions,
        write silver/player_stats.csv."""

    def build_team_stats(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load bronze team stats, write silver/team_stats.csv."""

    def process_all_data(self, start_year: int, end_year: int, positions=FANTASY_POSITIONS) -> None:
        """Runs both of the above."""
```

### Resolved during implementation (diverged from the original sketch above)

- **Bronze is no longer split per year.** `NflverseDataScraper` originally wrote
  one bronze file per season (`_save_by_season`) to mirror PFR's per-year
  convention. Since nflreadpy fetches an entire year range in a single call
  (unlike PFR's one-HTTP-request-per-year), and the processor was just going
  to re-concat those files anyway, this was pure overhead. The scraper now
  writes one `player_stats.csv` / `team_stats.csv` per fetch (`_save`), and
  `NflverseProcessor` takes explicit `start_year`/`end_year` and filters the
  single bronze file by `season`, independent of whatever range was fetched.
- **No name standardization needed.** The original plan carried over
  `standardize_name` from the PFR processor on the assumption it'd be needed
  for cross-table joins. It isn't: nflverse gives a stable `player_id` per
  player, and `build_player_stats`/`build_team_stats` don't join to each other
  in this processor at all (unlike PFR, which had to join separate
  passing/rushing/receiving tables by name). `src/processing/utils.py` was
  added then removed once this became clear. If a future join against a
  name-only external source (e.g. an ADP list) comes up, that's a gold-layer
  concern, not a silver-layer one.
- **Traded players**: confirmed `load_player_stats(summary_level="reg")`
  collapses a mid-season trade into a single row per player-season (max 1 row
  per `player_id` per season, verified directly) — no PFR-style `2TM`
  duplicate-row handling needed.

## 2. Gold/training-set builder (silver -> gold, career-aware) — IN PROGRESS

Location: `src/processing/gold.py`, class `TrainingSetBuilder`. Building this
incrementally (implement + test one piece at a time), same as the processor.
`gold_dir` is `data_dir/gold` (not `data_dir/gold/nflv`) — unlike bronze/silver,
gold is common across whatever sources feed it, since it just houses finished
training sets, not per-source raw/cleaned data.

This replaces the fixed 2/3-year rolling-window features
(`DataProcessor.create_rollup_stats`) with **expanding, prior-years-only**
career aggregates, so the model has an explicit signal for how much history
backs a given player's numbers.

**Explicitly deferred: the exact `stat_columns` list.** Which raw stats get
career features/shrinkage/trend is its own dedicated discussion, not decided
yet. All the pieces below take `stat_columns` as a parameter rather than
hardcoding a list.

### `_positional_baseline` — IMPLEMENTED (`src/processing/gold.py`, tested in
`test/processing/test_gold.py`)

Computes, per `(position, season)`, the trailing-5-season league-wide average
of each stat (seasons up to and including that season). This is the shrinkage
target for career averages.

**Why not an all-time positional average:** league-wide offensive output
drifts over the nflverse history (1999-present), so an all-time average would
be a stale reference for recent seasons. Verified concretely with
`fantasy_points_ppr` for WRs: trailing-5yr baseline by 2024 was ~77.4 vs.
~81.3 all-time — a real, non-trivial gap. `window_years` defaults to 5,
tunable/expandable later (e.g. different window sizes, decay weighting).

### `_add_career_features` — IMPLEMENTED (`src/processing/gold.py`, tested in
`test/processing/test_gold.py`)

```python
def _add_career_features(
    self,
    df: pd.DataFrame,
    positional_baseline_df: pd.DataFrame,
    stat_columns: List[str],  # TBD, see above
    player_grouping_col: str = "player_id",
    shrinkage_k: float = 3.0,
) -> pd.DataFrame:
    """For each player, sorted by season, compute expanding (inclusive of the
    current row's own season — the "prior years only" restriction is
    _shift_for_prediction's job, not this function's) aggregates:
      - {stat}_career_avg / _career_std / _career_max / _career_min.
        career_std is filled to 0 for a player's first season (undefined
        with 1 data point) rather than left NaN.
      - years_played: explicit career-length signal (this is the piece
        missing from the current model)
      - a shrinkage-adjusted average: blends career_avg toward
        _positional_baseline, weighted by years_played:
        shrunk_avg = (n / (n + k)) * career_avg + (k / (n + k)) *
        positional_baseline, with k=3 as a first guess (tunable later)
      - a trend feature: this season's own value minus career_avg
    """
```

Verified against real data (`Christian McCaffrey` vs. `Brian Thomas Jr.`,
the pair that motivated this whole redesign): by year 9, McCaffrey's
`shrunk_avg` (≈229) stays close to his real `career_avg` (≈279) since
`years_played` gives his own history a 0.75 weight — his down years (2020,
2021, 2024 injury seasons) don't erase a decade of track record. Brian
Thomas Jr.'s one outlier rookie season (284 points) gets `shrunk_avg` ≈ 129,
pulled hard toward the ~77-point WR positional baseline, since
`years_played=1` gives his own average only a 0.25 weight — no longer looks
as "established" as it would under a plain career average.

**Note on shrinkage symmetry (raised and confirmed this session):** the
formula is a convex combination, so it pulls in *either* direction — a
rookie with an unusually *bad* small-sample season gets pulled up toward the
baseline exactly as hard as an unusually great one gets pulled down. This is
intentional, not a bug: with only 1 year of evidence, an extreme observation
(good or bad) is more likely to be partly noise than a true reflection of
talent, so regression-to-the-mean should apply symmetrically (same logic as
the classic batting-average shrinkage example in sabermetrics). Known
limitation: this can't distinguish "genuinely weak rookie" from "talented
rookie in a bad situation" — that would need additional signal (draft
capital, target share, etc.) as its own feature later, not something this
estimator alone can capture.

### `_shift_for_prediction` — not yet implemented

Same role as today's `join_year` logic: target year N's label pairs with
career-to-date features computed only through year N-1.

### Team-stats join: capture team-change shift, not just team level

Motivation: the original PFR-based model found team offensive stats didn't
matter much to fantasy predictions, but that was under the leaky train/test
split (section 3) — worth re-testing team features now in case the leakage
was masking real signal, not just noise.

**The actual signal we want** (concrete example: Davante Adams, whose own
counting stats stayed down through his last Raiders/Jets seasons, but who
became a strong fantasy play once he landed with the Rams' much better
offense): not just "what's the level of this player's team," but **the shift
in team quality when a player changes teams**, decoupled from the player's
own stat line. This requires two different team lookups per training row, not
one:

- `origin_team` = player's `recent_team` in their year N-1 row (the team they
  actually produced last season's stats with)
- `destination_team` = player's `recent_team` in their year N row (their
  actual team for the season being predicted — for historical/training rows
  this is already known, since the season already happened)
- **Both looked up against team_stats from year N-1 only, never year N.**
  Using year N's team performance to predict year N's fantasy output would
  mean using information that doesn't exist yet at real prediction time (you
  don't know how a team will perform before the season happens) — same
  category of look-ahead bug as everything else this session. So: Adams'
  training row for predicting his Rams season pairs his own year N-1 (bad,
  Raiders/Jets) stats with `destination_team_offense` = **the Rams' own year
  N-1 performance (the season before Adams joined them)**, not the season he
  actually played for them.
- `team_offense_shift = destination_team_offense - origin_team_offense`
  (zero for anyone who didn't change teams) — the explicit "moved into a
  better/worse situation" signal, alongside `destination_team_offense` itself
  as the plain new-environment-quality level.

Verified this join is mechanically sound (see below), independent of the
Davante Adams logic above:

- **Relocated franchises (Raiders/Chargers/Rams) are not a join risk.**
  nflverse doesn't use historically-accurate codes for these three — e.g. the
  St. Louis Rams (2004-2012 in real life) are coded `"LA"` retroactively for
  the entire 2003-2024 span, well before their actual 2016 move. Same pattern
  for Oakland->Vegas (`"LV"` used from 2003) and San Diego->LA Chargers
  (`"LAC"` from 2003). Looks wrong if you're expecting historically accurate
  labels, but it doesn't matter for this join specifically: `player_stats` and
  `team_stats` were checked to use the *identical* code for the *identical*
  season range for every team, so `(team_code, season)` resolves cleanly with
  no dropped or mismatched rows. (Would only matter if we ever join against a
  third source using historically-accurate codes — not relevant here.)
- Minor unrelated quirk, same non-issue: Jacksonville is coded `"JAC"` for
  2001-2002 only and `"JAX"` everywhere else, consistently in both tables.
- **Traded/cut players (mid-season, not year-over-year): real but small.**
  Checked 2023: 14 of 577 (2.4%) of QB/RB/WR/TE players played for more than
  one team *within* that season. Season-level `recent_team` reflects only the
  *last* team played for, while accumulated stats span both teams. Decision:
  proceed with the straightforward join as-is; treat this as a documented,
  accepted limitation rather than something to engineer around now (fixing it
  would mean pulling weekly data and games-weighting team context — real
  effort for a small slice of rows). Revisit only if team features turn out
  to matter once the leakage fix lands.

### Live player pool (`data/2025_fantasy_players.csv`) needs regenerating, not fixing

This file is a leftover PFR-era artifact — team codes are PFR's convention
(`GNB`, `KAN`, `LVR`, `NWE`, `NOR`, `SFO`, `TAM`, `LAR`, `2TM`), not nflverse's
(`GB`, `KC`, `LV`, `NE`, `NO`, `SF`, `TB`, `LA`) — confirmed a 9-code mismatch,
not just the relocated-franchise ones. **Decision: throw it away**, generate
a fresh 2026 version instead of building a translation table for a file we
don't want to keep maintaining anyway. Prefer sourcing the live player pool
from nflverse itself (e.g. current rosters) for code consistency. This only
blocks `build_live_set`, not `build_training_set` — deferred until we get to
the live-set piece specifically.

### Required schema change vs. today

`DataProcessor.clean_stats` currently does
`final_df.drop(columns=['player', 'year', 'team'])` and folds identity
entirely into the `id` string (e.g. `"christian_mccaffrey_2020"`). The new
gold output must **keep both a raw `player` column and the target `season`/
`year` column** (not just fold them into `id`) — `player` for diagnostics
(see below), and `season`/`year` because the corrected split (section 3)
needs to filter/sort rows chronologically, not reverse-parse them out of the
`id` string every time.

## 3. Follow-on fix in `modelling.py` (revised this session — see below)

**Revised conclusion — a chronological split is the fix, not a player-grouped
one.** Earlier in this project we concluded `split_data`'s plain row-level
`train_test_split` (no grouping) was buggy specifically because it lets the
same player's rows land on both sides of the split. On reflection (prompted
by a good pushback, worth recording): a player-*grouped* split is the wrong
fix for this domain. The reasoning that generally justifies grouped splits
(e.g. medical imaging, where a patient's other scans usually aren't available
in production) doesn't hold here — a returning player's own prior-season
history is *always* going to be sitting in training data whenever the
production model predicts their next season, since the training set is just
"all of NFL history so far." A grouped split would evaluate a scenario
(zero information about the player at all) that's actually the *minority*
case in production, not the norm, and would throw out most usable data to do
it. Analogy that clarified this: a model predicting Apple's stock price would
of course train on Apple's own price history — you wouldn't hold Apple out
of training to "test fairly."

The real bug was narrower: the split was random, so a row can end up in
"train" whose features were built from seasons that postdate a "test" row's
target year — i.e. no guarantee the test set is strictly in the future
relative to the training set. The fix is a **chronological split**: hold out
the most recent N seasons as test, train on everything before. This still
lets returning players appear in both train and test (correct — that matches
deployment), while guaranteeing no row's features were built using
information from after the point it's predicting. Applies to any internal CV
folds inside `GridSearchCV`/`run_model_tuning` too, not just the outer split
(e.g. a time-series-aware CV scheme, not vanilla k-fold).

Soft follow-up, not a blocker: worth separately tracking validation error for
low-experience players (`years_played <= 1`) as a diagnostic once the model
is being evaluated, since that's specifically the population the career-length
redesign was about (the Brian Thomas Jr. case that started this whole
discussion) — a good overall score could still mask poor performance there.

## Explicitly deferred (came up in discussion, not in scope here)

- Sequence models (LSTM/Transformer) over variable-length career sequences.
- Prefix-expansion sequence construction (`seq[1..k] -> target_{k+1}` for
  k=1..N-1 per player) as the eventual training-example format for sequences.
  Noted risk to revisit then: prefix expansion overweights long careers in
  the loss (a 10-year player contributes 9 examples, a 2-year player
  contributes 1) — may need inverse-length weighting or capping.
- Multi-source gold assembly (PFR + nflverse side by side). Not built now;
  revisit if a concrete need for a second source shows up.
