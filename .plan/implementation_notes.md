## Phase 1 cleanup pass (per 2026_refresh_project_plan.md)

Four items raised, all done:

1. **Column exclusion bug fixed.** `build_training_set` was loading the full
   143-column `player_stats.csv` and only ever *adding* derived columns for
   `stat_columns` -- it never dropped the registry's excluded columns, so all
   90 of them (including `headshot_url`) rode along untouched into
   `training_set.csv`. Fixed by selecting `identity_columns + stat_columns`
   from `player_df` before running the pipeline. Verified against real data:
   `headshot_url`/`def_sacks`/`fg_made` gone, `player_display_name` (identity)
   retained, shape 518 -> 437 columns.

   This surfaced a registry schema gap: `excluded` conflated genuinely-useless
   columns (defense/kicking/punting/`headshot_url`) with identity/context
   columns gold.py actually needs (`player_id`, `position`, `season`,
   `recent_team`, `games`, etc.), with only a comment distinguishing them.
   Restructured `column_registry.yaml` to `identity`/`stats` (renamed from
   `included`)/`targets` (subset of `stats`). Renamed `get_included_columns`
   -> `get_stat_columns`, added `get_identity_columns`.

   **Revised again same session**: dropped `excluded` entirely and added an
   `nflverse:` source-namespace layer above `player_stats:` (raised as two
   follow-up questions -- see below for the reasoning on both). Registry is
   now `{source}.{table}.{identity,stats,targets}`, loader functions take
   `(source, table)`. `identity`/`stats` are no longer required to be
   exhaustive (a column judged not useful just isn't listed anywhere,
   instead of being explicitly recorded as `excluded`) -- traded away the
   "was this column deliberately rejected or just forgotten" audit trail and
   the completeness-check invariant, in exchange for not maintaining a
   90-entry list that no code actually reads (confirmed via grep: nothing
   outside the registry's own module/tests ever called
   `get_excluded_columns`). `test/processing/test_column_registry.py` updated
   to match (6 tests, source-parameterized). Verified `build_training_set()`
   output is unchanged after the restructure (same 11,152 x 437 shape,
   `headshot_url` still absent).

   On the source-namespace layer: checked first whether it was even
   necessary for disambiguation -- it isn't yet, nflverse's table names
   (`player_stats`, `team_stats`) don't collide with PFR's
   (`player_fantasy_stats`, `player_receiving_stats`, `player_rushing_stats`,
   `player_passing_stats`, `team_offense`), so a flat table-keyed registry
   would've worked fine without it. Added it anyway for consistency with the
   `data/bronze/{source}/`, `data/silver/{source}/` convention used
   everywhere else in the project, and because PFR doesn't actually feed
   `gold.py` right now anyway (per the earlier full-replacement decision) --
   so this doesn't unblock or change anything today, it's just naming the
   one source that exists correctly in case a second one ever needs to.

2. **`NflverseDataScraper` -> `NflverseScraper`**, matching the
   one-`*Scraper`-per-source convention stated in the plan. Mechanical rename
   across `src/data/nflverse.py` and `test/data/test_nflverse.py`.

3. **`src/processor.py` split, moved, and partially deleted.** Split into:
   - `src/processing/pro_football_reference.py`, class
     `ProFootballReferenceProcessor` (renamed from `DataProcessor` to match
     `NflverseProcessor`'s convention) -- carries over exactly the
     silver-layer methods (`standardize_name`, `standardize_team_name`,
     `parse_awards`, `merge_multi_player_rows`, `combine_year_data`,
     `add_ratio_stats`, `create_rollup_stats`, `write_to_silver`,
     `build_player_*_stats`, `build_team_stats`), with `bronze_dir`/
     `silver_dir` now namespaced under `pfr` (`data/bronze/pfr`,
     `data/silver/pfr`) to match the per-source convention. No `gold_dir`,
     no `current_year` -- gold is `TrainingSetBuilder`'s job now, not this
     processor's.
   - **Deleted, not moved**: `join_training_stats`, `join_live_stats`,
     `clean_stats`, `collapse_duplicate_columns`, and the old
     `build_training_set`/`build_live_set`/`process_all_data` (the parts of
     `process_all_data` that called them). These were PFR-schema-specific
     gold-assembly logic (joins keyed on separate pass/rush/rec tables,
     age columns, `2TM` handling, 2-3yr rolling windows) and are functionally
     superseded by `gold.py`'s `TrainingSetBuilder` (career features,
     shrinkage, proper backward-only join, no leakage). No real destination
     to move them to.
   - Also fixed to match: `src/data/pro_football_reference.py`'s
     `bronze_dir`/`html_dir` were still un-namespaced (`data/bronze`,
     `data/html`, no `pfr` subfolder) despite `NflverseScraper` already using
     `data/bronze/nflv` -- namespaced to `data/bronze/pfr`, `data/html/pfr`
     for consistency, so the new processor reads from the right place.
   - Test split accordingly: `test/processing/test_pro_football_reference.py`
     (11 tests moved, silver-layer coverage) created;
     `test/test_processor.py` deleted (its remaining 6 tests covered the
     deleted gold-assembly methods, so they're gone, not moved).
   - `README.md`'s Features section rewritten to describe the actual current
     `src/data/`/`src/processing/` layout instead of the old flat
     `scraper.py`/`processor.py` files.

**Not done, flagging for later:** `notebooks/fantasy_modelling.ipynb` still
imports `from src.data.pro_football_reference import ProFootballReferenceScraper`
and `from src.processor import DataProcessor` (the latter no longer exists at
all) -- fully on the old PFR pipeline, not touched in this pass. Low priority
(exploratory notebook, not production code) but will error if run as-is.

Full suite: 43 tests passing after this pass (was 49 before -- net change
from -6 deleted gold-assembly tests, +11 moved PFR-silver tests, and the
7 registry tests replacing the prior 6).

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

**`stat_columns` list — first pass done, pending your review.** Rather than a
hardcoded list or markdown doc, this is a real registry:
`src/processing/column_registry.yaml`, keyed by silver table name (so
`team_stats`, and later `ff_opportunity`, can be added as sibling keys),
each with exhaustive, mutually-exclusive `included`/`excluded` column lists.
`src/processing/column_registry.py` exposes `get_included_columns(table)` /
`get_excluded_columns(table)` for `gold.py` to read `stat_columns` from
directly, rather than gold.py hardcoding it. Added `pyyaml` as an explicit
dependency (was already present transitively via mlflow).

Rule of thumb applied for `player_stats`: keep passing/rushing/receiving raw
+ advanced/efficiency stats (EPA, CPOE, PACR/RACR, opportunity-share metrics
like `target_share`/`wopr`) and fantasy points; drop defense/kicking/punting
(irrelevant, position already filtered to QB/RB/WR/TE) and identity/context
columns (kept in the dataframe, just not career-averaged). Flagged a
"borderline" group in the YAML with comments rather than silently deciding
(return-specialist stats, possibly-redundant aggregate fumble counts,
penalties) for you to flip if you want them included.

Verified against real data: `included` (53) + `excluded` (90) = exactly the
143 real columns in `data/silver/nflv/player_stats.csv`, with zero missing
and zero extra — every real column is accounted for exactly once.

Noted but not integrated: you mentioned "expected fantasy points" as an
example useful metric — that exists in nflverse, but in a different table,
`nflreadpy.load_ff_opportunity()` (confirmed via a live check: has
`total_fantasy_points_exp`, `*_diff` vs. actual, etc.), not in
`player_stats`. Not pulled in as part of this pass.

**Added a third list: `targets`** (`fantasy_points`, `fantasy_points_ppr`,
read via `get_targets` — renamed from an earlier `labels`/`get_labels` pass)
— tracks which columns are candidate prediction targets for
`_join_with_target`'s `target_col`. Raised and resolved a real question along
the way: is using a player's own *prior*-season `fantasy_points_ppr` as a
feature to predict their *next* season leakage, since it's the same column as
the target? No — leakage would mean a row's features include its *own*
target-season value, which the pipeline already structurally prevents
(`_add_career_features` is inclusive only through each row's own season;
`_join_with_target`'s backward-only join guarantees a training row's features
never include its target season or later). Using a player's own trailing
performance to predict their future performance is standard autoregression,
not leakage, and excluding it would likely hurt the model, since it's
probably one of the strongest available signals. So `targets` is a
**subset of** `included`, not an alternative to it — enforced as an
invariant in `test/processing/test_column_registry.py`
(`test_player_stats__targets_are_a_subset_of_included`).

Tested in `test/processing/test_column_registry.py` (structural checks only
— no overlap between `included`/`excluded`, no duplicates, spot-checks for a
few known columns — deliberately not dependent on the real, gitignored data
file, so it doesn't require regenerating data to pass in a fresh checkout/CI).

### Registry wired into `TrainingSetBuilder.build_training_set` — IMPLEMENTED

Caught mid-session: the registry existed but nothing actually called it.
Added `build_training_set(target_col="fantasy_points_ppr")`, which:
validates `target_col` against `get_targets("player_stats")`, pulls
`stat_columns` from `get_included_columns("player_stats")` (not hardcoded),
loads `silver/nflv/player_stats.csv`, and chains
`_positional_baseline` -> `_add_career_features` -> `_join_with_target`,
writing the result to `gold_dir/training_set.csv`. Does **not** yet include
the team-shift join (career features only for now — team-shift is still
pending, see below).

Also fixed a real bug hit while wiring this up: `get_targets` still read the
YAML key `"labels"` after the section was renamed to `targets:`, which
would have raised `KeyError` on first real use. Fixed, plus a leftover
`get_labels` reference in the tests.

Also fixed a real performance issue surfaced by testing against the *full*
53-column registry for the first time (prior tests only ever used 1-2 stat
columns, which never triggered it): assigning ~5 new columns per stat in a
python loop (`df[new_col] = ...`, called ~250+ times) hits pandas'
`PerformanceWarning: DataFrame is highly fragmented`. Fixed in both
`_positional_baseline` and `_add_career_features` by building each batch of
new columns as a dict and adding them with a single `pd.concat(axis=1)`
instead of one-at-a-time assignment. Verified output is byte-identical to
before the refactor (same shape, same McCaffrey values) with zero warnings.

Verified end-to-end against real data:
`build_training_set()` -> **11,152 rows x 518 columns**.

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

### `_join_with_target` — IMPLEMENTED (`src/processing/gold.py`, tested in
`test/processing/test_gold.py`; renamed from the originally-planned
`_shift_for_prediction` — "target" is the correct/standard term for a
regression label, so the new name says what it does)

Same role as today's `join_year` logic: target year N's value pairs with
career-to-date features from the player's most recent prior season — **not
necessarily season N-1**. First implementation required an exact N-1 match
(inner join on player + season-shifted-by-1), which turned out to be wrong:
a player who missed an entire season (injury, out of the league) would be
silently dropped from training for their return season, and — more
importantly — would get **no live-set prediction at all** for a real
"predict this year for a player who missed all of last year" case. Fixed
using `pd.merge_asof(..., direction="backward", allow_exact_matches=False)`,
which matches each target season to the nearest prior season with data,
however far back that is.

```python
def _join_with_target(
    self,
    features_df: pd.DataFrame,
    target_col: str,
    player_grouping_col: str = "player_id",
) -> pd.DataFrame:
    """Joins each player's season-N target value onto their most recent
    prior season's feature row (nearest-backward asof join per player), so
    training features never include information from the season being
    predicted, and a player with a gap season still gets matched to their
    last active season instead of being dropped."""
```

Adds `seasons_since_played` (`target_season - season - 1`, i.e. the number of
full seasons missed — 0 for a normal adjacent-year match, 1+ for a gap) so
the model can tell a fresh prior-season match apart from a multi-year-stale
comeback match, since both are now possible matches where before only
adjacent-year matches existed.

Still gets two exclusions for free without special-casing:
- A player's first season never appears as a *target* (no prior season
  exists at all to match backward to) — this is what naturally drops rookie
  seasons from training, same behavior as the old pipeline's explicit "drop
  first year" step, just as a side effect of the join instead of a separate
  filter.
- A player's most recent season never appears as a *feature* row in the
  output (no target season exists yet, since it hasn't happened) — correct,
  there's nothing to train on for it (that's `build_live_set`'s job instead).

Verified end-to-end on real data through `_positional_baseline` ->
`_add_career_features` -> `_join_with_target`: McCaffrey's full chain
produces 8 correctly-shifted rows (e.g. `season=2020` row's features/
`shrunk_avg` pair with `target_season=2021`, `target=127.50` — his actual
2021 output), 10,557 total training rows in the current partial pipeline
(single `fantasy_points_ppr` stat column, no team-shift join yet). Also
verified the gap-bridging fix against a real gap-season player (John Allred,
missed 2001 entirely): his 2000 season now correctly pairs with
`target_season=2002` (`seasons_since_played=1`, one season missed) instead
of being dropped.

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