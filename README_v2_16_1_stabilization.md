# Basketball Pricing Engine v2.16.1 — stabilization patch

This is a deliberately small stabilization patch on top of v2.16.0. It does **not** rewrite the pace/FGA chain, bucket weights, exact/near-state architecture, Monte Carlo pricing, or the 200-minute engine.

## What changed

### 1) Player 2PA matchup bug fixed
v2.16 simulated player 2PA with the opponent **PTS** modifier. That mixed an outcome with an opportunity channel.

v2.16.1 now uses a real **2PA opponent/position modifier**:
- opponent overall 2PA allowance,
- same-position 2PA/36 relative to league,
- the existing shrink/confidence logic.

2P% remains a separate, heavily-shrunk efficiency modifier.

### 2) Small player H2H layer restored
Player H2H had become audit-only / zero numerical weight. The intended design was a *small* current-season H2H correction.

v2.16.1 applies H2H only to opportunity rates (2PA / 3PA / REB / AST), never shooting percentages. The blend is shrunk by:
- number of H2Hs,
- current rotation similarity,
- similarity between H2H minutes and today's projected minutes.

The blend weight is capped at 5%, and the final per-stat H2H modifier is capped at +/-3%.

### 3) Multi-OUT concentration safety for BLK / DREB / OREB
The fixed roster-state caps could make the team model mathematically unable to react enough when multiple high-contribution bigs were OUT. Example: even if two rim protectors owned a very large share of healthy BLK, the old BLK roster modifier could not fall below 0.88.

v2.16.1 measures the **healthy share of the stat lost by the selected OUT players**. It only relaxes the lower cap when the synthetic healthy -> OUT counterfactual already points downward.

This is deliberately strongest for BLK, lighter for DREB/OREB. No positive boost is created by this rule.

The Team roster-state audit now exposes:
- OUT healthy stat share,
- base lower cap,
- adaptive lower cap,
- applied roster-state modifier.

### 4) Exact bookmaker-line comparison / arb audit
Team Markets now accepts an optional bookmaker CSV *after the model is frozen*.

Two-way rows:
`Scope,Market,Line,Over Odds,Under Odds`

Scopes are `TOTAL` or the current team abbreviation.

Three-way team-with-most rows:
`Scope=MOST,Market,Away Odds,Tie Odds,Home Odds`

The app prices the **exact bookmaker line from the current raw Monte Carlo arrays**, showing:
- model probability,
- model fair odds,
- model EV at the offered price,
- price required for the selected target EV,
- bookmaker implied-probability sum,
- whether the bookmaker prices themselves form an actual arbitrage.

Important: a model fair price is not a tradable quote. A model/book disagreement is **not an executable arbitrage** unless two real books offer the opposing prices.

## What was intentionally NOT changed

- No global player-to-team forced normalization was added yet.
- No automatic haircut to player per-minute rates when manual minutes rise.
- No new PF model.
- No change to pace or the possession-consistent FGA chain.
- No change to Old / G6-10 / L5 weights.
- No change to the v2.15.1 stale-state hotfix.

The Zandalasini test showed why this is important: at an unrealistic 25-minute assumption the projection looked inflated; with a normal 22-minute assumption the model landed close to the market. Minutes remain a first-order trader input, so this patch avoids 'fixing' valid per-minute rates because of a bad minute assumption.

## Validation

- Existing v2.14 smoke test: PASS
- Existing v2.15 smoke test: PASS
- New v2.16.1 smoke test:
  - player 2PA responds to `opp_2pa`, not `opp_pts`: PASS
  - H2H layer remains <= 5% blend and +/-3% modifier: PASS

## Recommended validation sequence

1. Set realistic minutes first.
2. Freeze Confirmed OUT and manual minute state.
3. Run Team Markets once.
4. Upload the bookmaker CSV to the exact comparison panel.
5. Inspect large disagreements before treating them as value, especially BLK/DREB/OREB and team-with-most.
6. Run selected player props only after minutes are frozen.
