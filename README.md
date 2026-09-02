> **Latest package version: v2.18.2** — consolidated v2.18.0 base + v2.18.1 OT/DREB fixes + continuous residual player H2H. See `README_v2_18_2_continuous_player_h2h_consolidated.md`.

# Basketball Pricing Engine v2.15.0 — patch over v2.14

Η v2.15 κρατάει την αρχιτεκτονική της v2.14 (200-minute rotation, learned minute replacement, opponent/position model, shared pace, non-overlapping Old/G6–10/L5) και αλλάζει κυρίως το **availability / role layer** και την ταχύτητα του Player Deep Analysis.

## 1. Exact + near availability state χωρίς double counting

Το παλιό `exact joint OUT ή synthetic fallback` αντικαθίσταται στο ενεργό app από **single-score-per-game** availability similarity.

Για κάθε historical game και κάθε stat υπάρχει **ένα και μόνο ένα** similarity score προς τη σημερινή κατάσταση. Δεν υπάρχουν nested samples τύπου:

- 4/5 match
- μετά το ίδιο game ξανά ως 3/5
- μετά ξανά ως 2/5

Άρα ένα game δεν μπορεί να μετρήσει πολλές φορές. Το similarity είναι **INNER weight** μέσα στο ήδη υπάρχον Old / G6–10 / L5 bucket. Τα outer weights παραμένουν 55/20/25 (ή 35/20/45 μόνο όταν ο trader δηλώνει ξεχωριστό structural role change).

## 2. Stat-specific relevance των απουσιών

Η ίδια απουσία δεν έχει ίδιο βάρος σε κάθε stat.

- AST/TOV: μεγαλύτερη relevance σε handlers / guards / wings.
- REB/OREB/DREB: πολύ μεγαλύτερη relevance στις frontcourt απουσίες.
- 3PA: κυρίως perimeter role.
- FGA/PTS/FTA: πιο portable, αλλά πάλι role/position-aware.

Η relevance ξεκινά από το πραγματικό event volume της OUT παίκτριας και shrinked role/position compatibility. Η near-state ιστορία κατόπιν επηρεάζει **το πραγματικό per-minute rate της focal player**, όχι με νέο extra sample.

Παράδειγμα: Allemand/Mabrey OUT δεν δίνουν πλέον generic μεγάλο scoring/rebound boost στην Harrison μόνο και μόνο επειδή είναι OUT. Για Harrison REB το Morrow/Sabally state είναι πολύ πιο σχετικό. Για Rice AST συμβαίνει το αντίστροφο.

## 3. Small sample: 5 games maturity, όχι hard cutoff

Στο learned teammate replacement matrix το `<5 common games => empirical = 0` αφαιρέθηκε ως cliff.

- 1 game: structural prior μόνο (δεν μπορεί να ταυτοποιήσει covariance).
- 2–4 games: empirical signal υπάρχει αλλά shrinked πολύ έντονα.
- 5 games: maturity point.
- >5: confidence αυξάνει ομαλά με `N/(N+12)`.

Το 5 παραμένει safety/maturity threshold, όχι ON/OFF switch.

## 4. Near-state confidence

Για availability-state matching χρησιμοποιείται effective evidence mass:

`evidence = Σ similarity²`

και conservative shrinkage:

`confidence = evidence/(evidence + K) × min(1, evidence/5)`

Έτσι ένα 4/5 game δίνει χρήσιμη αλλά μικρότερη πληροφορία από exact game. 1–4 συγκρίσιμα games δεν μηδενίζονται, αλλά δεν μπορούν να πάρουν μεγάλο βάρος.

Τα numerical hyperparameters παραμένουν conservative priors και πρέπει να backtestαριστούν walk-forward πριν θεωρηθούν calibrated probabilities.

## 5. Player role fallback: released events ακολουθούν replacements

Το synthetic player fallback δεν μοιράζει πλέον τα vacated FGA/AST/REB γενικά ανάλογα με το υπάρχον contribution όλης της ομάδας.

Κάθε OUT παίκτρια απελευθερώνει τα δικά της events και αυτά δρομολογούνται κυρίως μέσω της **ίδιας learned teammate replacement matrix** που χρησιμοποιείται για τα λεπτά. Η event propensity της replacement player λειτουργεί μόνο ως δευτερεύον tie-breaker.

Άρα:

- guard creator OUT -> creation κυρίως σε πραγματικούς guard/wing replacements,
- frontcourt OUT -> rebound/frontcourt volume κυρίως σε αντίστοιχες replacements,
- δεν υπάρχει generic Harrison +25% creation επειδή λείπουν πολλοί guards.

Το residual fallback shrinkάρεται **ξεχωριστά ανά FGA / 3PA / FTA / AST / REB**, ανάλογα με το historical near-state confidence του συγκεκριμένου stat.

## 6. Team Markets έχουν το ίδιο near-state protocol

Τα Team Markets χρησιμοποιούν όλο το roster από πίσω, ανεξάρτητα από ποιες παίκτριες έχεις επιλέξει στο Player Props UI.

Κάθε team feature έχει δικό του availability map:

- FGA
- 3PA / 3P share
- FTA
- TOV
- OREB / DREB
- AST
- STL / BLK / PF

Έτσι historical game χωρίς guards μπορεί να είναι πολύ relevant για team AST/3PA αλλά λιγότερο relevant για team REB. Το current-opponent H2H παραμένει disjoint από το baseline όπως στη v2.14.

Το synthetic 200-minute team bridge παραμένει, αλλά τώρα fades **ανά stat** βάσει exact/near-state confidence.

## 7. Minutes δεν διπλομετριούνται

Η σειρά παραμένει:

1. healthy 200-minute rotation,
2. confirmed OUT minutes released μέσω learned replacement matrix,
3. trader/metadata minutes ως explicit final targets,
4. role/event adjustment μόνο σε per-minute opportunities.

Άρα manual `Projected Min = 32` δεν παίρνει και δεύτερο injury minute boost.

## 8. 50,000 simulations + faster Deep Analysis

- Player default: **50,000 sims**.
- Coupled Team/Game default: **50,000 sims**.
- Η μία player simulation παράγει ήδη PTS / REB / AST / 3PM / 3PA / FTA / PR / PA / AR / PRA.
- Το automatic market table αποθηκεύεται μαζί με το player run.
- Σε Streamlit versions που υποστηρίζουν `st.fragment`, αλλαγή Deep-dive player ή Market Ladder ξανατρέχει μόνο το μικρό pricing panel και **όχι** minutes / state / matchup / Monte Carlo.

## 9. Files στο patch

Αντικατάστησε πάνω στην υπάρχουσα v2.14:

- `streamlit_app.py`
- `core/availability.py`
- `core/availability_impact.py`
- `core/redistribution.py`
- `core/minutes_engine.py`
- `core/player_model.py`
- `core/team_model.py`

Νέο test:

- `tests/v215_smoke.py`

## 10. Validation

Το patch πέρασε:

- `tests/v214_smoke.py`
- `tests/v215_smoke.py`

Το v2.15 smoke ελέγχει ειδικά ότι:

- κάθε game έχει μόνο ένα similarity score ανά stat,
- ένα guard absence έχει μεγαλύτερη AST relevance και ένα forward absence μεγαλύτερη REB relevance στο synthetic test,
- τέσσερα common games έχουν μικρό αλλά μη μηδενικό empirical replacement confidence,
- οι default simulations είναι 50k.

### Σημαντικό

Η v2.15 βελτιώνει το structural model και μειώνει αυθαίρετα injury boosts, αλλά **δεν είναι ακόμα fully fitted/backtested prop model**. Τα fair odds πρέπει να επανελεγχθούν σε walk-forward historical validation πριν χρησιμοποιηθούν ως calibrated true probabilities.

---

## v2.17 structural calibration update

See `README_v2_17_structural_calibration.md`. Team 3P_SHARE, FTA, TOV and AST-per-made-FG now use a walk-forward learned structural layer instead of fixed opponent/H2H weights. The learned layer activates only if it beats the existing Old/G6-10/L5 baseline out of sample.

## v2.17.1 adaptive player role-state

Adds an evidence-weighted local-level state-space correction for player 2PA/3PA/FTA/REB/AST per-minute opportunity rates. The process variance is selected by walk-forward predictive likelihood and model-averaged against the static state; shooting percentages and trader minutes remain separate. See `README_v2_17_1_adaptive_player_role_state.md`.
