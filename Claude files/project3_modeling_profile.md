# Project 3 Modeling Table Profile

## Overview
- rows: `12300`
- columns carried into the modeling table: `151`
- original-row context columns preserved: `141`
- positive label (`target_recovered_by_retry`): `1271` (10.33%)
- first-retry positive label: `1033` (8.40%)
- rows with any retry chain: `4044` (32.88%)

## Recovery by Soft/Hard Decline
- hard decline: `2149` rows, `0` recovered (0.00%)
- soft decline: `10151` rows, `1271` recovered (12.52%)

## Largest Segments
### By response code
- `05`: `4856` rows, recovery `10.85%`
- `51`: `3352` rows, recovery `17.15%`
- `54`: `708` rows, recovery `0.00%`
- `57`: `592` rows, recovery `5.24%`
- `62`: `554` rows, recovery `0.00%`
- `14`: `407` rows, recovery `0.00%`
- `91`: `228` rows, recovery `16.23%`
- `61`: `202` rows, recovery `10.40%`
- `65`: `195` rows, recovery `8.21%`
- `RC`: `187` rows, recovery `3.74%`

### By merchant country
- `US`: `2472` rows, recovery `13.39%`
- `DE`: `1083` rows, recovery `13.76%`
- `GB`: `960` rows, recovery `11.56%`
- `BR`: `933` rows, recovery `8.68%`
- `IN`: `719` rows, recovery `3.06%`
- `FR`: `623` rows, recovery `12.52%`
- `JP`: `455` rows, recovery `16.26%`
- `AE`: `412` rows, recovery `7.77%`
- `AU`: `380` rows, recovery `10.00%`
- `NL`: `339` rows, recovery `10.62%`

### By processor
- `global-acquirer-b`: `2401` rows, recovery `11.37%`
- `global-acquirer-a`: `1641` rows, recovery `12.00%`
- `high-risk-or-orchestrator-a`: `1336` rows, recovery `14.07%`
- `cross-border-fx-specialist-a`: `1332` rows, recovery `7.73%`
- `cross-border-fx-specialist-b`: `1254` rows, recovery `7.26%`
- `high-risk-or-orchestrator-b`: `1251` rows, recovery `13.91%`
- `regional-card-specialist-b`: `1240` rows, recovery `7.74%`
- `regional-card-specialist-a`: `1171` rows, recovery `8.80%`
- `regional-bank-processor-b`: `227` rows, recovery `8.37%`
- `regional-bank-processor-a`: `224` rows, recovery `5.36%`

## Sufficiency Read
- The dataset is large enough for a baseline tabular model.
- The positive class is healthy enough for MVP modeling, but some segment-level slices will still be sparse.
- This supports training a first recovery-probability model before generating more data.
- We should revisit data expansion only after checking model stability across major segments.
