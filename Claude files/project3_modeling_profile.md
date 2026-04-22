# Project 3 Modeling Table Profile

## Overview
- rows: `12300`
- columns carried into the modeling table: `151`
- original-row context columns preserved: `141`
- positive label (`target_recovered_by_retry`): `1806` (14.68%)
- first-retry positive label: `1143` (9.29%)
- rows with any retry chain: `4044` (32.88%)

## Recovery by Soft/Hard Decline
- hard decline: `2149` rows, `0` recovered (0.00%)
- soft decline: `10151` rows, `1806` recovered (17.79%)

## Largest Segments
### By response code
- `05`: `4856` rows, recovery `15.44%`
- `51`: `3352` rows, recovery `22.70%`
- `54`: `708` rows, recovery `0.00%`
- `57`: `592` rows, recovery `9.97%`
- `62`: `554` rows, recovery `0.00%`
- `14`: `407` rows, recovery `0.00%`
- `91`: `228` rows, recovery `30.70%`
- `61`: `202` rows, recovery `10.89%`
- `65`: `195` rows, recovery `10.77%`
- `RC`: `187` rows, recovery `17.65%`

### By merchant country
- `US`: `2472` rows, recovery `15.13%`
- `DE`: `1083` rows, recovery `15.14%`
- `GB`: `960` rows, recovery `15.42%`
- `BR`: `933` rows, recovery `13.40%`
- `IN`: `719` rows, recovery `14.19%`
- `FR`: `623` rows, recovery `15.57%`
- `JP`: `455` rows, recovery `17.58%`
- `AE`: `412` rows, recovery `14.08%`
- `AU`: `380` rows, recovery `8.95%`
- `NL`: `339` rows, recovery `11.50%`

### By processor
- `global-acquirer-b`: `2401` rows, recovery `16.16%`
- `global-acquirer-a`: `1641` rows, recovery `15.72%`
- `high-risk-or-orchestrator-a`: `1336` rows, recovery `21.48%`
- `cross-border-fx-specialist-a`: `1332` rows, recovery `9.38%`
- `cross-border-fx-specialist-b`: `1254` rows, recovery `11.88%`
- `high-risk-or-orchestrator-b`: `1251` rows, recovery `20.70%`
- `regional-card-specialist-b`: `1240` rows, recovery `10.73%`
- `regional-card-specialist-a`: `1171` rows, recovery `10.93%`
- `regional-bank-processor-b`: `227` rows, recovery `9.69%`
- `regional-bank-processor-a`: `224` rows, recovery `11.61%`

## Sufficiency Read
- The dataset is large enough for a baseline tabular model.
- The positive class is healthy enough for MVP modeling, but some segment-level slices will still be sparse.
- This supports training a first recovery-probability model before generating more data.
- We should revisit data expansion only after checking model stability across major segments.
