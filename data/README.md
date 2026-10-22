# Datasets

Twelve datasets, each with a dirty table, a clean reference table and a rule library.

| Dataset | Source | Task | Label column |
|---------|--------|------|--------------|
| adult | REIN | classification | income |
| beers | REIN | classification | style |
| bike | REIN | regression | cnt |
| breast_cancer | REIN | classification | class |
| har | REIN | clustering | gt |
| mercedes | REIN | regression | y |
| nasa | REIN | regression | sound_pressure_level |
| smartfactory | REIN | classification | labels |
| soilmoisture | REIN | regression | soil_moisture |
| flights | UniClean | classification | arrival_delay_bucket |
| hospitals | UniClean | classification | Condition |
| soccer | UniClean | classification | manager |

The clean and dirty tables are used as published by the REIN benchmark (Abdelaal et al.,
EDBT 2023) and the UniClean benchmark. An `index` column is appended and missing values are
written as `empty`. No errors are injected into these tables. The error injection inside
DemandClean acts only on the training base of the policy.

Each directory holds

- `dirty_index.csv`, the dirty table
- `clean_index.csv`, the clean reference table
- `rules.txt`, the rule library (DOMAIN, REGEX, FD, CFD and DC sections)
- `README.md`, size and label of the table
