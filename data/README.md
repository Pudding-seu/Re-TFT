# Data

## Belgium National Grid Dataset (primary benchmark)

The Belgian transmission-level load dataset used in all primary experiments is
**not redistributed** in this repository due to Elia's open-data licence terms,
which require obtaining the data from the official portal.

### Acquisition

1. **Load (target variable):** Elia Open Data Portal —
   https://www.elia.be/en/grid-data/open-data/load
   Product: *Total Load*, resolution **15 min**, year **2023**.

2. **Meteorological covariates:** Open-Meteo historical API
   (https://open-meteo.com/), sampled at representative nodes across Belgium
   and spatially aggregated per 15-min interval.

3. Assemble as `data/belgium.csv` with exactly these columns:

   | Column        | Description                                      |
   |---------------|--------------------------------------------------|
   | `date`        | Timestamp, format `YYYY/M/D H:MM` (e.g. `2023/1/1 0:15`) |
   | `humidity`    | Spatial average relative humidity (%)            |
   | `speed`       | Spatial average wind speed (m/s)                 |
   | `mintem`      | Spatial minimum temperature (K)                  |
   | `temperature` | Spatial average temperature (K)                  |
   | `maxtem`      | Spatial maximum temperature (K)                  |
   | `load`        | Total measured load (MW) — forecast target       |

### Expected file

- **35,041 lines** (header + 35,040 records = 365 days × 96 steps)
- Continuous 15-min frequency, year 2023
- File size: **2,935,897 bytes**

### Integrity verification

The exact file used for all reported experiments has the following checksum.
Verify your independently prepared copy against it:

```
SHA-256: D890765C9282895A10795DA086BE0EB8A620D0C0E48C0173CAC998B03C253BC5
```

```powershell
# Windows
Get-FileHash data/belgium.csv -Algorithm SHA256
```
```bash
# Linux / macOS
sha256sum data/belgium.csv
```

> Note: the checksum is byte-exact. If your download covers a different export
> window or the portal has revised historical values, the hash will differ even
> when the content is scientifically equivalent; in that case compare summary
> statistics (row count, load min/max ≈ 6,000–13,000 MW) instead.

### Quick check

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/belgium.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M", errors="coerce")
print(df.shape)            # (35040, 7)
print(df["load"].min(), df["load"].max())
print(df.isna().sum())
EOF
```

## London Microgrid Dataset (regional-scale benchmark)

The London dataset is reconstructed from the **"Smart Meters in London"**
repository, whose raw load originates from the **"Low Carbon London"** project
led by UK Power Networks (half-hourly readings from 5,500+ households).

### Acquisition & preprocessing

1. **Load (target):** "Smart Meters in London" smart-meter readings.
   Because meters were installed on a rolling basis, a *threshold-filtering*
   step isolates a stable window in which >98% of meters are online; the
   per-household records are then spatially aggregated and **down-sampled to
   hourly** resolution.

2. **Meteorological covariates:** hourly temperature, relative humidity and
   wind speed for Greater London. These were taken from **archived historical
   weather records originally obtained from the Dark Sky API**. Daily maximum
   and minimum temperatures are broadcast across the hourly steps of each day.

3. The aggregated load and weather are aligned by an inner join on timestamp;
   rows with missing values are dropped, yielding seven columns
   (`Date, Load, Temperature, Humidity, Wind Speed, MinTem, MaxTem`).

> **Reproducibility note — Dark Sky API.** The Dark Sky API was discontinued
> for public access in 2023, so the weather features above are drawn from
> previously archived records rather than live queries. If you cannot access
> the original archive, an equivalent public reanalysis source
> (e.g., [Open-Meteo](https://open-meteo.com/)) can be substituted for the
> Greater-London nodes to regenerate equivalent features.

## Spain National Grid Dataset (macro-scale benchmark)

The Spain dataset is derived from the open-source **Spanish electricity grid**
dataset on Kaggle (≈4 years of hourly total load, ~35,064 steps). Hourly
meteorological data are aggregated from five major cities (Madrid, Barcelona,
Valencia, Seville, Bilbao); the selected features are temperature, atmospheric
pressure, humidity, wind speed and cloud cover.

## What is / is not included in this repository

To keep the repository lightweight and within dataset licences, the raw and
processed CSVs are **not redistributed here**. This `data/` folder documents how
to obtain and assemble each dataset using the recipes above so that
`data/belgium.csv` (and the optional London/Spain equivalents) can be rebuilt
locally. The expected column schema and, for Belgium, the SHA-256 checksum are
provided so an independently prepared copy can be verified.
