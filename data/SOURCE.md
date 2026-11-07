# Data provenance

`wreckhouse_2023_hourly.csv`, `wreckhouse_2024_hourly.csv`, `wreckhouse_2025_hourly.csv`: hourly wind observations at
**Wreckhouse, NL** (Environment and Climate Change Canada, climate ID 8404343), frozen so the project runs offline.
The project pools the three years into a long-term wind climate (every hour weighted equally) and uses single years as
robustness checks.

- Source: ECCC MSC GeoMet OGC API, collection `climate-hourly`, e.g.
  <https://api.weather.gc.ca/collections/climate-hourly/items?CLIMATE_IDENTIFIER=8404343&LOCAL_YEAR=2025&properties=LOCAL_DATE,WIND_DIRECTION,WIND_SPEED&limit=10000&f=json>
  (change `LOCAL_YEAR` for 2023 and 2024).
- Retrieved: 2025 on 2026-11-07 15:48 UTC; 2023 and 2024 on 2026-11-07 18:38 UTC, all with `wakewise.fetch_wind(year=...)`.
- Licence: Open Government Licence - Canada (attribution: Environment and Climate Change Canada).
- Columns: `LOCAL_DATE` (local time), `WIND_DIRECTION` (tens of degrees the wind blows from; 0 = calm),
  `WIND_SPEED` (km/h at 10 m).
- Rows: 8,750 (2023), 8,774 (2024), 8,740 (2025). Hours with a missing direction or speed are dropped in `wakewise.load_wind()`.

Status: observed (wind). Everything else in the model (turbine, parcel, costs) is illustrative and labelled so in the notebook.
