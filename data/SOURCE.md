# Data provenance

`wreckhouse_<year>_hourly.csv` for every year 2007 to 2025: hourly wind observations at **Wreckhouse, NL**
(Environment and Climate Change Canada, climate ID 8404343), frozen so the project runs offline. The main instance pools
2023 to 2025 into a long-term wind climate (every hour weighted equally); the robustness test uses all 19 years.

- Source: ECCC MSC GeoMet OGC API, collection `climate-hourly`, e.g.
  <https://api.weather.gc.ca/collections/climate-hourly/items?CLIMATE_IDENTIFIER=8404343&LOCAL_YEAR=2025&properties=LOCAL_DATE,WIND_DIRECTION,WIND_SPEED&limit=10000&f=json>
  (change `LOCAL_YEAR` for 2023 and 2024).
- Retrieved: 2025 on 2026-11-07 15:48 UTC; 2023 and 2024 on 2026-11-07 18:38 UTC; 2007 to 2022 on 2026-11-07 19:31 UTC, all with `wakewise.fetch_wind(year=...)`.
- Licence: Open Government Licence - Canada (attribution: Environment and Climate Change Canada).
- Columns: `LOCAL_DATE` (local time), `WIND_DIRECTION` (tens of degrees the wind blows from; 0 = calm),
  `WIND_SPEED` (km/h at 10 m).
- Rows: about 8,500 to 8,780 per year, except 2021 (6,261, about 70% complete). 2006 is excluded (only 2,964 hours). Hours with a missing direction or speed are dropped in `wakewise.load_wind()`.

Status: observed (wind). Everything else in the model (turbine, parcel, costs) is illustrative and labelled so in the notebook.
