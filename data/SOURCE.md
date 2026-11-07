# Data provenance

`wreckhouse_2025_hourly.csv`: hourly wind observations for 2025 at **Wreckhouse, NL**
(Environment and Climate Change Canada, climate ID 8404343), frozen so the project runs offline.

- Source: ECCC MSC GeoMet OGC API, collection `climate-hourly`
  <https://api.weather.gc.ca/collections/climate-hourly/items?CLIMATE_IDENTIFIER=8404343&LOCAL_YEAR=2025&properties=LOCAL_DATE,WIND_DIRECTION,WIND_SPEED&limit=10000&f=json>
- Retrieved: 2026-11-07 15:48 UTC with `wakewise.fetch_wind()`.
- Licence: Open Government Licence - Canada (attribution: Environment and Climate Change Canada).
- Columns: `LOCAL_DATE` (local time), `WIND_DIRECTION` (tens of degrees the wind blows from; 0 = calm),
  `WIND_SPEED` (km/h at 10 m).
- Rows: 8,740 hours. Hours with a missing direction or speed are dropped in `wakewise.load_wind()`.

Status: observed (wind). Everything else in the model (turbine, parcel, costs) is illustrative and labelled so in the notebook.
