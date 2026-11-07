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

Status: observed (wind). The turbine, the flat parcel and the costs are illustrative and labelled so in the notebook.

## Terrain (Section 10 of the notebook)

Two GeoTIFF windows (59.55 to 59.00 W, 47.60 to 47.95 N, EPSG:4326), cut from cloud-optimized GeoTIFFs with HTTP range
reads by `wakewise.fetch_raster(url, path)` so the notebook runs offline:

- `gwa_wreckhouse_ws100.tif`: mean wind speed at 100 m above ground (m/s), about 250 m grid, from the **Global Wind
  Atlas 4** country file for Canada, <https://gwa.cdn.nazkamapps.com/country_tifs_v4/CAN_wind-speed_100m.tif>.
  Developed and operated by DTU Wind and Energy Systems in partnership with the World Bank Group (ESMAP).
  Licence: CC BY 4.0. Retrieved 2026-11-07 19:56 UTC.
- `dem_wreckhouse_glo90.tif`: elevation (m), 3 arc-second (about 90 m) grid, from the **Copernicus DEM GLO-90** tile
  N47 W060 on the AWS Open Data registry,
  <https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N47_00_W060_00_DEM/Copernicus_DSM_COG_30_N47_00_W060_00_DEM.tif>.
  © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union
  and ESA. Retrieved 2026-11-07 19:57 UTC.

Status: elevation is observed (satellite-derived). The wind atlas is a model meant for planning and screening, not for
final wind-farm siting, so the terrain parcel is a screening-level study.

## AI training set (Section 11)

`results/ai_training_set.csv` and `results/ai_qubos.npz` are generated from the data above by `make_ai_training_set.py`
(285 QUBO instances with CVaR-trained QAOA angles). `results/ai_angles.json` holds the angles used on hardware;
Section 11 of the notebook recomputes them from the training set and checks that they match.
