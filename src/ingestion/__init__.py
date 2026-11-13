"""
Data ingestion modules for CHBIndex.

Each module pulls from one external data source and writes to BigQuery:
    census_tiger.py    — Census tract geometries (TIGER API)
    epa_aqs.py         — PM2.5 monitor readings (EPA AQS API)
    gee_lst.py         — Landsat LST rasters (Google Earth Engine)
    cdc_places.py      — Health outcome rates (CDC PLACES Socrata)
    hospitals.py       — Hospital locations, beds, ED status (CA: CDPH licensing data)
    census_blocks.py   — Census 2020 block population + interior points (dashboard hex mask)
    acs_race_ethnicity.py — ACS B03002 race/ethnicity counts (equity audit only)
"""
