# Hand-off to path planning (the ONLY thing downstream consumes)

This module selects a landing site; it does NOT plan a route. Outputs in outputs/:

Rasters (identical grid, south-polar-stereo CRS, 5 m, shared nodata):
- suitability.tif  float32 [0,1]  multi-criteria landing suitability
- hazard.tif       uint8 {0,1}    hard no-go (steep / rough / crater interior)
- slope.tif        float32 deg
- roughness.tif    float32

Vectors (EPSG:4326 lon/lat):
- landing_candidates.geojson  ranked Points (props: rank, score, slope, roughness, dist_to_F2)
- landing_site_polygon.geojson  safety ellipse around the top site
- target_crater.geojson         F2 outline (rover destination)

manifest.json: per-layer path, description, units, CRS, pixel size, nodata.
Rule: produce artifacts only — no route, cost-of-motion surface, or waypoints.
