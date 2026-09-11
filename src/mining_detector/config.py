"""Pilot AOI and run configuration.

Pilot region: La Pampa / Malinowski river corridor, Madre de Dios, Peru --
one of the most heavily documented illegal artisanal gold-mining fronts in
the Amazon, along the Interoceanic Highway inside/adjacent to the Tambopata
buffer zone.
"""

# (minx, miny, maxx, maxy) in WGS84 -- covers the La Pampa deforestation front.
PILOT_AOI_BBOX = (-70.05, -13.05, -69.75, -12.80)

PILOT_NAME = "madre_de_dios_la_pampa"

# Year-over-year, SAME calendar month, not two dates within one dry-season
# progression: comparing June vs August within a single year confounds real
# land-cover change with seasonal vegetation-stress phenology (verified on
# this AOI -- scene-wide NDVI dropped 0.48->0.32 and NDMI 0.19->0.12 between
# those dates with no plausible mining-driven cause). Matching month-of-year
# keeps sun angle/season/vegetation phenology comparable across t0/t1, so a
# large reflectance shift is far more likely to reflect genuine change.
DEFAULT_DATE_T0 = "2025-07-27"
DEFAULT_DATE_T1 = "2026-07-27"

MAX_CLOUD_COVER = 30.0
