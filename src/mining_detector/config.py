"""Pilot AOI and run configuration.

Pilot region: La Pampa / Malinowski river corridor, Madre de Dios, Peru --
one of the most heavily documented illegal artisanal gold-mining fronts in
the Amazon, along the Interoceanic Highway inside/adjacent to the Tambopata
buffer zone.
"""

# (minx, miny, maxx, maxy) in WGS84 -- covers the La Pampa deforestation front.
PILOT_AOI_BBOX = (-70.05, -13.05, -69.75, -12.80)

PILOT_NAME = "madre_de_dios_la_pampa"

# Wet/dry season affects cloud cover heavily in the Amazon -- prefer dates in
# the Jun-Sep dry season for usable low-cloud scenes.
DEFAULT_DATE_T0 = "2026-06-01"
DEFAULT_DATE_T1 = "2026-08-01"

MAX_CLOUD_COVER = 30.0
