# Builds the Asymmetric scenario by selecting trips from a pre-validated pool
# (configs/trips_asym_pool.rou.xml), so every trip is guaranteed routable.
#
# Target: HEAVY_EDGE carries ~4x the traffic of each of the 3 LIGHT_EDGES,
# total ~300 veh/hr, spread across the full 3600s. seed=42 (baked into the
# pool generation already).

import xml.etree.ElementTree as ET
import random

random.seed(42)

HEAVY_EDGE = "A1A0"
LIGHT_EDGES = ["A0B0", "B0B1", "B1A1"]

# Target counts over the 3600s run (matches the locked spec: ~171 veh/hr
# heavy, ~43 veh/hr each light approach => ~300 veh/hr total)
HEAVY_TARGET = 171
LIGHT_TARGET_EACH = 43

pool = ET.parse('configs/trips_asym_pool.rou.xml').getroot()
all_trips = pool.findall('trip')

heavy_trips = [t for t in all_trips if t.get('from') == HEAVY_EDGE]
light_trips = {e: [t for t in all_trips if t.get('from') == e] for e in LIGHT_EDGES}

print(f"Pool available: heavy={len(heavy_trips)}, " +
      ", ".join(f"{e}={len(light_trips[e])}" for e in LIGHT_EDGES))

selected = []

# Sample evenly across the hour rather than just taking the first N
# (which would cluster all heavy trips at the start of the pool's time range)
def sample_spread(trips, n):
    if len(trips) <= n:
        return trips
    trips_sorted = sorted(trips, key=lambda t: float(t.get('depart')))
    step = len(trips_sorted) / n
    return [trips_sorted[int(i * step)] for i in range(n)]

selected += sample_spread(heavy_trips, HEAVY_TARGET)
for e in LIGHT_EDGES:
    selected += sample_spread(light_trips[e], LIGHT_TARGET_EACH)

selected.sort(key=lambda t: float(t.get('depart')))

# Renumber ids sequentially and write output
root = ET.Element('routes')
root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')

for i, t in enumerate(selected):
    t.set('id', str(i))
    root.append(t)

tree = ET.ElementTree(root)
ET.indent(tree, space='    ')
tree.write('configs/trips_asym_selected.rou.xml', encoding='UTF-8', xml_declaration=True)

print(f"Selected: {len(selected)} trips "
      f"(heavy={min(HEAVY_TARGET,len(heavy_trips))}, "
      f"light={sum(min(LIGHT_TARGET_EACH,len(light_trips[e])) for e in LIGHT_EDGES)})")
print(f"Approx rate: ~{len(selected)/1.0:.0f} veh/hr total")
print("Written to configs/trips_asym_selected.rou.xml")