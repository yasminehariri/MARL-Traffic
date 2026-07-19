# Generates asymmetric route file
# One approach (A1A0 — northbound) carries 4x more traffic than others
# Total ~300 veh/hr, seed 42

import random
random.seed(42)

HEAVY_EDGE = "A1A0"
LIGHT_EDGES = ["A0B0", "B0B1", "B1A1"]

# All edges for destination sampling
ALL_EDGES = ["A0A1", "A1A0", "A0B0", "B0A0", "A1B1", "B1A1", "B0B1", "B1B0"]

# periods in seconds
HEAVY_PERIOD = 21   # ~171 veh/hr
LIGHT_PERIOD = 84   # ~43 veh/hr each
END_TIME = 3600

vehicles = []
vid = 0

# Heavy approach vehicles
t = 0.0
while t < END_TIME:
    # pick a destination different from origin
    dest = random.choice([e for e in ALL_EDGES if e != HEAVY_EDGE])
    vehicles.append((t, vid, HEAVY_EDGE, dest))
    vid += 1
    t += HEAVY_PERIOD

# Light approach vehicles
for edge in LIGHT_EDGES:
    t = 0.0
    while t < END_TIME:
        dest = random.choice([e for e in ALL_EDGES if e != edge])
        vehicles.append((t, vid, edge, dest))
        vid += 1
        t += LIGHT_PERIOD

# Sort by departure time
vehicles.sort(key=lambda x: x[0])

# Write XML
lines = []
lines.append('<?xml version="1.0" encoding="UTF-8"?>')
lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation='
             '"http://sumo.dlr.de/xsd/routes_file.xsd">')

for depart, vid, from_edge, to_edge in vehicles:
    lines.append(f'    <trip id="{vid}" depart="{depart:.2f}" '
                 f'from="{from_edge}" to="{to_edge}"/>')

lines.append('</routes>')

with open('configs/routes_asym_trips.xml', 'w', encoding='UTF-8') as f:
    f.write('\n'.join(lines))

print(f"Generated {len(vehicles)} trips")
print(f"Heavy ({HEAVY_EDGE}): ~{3600//HEAVY_PERIOD} vehicles")
print(f"Light (each): ~{3600//LIGHT_PERIOD} vehicles")
print(f"Total rate: ~{(3600//HEAVY_PERIOD + 3*(3600//LIGHT_PERIOD))} veh/hr approx")