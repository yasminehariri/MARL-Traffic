import xml.etree.ElementTree as ET

high = ET.parse('configs/trips_peak_high_routed.rou.xml').getroot()
low = ET.parse('configs/trips_peak_low_routed.rou.xml').getroot()

# Create new root
merged = ET.Element('routes')
merged.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
merged.set('xsi:noNamespaceSchemaLocation', 
           'http://sumo.dlr.de/xsd/routes_file.xsd')

# Add all vehicles from high period
for vehicle in high.findall('vehicle'):
    merged.append(vehicle)

# Add all vehicles from low period — renumber IDs to avoid conflicts
high_count = len(high.findall('vehicle'))
for i, vehicle in enumerate(low.findall('vehicle')):
    vehicle.set('id', str(high_count + i))
    merged.append(vehicle)

# Sort by depart time
vehicles = merged.findall('vehicle')
vehicles.sort(key=lambda v: float(v.get('depart')))

# Rebuild merged with sorted vehicles
final = ET.Element('routes')
final.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
final.set('xsi:noNamespaceSchemaLocation',
          'http://sumo.dlr.de/xsd/routes_file.xsd')
for v in vehicles:
    final.append(v)

tree = ET.ElementTree(final)
ET.indent(tree, space='    ')
tree.write('configs/routes_peak.rou.xml', 
           encoding='UTF-8', xml_declaration=True)

print(f"Merged: {len(high.findall('vehicle'))} high-period + "
      f"{len(low.findall('vehicle'))} low-period vehicles")
print(f"Total: {len(vehicles)} vehicles")
print("Written to configs/routes_peak.rou.xml")