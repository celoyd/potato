import json
from sys import argv

with open(argv[1]) as log:
    for line in log:
        if "lat" in line and "lon" in line:
            line = line.split(" ")
            chunk = " ".join(line[5:9]).replace("'", '"')
            place = json.loads(chunk)
            s = repr(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "coordinates": [place["lon"], place["lat"]],
                        "type": "Point",
                    },
                }
            )
            s = s.replace("'", '"')
            print(s, ",\n")
