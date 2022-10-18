#!/usr/bin/env python

import polars as pl
from uuid import uuid5, UUID

# Process behaviours ----------------------------------------------------------

behaviours = ["Walk", "Cycle", "PT", "Drive"]
behaviour_namespace = UUID("24875ff2-c3ee-449a-85ad-c271bd369caf")
uuids = [str(uuid5(behaviour_namespace, b)) for b in behaviours]

df = pl.DataFrame({
    "name": behaviours,
    "uuid": uuids,
})

df.write_json("output/behaviours.json", row_oriented=True)
