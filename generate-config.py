#!/usr/bin/env python

import polars as pl
import numpy as np
import pandas as pd
from uuid import uuid5, UUID
from scipy.stats import bernoulli, truncnorm

# Configuration ---------------------------------------------------------------

n_scenarios = 10
# Process behaviours ----------------------------------------------------------

behaviours = ["Walk", "Cycle", "PT", "Drive"]
behaviour_namespace = UUID("24875ff2-c3ee-449a-85ad-c271bd369caf")
behaviour_uuids = [str(uuid5(behaviour_namespace, b)) for b in behaviours]

behaviour_df = pl.DataFrame({
    "name": behaviours,
    "uuid": behaviour_uuids,
})

behaviour_df.write_json("output/behaviours.json", row_oriented=True)

# Process beliefs -------------------------------------------------------------

belief_namespace = UUID("034d3135-f1f0-441b-a476-9fac37cafc92")

beliefs = np.array([
    "I care about the environment",
    "I want to get to work quickly",
    "I care about the social importance of the car",
    "I want to keep fit",
    "I do not want to perform exercise on my commute",
    "Cycling is hard work",
    "I'm not fit enough to walk",
    "I don't think cycling is cool / fun",
    "Car driving is more convenient",
    "I'm scared of getting hit by a car",
    "My bike might get stolen",
    "Cycling is dangerous",
    "I get to see the environment when I cycle",
    "Walking allows me to experience the environment",
    "I feel unsafe walking"
])

belief_uuids = np.array([str(uuid5(belief_namespace, b)) for b in beliefs])


def n(loc, scale):
    clip_a = -1
    clip_b = 1
    a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
    return truncnorm(a, b, loc=loc, scale=scale)


# This is a belief x behaviour array of random distributions for the parameters
perceptions = np.array([
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
])

relationships = np.array([
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
])

dist_beliefs = [
    bernoulli(0.6) for _i in range(len(beliefs))
]

include_beliefs = np.array([
    b.rvs(size=n_scenarios) for b in dist_beliefs
])

beliefs_scenarios = [pd.DataFrame({
    "beliefs": beliefs[np.where(col)],
    "uuids": belief_uuids[np.where(col)],
    "perceptions": np.array(list(
        {behaviour_uuids[i]: row[i].rvs() for i in range(len(row))}
        for row in perceptions[np.where(col)])),
    "relationships": np.array(list(
        {belief_uuids[i]: row[i].rvs() for i in np.where(col)[0]}
        for row in relationships[np.where(col)]
    ))
}) for col in include_beliefs.T]

for i in range(len(beliefs_scenarios)):
    beliefs_scenarios[i].to_json(
        f"output/beliefs_{i}.json", orient="records"
    )
