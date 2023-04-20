#!/usr/bin/env python
"""This script generates configurations for the concept model"""

import os
import json
import sys
from uuid import uuid5, UUID

import numpy as np
import pandas as pd
import polars as pl
from networkx import watts_strogatz_graph
from scipy.stats import bernoulli, truncnorm

# Configuration ---------------------------------------------------------------

SCENARIO_ID = sys.argv[1]
os.makedirs(f"output/scenario/{SCENARIO_ID}")
np.random.seed(543879 + int(SCENARIO_ID))

# Process behaviours ----------------------------------------------------------

behaviours = ["Walk", "Cycle", "PT", "Drive"]
behaviour_namespace = UUID("24875ff2-c3ee-449a-85ad-c271bd369caf")
behaviour_uuids = np.array(
    [str(uuid5(behaviour_namespace, b)) for b in behaviours])

behaviour_df = pl.DataFrame({
    "name": behaviours,
    "uuid": behaviour_uuids,
})

behaviour_df.write_json(
    f"output/scenario/{SCENARIO_ID}/behaviours.json", row_oriented=True)

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
    "I feel unsafe walking",

    "Driving / PT allows me to get to work not sweaty",
    "The traffic is too bad to drive",
    "Driving lets me get to work on time",
    "PT is unreliable",
    "My car is bad"
])

belief_uuids = np.array([str(uuid5(belief_namespace, b)) for b in beliefs])


def truncn_at_m1_1(location, scale):
    """Return a truncated normal distribution, capped on -1, 1.

    :param l: The location (i.e., the mean)
    :param s: The scale (i.e., the variance)
    :return: The scipy distribution.
    """
    clip_a = -1
    clip_b = 1
    n_a, n_b = (clip_a - location) / scale, (clip_b - location) / scale
    return truncnorm(n_a, n_b, loc=location, scale=scale)


# This is a belief x behaviour array of random distributions for the parameters
# Behaviours: walk, cycle, PT, drive
perceptions = np.array([
    # I care about the environment
    [truncn_at_m1_1(0.6, 0.1 / 3), truncn_at_m1_1(0.7, 0.1 / 3),
     truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(-0.9, 0.1 / 3)],
    # I want to get to work quickly
    [truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3)],
    # I care about the social importance of the car
    [truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
     truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.8, 0.1 / 3)],
    # I want to keep fit
    [truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.8, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3)],
    # I do not want to perform exercise on my commute
    [truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.6, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.05 / 3), truncn_at_m1_1(0.1, 0.05 / 3)],

    # Cycling is hard work
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.8, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # I'm not fit enough to walk
    [truncn_at_m1_1(-0.8, 0.1 / 3), truncn_at_m1_1(-0.8, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.05 / 3), truncn_at_m1_1(0.1, 0.05 / 3)],
    # I don't think cycling is cool / fun
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # Car driving is more convenient
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0., 0.05 / 3),
     truncn_at_m1_1(-0.1, 0.05 / 3), truncn_at_m1_1(0.5, 0.1 / 3)],
    # I'm scared of being hit by a car
    [truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.7, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],

    # My bike might get stolen
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # Cycling is dangerous
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
     truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3)],
    # I get to see the environment when I cycle
    [truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.7, 0.1 / 3),
     truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3)],
    # Walking allows me to experience the environment
    [truncn_at_m1_1(0.7, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
     truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3)],
    # I feel unsafe walking
    [truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3)],

    # Driving / PT allows me to get to work not sweaty
    [truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
     truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3)],
    # The traffic is too bad to drive
    [truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
     truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.9, 0.1 / 3)],
    # Driving lets me get to work on time
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3),
     truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3)],
    # PT is unreliable
    [truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
     truncn_at_m1_1(-0.9, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3)],
    # My car is bad
    [truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
     truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3)],
])

relationships = np.array([
    # I care about the environment
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I want to get to work quickly
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.5, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I care about the social importance of the car
    [
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3)
    ],
    # I want to keep fit
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(-0.4, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I do not want to perform exercise on my commute
    [
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # Cycling is hard work
    [
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I'm not fit enough to walk
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.5, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I don't think cycling is cool / fun
    [
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # Car driving is more convenient
    [
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3)
    ],
    # I'm scared of being hit by a car
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.6, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # My bike might get stolen
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # Cycling is dangerous
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I get to see the environment when I cycle
    [
        truncn_at_m1_1(0.5, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # Walking allows me to experience the environment
    [
        truncn_at_m1_1(0.5, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # I feel unsafe walking
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3),
        truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # Driving / PT allows me to get to work not sweaty
    [
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3),
        truncn_at_m1_1(0.6, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.5, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # The traffic is too bad to drive
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3)
    ],
    # Driving lets me get to work on time
    [
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3),
        truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
        truncn_at_m1_1(0.6, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(-0.2, 0.1 / 3)
    ],
    # PT is unreliable
    [
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3),
        truncn_at_m1_1(0.5, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
    # My car is bad
    [
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
        truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
        truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3)
    ],
])

dist_beliefs = [
    1 for _i in range(len(beliefs))
]

include_beliefs = np.array([1 for _b in dist_beliefs])

beliefs_df = pd.DataFrame({
    "name": beliefs[np.where(include_beliefs)],
    "uuid": belief_uuids[np.where(include_beliefs)],
    "perceptions": np.array([
        {
            behaviour_uuids[i]: row[i].rvs() for i in range(len(row))
        } for row in perceptions[np.where(include_beliefs)]
    ]),
    "relationships": np.array([
        {
            belief_uuids[i]: row[i].rvs() for i in np.where(include_beliefs)[0]
        } for row in relationships[np.where(include_beliefs)]
    ])
})

beliefs_df.to_json(
    f"output/scenario/{SCENARIO_ID}/beliefs.json",
    orient="records"
)

# PRS -------------------------------------------------------------------------

# This is a belief x behaviour array of random distributions for the parameters
# Behaviours: walk, cycle, PT, drive
prs_mat = np.array([
    # I care about the environment
    [truncn_at_m1_1(0.6, 0.1 / 3), truncn_at_m1_1(0.7, 0.1 / 3),
     truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(-0.9, 0.1 / 3)],
    # I want to get to work quickly
    [truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.5, 0.1 / 3)],
    # I care about the social importance of the car
    [truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(0.0, 0.1 / 3),
     truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(0.8, 0.1 / 3)],
    # I want to keep fit
    [truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.8, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3)],
    # I do not want to perform exercise on my commute
    [truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.6, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.05 / 3), truncn_at_m1_1(0.1, 0.05 / 3)],
    # Cycling is hard work
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.8, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # I'm not fit enough to walk
    [truncn_at_m1_1(-0.8, 0.1 / 3), truncn_at_m1_1(-0.8, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.05 / 3), truncn_at_m1_1(0.1, 0.05 / 3)],
    # I don't think cycling is cool / fun
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.2, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # Car driving is more convenient
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0., 0.05 / 3),
     truncn_at_m1_1(-0.1, 0.05 / 3), truncn_at_m1_1(0.5, 0.1 / 3)],
    # I'm scared of being hit by a car
    [truncn_at_m1_1(-0.2, 0.1 / 3), truncn_at_m1_1(-0.7, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # My bike might get stolen
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
     truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3)],
    # Cycling is dangerous
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
     truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3)],
    # I get to see the environment when I cycle
    [truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.7, 0.1 / 3),
     truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3)],
    # Walking allows me to experience the environment
    [truncn_at_m1_1(0.7, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
     truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3)],
    # I feel unsafe walking
    [truncn_at_m1_1(-0.5, 0.1 / 3), truncn_at_m1_1(-0.1, 0.1 / 3),
     truncn_at_m1_1(0.1, 0.1 / 3), truncn_at_m1_1(0.1, 0.1 / 3)],
    # Driving / PT allows me to get to work not sweaty
    [truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(-0.5, 0.1 / 3),
     truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3)],
    # The traffic is too bad to drive
    [truncn_at_m1_1(0.4, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3),
     truncn_at_m1_1(-0.3, 0.1 / 3), truncn_at_m1_1(-0.9, 0.1 / 3)],
    # Driving lets me get to work on time
    [truncn_at_m1_1(0.0, 0.05 / 3), truncn_at_m1_1(0.0, 0.05 / 3),
     truncn_at_m1_1(-0.1, 0.1 / 3), truncn_at_m1_1(0.6, 0.1 / 3)],
    # PT is unreliable
    [truncn_at_m1_1(0.3, 0.1 / 3), truncn_at_m1_1(0.3, 0.1 / 3),
     truncn_at_m1_1(-0.9, 0.1 / 3), truncn_at_m1_1(0.4, 0.1 / 3)],
    # My car is bad
    [truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(0.2, 0.1 / 3),
     truncn_at_m1_1(0.2, 0.1 / 3), truncn_at_m1_1(-0.3, 0.1 / 3)],
])

prs = [
    {
        "beliefUuid": belief_uuids[i],
        "behaviourUuid": behaviour_uuids[j],
        "value": prs_mat[i, j].rvs()
    } for i in np.where(include_beliefs)[0] for j in range(prs_mat.shape[1])
]

with open(f"output/scenario/{SCENARIO_ID}/prs.json", "w", encoding="utf-8") as outfile:
    json.dump(prs, outfile)

# Generate agent --------------------------------------------------------------

N_AGENTS = 5000

agent_namespace = UUID("1a9e3ee9-a068-41f8-9f46-a5f684f0101e")

agent_uuids = [
    str(uuid5(agent_namespace, f"agent_20221019v1_{i}")) for i in range(N_AGENTS)]

# Generate friends

network = watts_strogatz_graph(
    n=N_AGENTS,
    k=10,
    p=0.3
)

# 80% chance of being friends with yourself
probabilities_of_self_friendship = np.random.random(size=N_AGENTS)

for i, prob in enumerate(probabilities_of_self_friendship):
    if prob <= 0.8:
        network.add_edge(i, i)

# Add weights


def w_dist_gen():
    """Generate a distribution to draw weights from"""
    alpha, beta = (0 - 0.5) / 0.15, (1 - 0.5) / 0.15
    return truncnorm(alpha, beta, loc=0.5, scale=0.15)


w_dist = w_dist_gen()

for edge in network.edges:
    network.edges[edge[0], edge[1]]["weight"] = w_dist.rvs()

friends = np.array([{} for _i in range(N_AGENTS)])
for u, v, weight in network.edges(data="weight"):
    friends[u][agent_uuids[v]] = weight


# Generate deltas

deltas = np.random.normal(loc=1.0-0.001, scale=0.1,
                          size=(N_AGENTS, len(belief_uuids)))
# Ensure fully positive (v. unlikely for this not to be the case)
deltas = np.abs(deltas) + 0.0001


# Generate activations
# Each initial activation drawn from 0, 0.1 capped at -1, +1.


def random_activation():
    """Generate a random activation from N(0, 0.1) capped at -1, 1."""
    if np.random.random() <= 0.5:
        return 0.0
    loc = 0.0
    scale = 0.1
    alpha, beta = (-1 - loc) / scale, (1 - loc) / scale
    return truncnorm.rvs(alpha, beta, loc=loc, scale=scale)


activations = [
    {
        0: {
            belief_uuid: random_activation()
            for belief_uuid in belief_uuids[np.where(include_beliefs)[0]]
        }
    }
    for _j in range(N_AGENTS)
]

# scenario
for agent_id in range(N_AGENTS):
    if np.random.random() <= 0.4:
        activations[agent_id][0][belief_uuids[11]] = truncn_at_m1_1(-0.5,0.15).rvs()

# Generate initial actions


def choose_initial_actions(activations, prs):
    """Choose the initial actions of agents."""
    # pylint: disable=redefined-outer-name
    return np.argmax(np.dot(activations, prs), axis=1).reshape(N_AGENTS)


prs_select_mat = np.zeros(
    (len(belief_uuids), len(behaviour_uuids)), dtype=np.float64)

for inner in prs:
    prs_select_mat[
        np.where(belief_uuids == inner["beliefUuid"])[0][0],
        np.where(behaviour_uuids == inner["behaviourUuid"])[0][0],
    ] = inner["value"]

initial_actions = choose_initial_actions(
    np.array([
        [
            activations[agent_i][0][belief_uuid]
            for belief_uuid in belief_uuids[np.where(include_beliefs)[0]]
        ] for agent_i in range(N_AGENTS)
    ]),
    prs_select_mat[np.where(include_beliefs)[0]]
)

actions = [
    {
        0: behaviour_uuids[initial_actions[j]]
    } for j in range(N_AGENTS)
]
# Process and save agents

agents = pd.DataFrame({
    "uuid": agent_uuids,
    "friends": friends,
    "deltas": [
        {
            belief_uuids[belief_i]: deltas[agent_i, belief_i]
            for belief_i in np.where(include_beliefs)[0]
        } for agent_i in range(N_AGENTS)
    ],
    "activations": activations,
    "actions": actions
})

agents.to_json(
    f"output/scenario/{SCENARIO_ID}/agents.json.zst", orient="records"
)