#!/usr/bin/env python

import polars as pl
import os
import numpy as np
import json
import pandas as pd
import boto3
import logging
import networkx
from botocore.exceptions import ClientError
from uuid import uuid5, UUID
from scipy.stats import bernoulli, truncnorm

# s3 --------------------------------------------------------------------------


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


# Configuration ---------------------------------------------------------------

n_scenarios = 10

# Process behaviours ----------------------------------------------------------

behaviours = ["Walk", "Cycle", "PT", "Drive"]
behaviour_namespace = UUID("24875ff2-c3ee-449a-85ad-c271bd369caf")
behaviour_uuids = np.array(
    [str(uuid5(behaviour_namespace, b)) for b in behaviours])

behaviour_df = pl.DataFrame({
    "name": behaviours,
    "uuid": behaviour_uuids,
})

behaviour_df.write_json("output/behaviours.json", row_oriented=True)

# upload_file("output/behaviours.json", "concept-abm",
#             object_name="configuration/global/behaviours.json")

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


def n(loc, scale):
    clip_a = -1
    clip_b = 1
    a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
    return truncnorm(a, b, loc=loc, scale=scale)


# This is a belief x behaviour array of random distributions for the parameters
# Behaviours: walk, cycle, PT, drive
perceptions = np.array([
    # I care about the environment
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    # I want to get to work quickly
    [n(-0.3, 0.1), n(0.0, 0.1), n(0.1, 0.1), n(0.5, 0.1)],
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
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
])

relationships = np.array([
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1)
    ],
    [
        n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1),
        n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1),
        n(0.4, 0.1), n(-0.9, 0.1), n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1),
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

    # upload_file(f"output/beliefs_{i}.json", "concept-abm",
    #             object_name=f"configuration/scenario/{i}/beliefs.json")

# PRS -------------------------------------------------------------------------

# This is a belief x behaviour array of random distributions for the parameters
# Behaviours: walk, cycle, PT, drive
prs_mat = np.array([
    # I care about the environment
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    # I want to get to work quickly
    [n(-0.3, 0.1), n(0.0, 0.1), n(0.1, 0.1), n(0.5, 0.1)],
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
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
    [n(0.6, 0.1), n(0.7, 0.1), n(0.4, 0.1), n(-0.9, 0.1)],
])

prs = list([
    list([
        {
            "beliefUuid": belief_uuids[i],
            "behaviourUuid": behaviour_uuids[j],
            "value": prs_mat[i, j].rvs()
        } for i in np.where(col)[0] for j in range(prs_mat.shape[1])
    ])
    for col in include_beliefs.T
])

for i, p in enumerate(prs):
    with open(f"output/prs_{i}.json", "w") as outfile:
        json.dump(p, outfile)

    # upload_file(f"output/prs_{i}.json", "concept-abm",
    #             object_name=f"configuration/scenario/{i}/prs.json")

# Generate agent --------------------------------------------------------------

n_agents = 5000

agent_namespace = UUID("1a9e3ee9-a068-41f8-9f46-a5f684f0101e")

agent_uuids = [
    str(uuid5(agent_namespace, f"agent_20221019v1_{i}")) for i in range(n_agents)]

# Generate friends

a, b = (0 - 0.3) / 0.1, (1 - 0.3) / 0.1
p_dist = truncnorm(a, b, loc=0.3, scale=0.1)

networks = [
    networkx.watts_strogatz_graph(
        n=n_agents,
        k=np.random.binomial(20, 0.5),
        p=p_dist.rvs()
    ) for _i in range(n_scenarios)
]

# 80% chance of being friends with yourself
rng = np.random.default_rng()

for network in networks:
    for i in range(n_agents):
        if rng.random() <= 0.8:
            network.add_edge(i, i)

# Add weights
a, b = (0 - 0.5) / 0.15, (1 - 0.5) / 0.15
w_dist = truncnorm(a, b, loc=0.5, scale=0.15)

for network in networks:
    for edge in network.edges:
        network.edges[edge[0], edge[1]]["weight"] = w_dist.rvs()

friends = np.full([n_scenarios, n_agents], {})
for i, network in enumerate(networks):
    for edge in network.edges:
        friends[i, edge[0]][agent_uuids[edge[1]]
                            ] = network.edges[edge[0], edge[1]]["weight"]

# Generate deltas

deltas = rng.normal(loc=1.0-0.001, scale=0.1,
                    size=(n_scenarios, n_agents, len(belief_uuids)))
# Ensure fully positive (v. unlikely for this not to be the case)
deltas = np.abs(deltas) + 0.0001


# Generate activations

# Each scenario has a different cutoff prob. distributed N(0.5,0.1) truncated
loc = 0.5
scale = 0.1
a, b = (0 - loc) / scale, (1 - loc) / scale
pr = truncnorm.rvs(a, b, loc=loc, scale=scale, size=n_scenarios)

# Each initial activation drawn from 0, 0.1 capped at -1, +1.


def random_activation(scenario):
    rng = np.random.default_rng()
    if rng.random() <= pr[scenario]:
        return 0.0
    else:
        loc = 0.0
        scale = 0.1
        a, b = (-1 - loc) / scale, (1 - loc) / scale
        return truncnorm.rvs(a, b, loc=loc, scale=scale)


activations = [
    [
        {
            0: {
                belief_uuid: random_activation(i)
                for belief_uuid in belief_uuids[np.where(include_beliefs[:, i])[0]]
            }
        }
        for j in range(n_agents)
    ] for i in range(n_scenarios)
]

# Generate initial actions


def choose_initial_actions(activations, prs):
    return np.argmax(np.dot(activations, prs), axis=1).reshape(n_agents)


prs_select_mat = np.zeros(
    (n_scenarios, len(belief_uuids), len(behaviour_uuids)), dtype=np.float64)

for i, p in enumerate(prs):
    for inner in p:
        prs_select_mat[
            0,
            np.where(belief_uuids == inner["beliefUuid"])[0][0],
            np.where(behaviour_uuids == inner["behaviourUuid"])[0][0],
        ] = inner["value"]

initial_actions = [
    choose_initial_actions(
        np.array([
            [
                activations[i][agent_i][0][belief_uuid]
                for belief_uuid in belief_uuids[np.where(include_beliefs[:, i])[0]]
            ] for agent_i in range(n_agents)
        ]),
        prs_select_mat[i][np.where(include_beliefs[:, i])[0]]
    )
    for i in range(n_scenarios)
]

actions = [
    [
        {
            0: behaviour_uuids[initial_actions[i][j]]
        } for j in range(n_agents)
    ] for i in range(n_scenarios)
]
# Process and save agents

agents = [
    pd.DataFrame({
        "uuid": agent_uuids,
        "friends": friends[i],
        "deltas": [
            {
                belief_uuids[belief_i]: deltas[i, agent_i, belief_i]
                for belief_i in np.where(include_beliefs[:, i])[0]
            } for agent_i in range(n_agents)
        ],
        "activations": activations[i],
        "actions": actions[i]
    }) for i in range(n_scenarios)
]

for i in range(n_scenarios):
    agents[i].to_json(
        f"output/agents_{i}.json.zst", orient="records"
    )

    # upload_file(f"output/beliefs_{i}.json", "concept-abm",
    #             object_name=f"configuration/scenario/{i}/beliefs.json")
