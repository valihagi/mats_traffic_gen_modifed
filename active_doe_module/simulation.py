import json, random

def run_simulation(variations: dict, setup=None):
    """
    Define how to run a simulation in this function
    """
    if not setup:
        with open("setup.json", "r") as fp:
            setup = json.load(fp)
    return { response["Name"]:random.random() for response in setup["Responses"] }