import os
import json

def initialize(map: str):
    config_path = os.path.join(os.getcwd(), "data", "map.json")
    with open(config_path, 'r') as file:
        data = json.load(file)
    matrix = data[map]["matrix"]
    numberOfPlayers = data[map]["numberOfPlayers"]
    numberOfRobots = data[map]["numberOfRobots"]
    tic = data[map]["tic"]
    missions = data[map]["missions"]
    numberOfVips = data[map]["numberOfVips"]

    return matrix, numberOfPlayers, numberOfRobots, numberOfVips, tic, missions