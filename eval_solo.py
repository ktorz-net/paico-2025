#!env python3
import json, time, matplotlib.pyplot as plt
import hacka.games.moveit as moveit

from grpred.team import bots as rbots
from grporange.grporange.team import bots as obots
from grpbleu.src.team import bots as bbots
from grpvert.team import bots as vbots

import evaltool

# Build the list of bots.
teams= [ rbots, obots, bbots, vbots ]
teamNames= [ "red", "orange", "blue", "green" ]
okTeams= [
    [False, True],
    [False, False, False, False],
    [False, True],
    [False, False, False]
]

eval= evaltool.Eval( "solo", 10 )

challengers= eval.initChallengers(
    "small-1",
    teams,
    teamNames,
    okTeams
)

eval.write( "\n## Solo - Small - No VIP\n" )
eval.bench(challengers, ["small-1", "small-2", "small-3"] )

eval.write( "\n## Solo - Medium - No VIP\n" )
eval.bench(challengers, ["medium-1", "medium-2", "medium-3", "medium-4"] )

eval.write( "\n## Solo - Increasing Medium 2\n" )
eval.increasingTeam( challengers, "medium-2" )

eval.write( "\n## Solo - Large - No VIP\n" )
eval.bench(challengers, ["large-1", "large-2", "large-3", "large-4"] )

eval.write( "\n## Solo - Increasing Large 4\n" )
eval.increasingTeam( challengers, "large-4" )


