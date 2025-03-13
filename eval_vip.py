#!env python3
import json, time, matplotlib.pyplot as plt
import hacka.games.moveit as moveit

from grporange.grporange.team import bots as obots
from grpbleu.src.team import bots as bbots
from grpvert.team import bots as vbots

import evaltool

# Build the list of bots.
teams= [ obots, bbots, vbots ]
teamNames= [ "orange", "blue", "green" ]
okTeams= [ 
    [True, True, True, True],
    [False, False],
    [True, False, False]
]

eval= evaltool.Eval( "vip", 100 )
eval.setVip(1)

challengers= eval.initChallengers(
    "small-1",
    teams,
    teamNames,
    okTeams
)

eval.write( "\n## Solo - Small - VIP\n" )
eval.bench(challengers, ["small-1", "small-2", "small-3"] )

eval.write( "\n## Solo - Medium - VIP\n" )
eval.bench(challengers, ["medium-1", "medium-2", "medium-3", "medium-4"] )

eval.write( "\n## Solo - Increasing Medium 2\n" )
eval.increasingTeam( challengers, "medium-2" )

eval.write( "\n## Solo - Large - VIP\n" )
eval.bench(challengers, ["large-1", "large-2", "large-3", "large-4"] )

eval.write( "\n## Solo - Increasing Large 4\n" )
eval.increasingTeam( challengers, "large-4" )


