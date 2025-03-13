#!env python3
from grpred.bot import GhostBot
from grpred.team import bots as rbots
from grporange.grporange.team import bots as obots
#from grpbleu.src.team import bots as bbots
from grpvert.team import bots as vbots

import evaltool

# Parramétre de l'expérience
configName= "medium-2"
maxTime= 6.0
nbOfGames= 100
vip= 0

#Prof version of bleuBot
def bbots():
    from grpbleu.src.bot import Bot
    return [ Bot(), GhostBot( Bot ) ]

# Build the list of bots.
eval= evaltool.Eval( f"solo-{configName}", nbOfGames )
eval.setVip(vip)
colorTeam= [ "red", "orange", "bleu", "vert" ]
challengers= eval.initChallengers(
    [ rbots, obots, bbots, vbots ],
    colorTeam,
    [
    [True, False],
    [True, True, True, True],
    [False, True],
    [True, False, False]
]
)

challengers= eval.testChallengers( challengers, configName, 3, maxTime )
eval.report( f"\n challengers: { list(challengers.keys()) }\n\n" )

challengers= eval.testChallengers( challengers, configName, 10, maxTime )
eval.report( f"\n challengers: { list(challengers.keys()) }\n\n" )

eval.report( f"\n## Solo - Increasing {configName}\n" )
eval.increasingTeam( challengers, configName )

for color in colorTeam :
    eval.mergeLogs(color)