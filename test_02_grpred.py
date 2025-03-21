import evaltool, localtests

from grpred.team import bots

name= 'red'
evaltool= evaltool.Eval( f"{name}-miror-test", 10 )
challengers= { f"{name}-0": (bots, 0) }

def test_solo_bot_0():
    global name, evaltool, challengers
    print( challengers )
    challengers= localtests.testBot(name, evaltool, challengers, 0)

def test_duo_medium():
    global name, evaltool, challengers
    evaltool.report( f"\n## Miror Medium :\n" )
    challengers= localtests.testDuoMedium( evaltool, challengers, f"{name}-0", f"{name}-0" )
