import evaltool, localtests

from grpbleu.src.team import bots

name= 'bleu'
evaltool= evaltool.Eval( name, 10, "test-miror" )
challengers= { f"{name}-0": (bots, 0) }

def test_solo_bot_0():
    global name, evaltool, challengers
    print( challengers )
    challengers= localtests.testBot(name, evaltool, challengers, 0)

def test_duo_medium_1robot():
    global name, evaltool, challengers
    evaltool.report( f"\n## Miror Medium (1 robot) :\n" )
    for botName in challengers:
        evaltool.testConfront( challengers, botName, botName, "medium-31", 3, 21.0 )
        evaltool.testConfront( challengers, botName, botName, "medium-31", 10, 21.0 )

def test_duo_medium():
    global name, evaltool, challengers
    evaltool.report( f"\n## Miror Medium :\n" )
    for botName in challengers:
        localtests.testDuoMedium( evaltool, challengers, botName, botName )

def test_duo_large():
    global name, evaltool, challengers
    evaltool.report( f"\n## Miror Large :\n" )
    for botName in challengers:
        localtests.testDuoLarge( evaltool, challengers, botName, botName )

def test_duo_medium_vip():
    global name, evaltool, challengers
    evaltool.setVip(1)
    evaltool.report( f"\n## Miror Medium (ViP)\n" )
    for botName in challengers:
        localtests.testDuoMedium( evaltool, challengers, botName, botName )

def test_duo_large_vip():
    global name, evaltool, challengers
    evaltool.setVip(1)
    evaltool.report( f"\n## Miror Large (ViP)\n" )
    for botName in challengers:
        localtests.testDuoLarge( evaltool, challengers, botName, botName )
