from . import complete_bot
from . import tsp_bot
from . import multibot
from . import multiplayerbot

def bots(): 
    return [complete_bot.CompleteBot(), multibot.MultiBot(), tsp_bot.TspBot(), multiplayerbot.MultiPlayerBot() ]
