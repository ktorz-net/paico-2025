import hacka.games.moveit as moveit
import json
import multiplayerbot, VIPBot, MonteCarloBot, MultiDangerPathMCBot, VIPAvoidPathBot, multiplayerbotennemy

# Charger la configuration depuis un fichier JSON
with open("./config-7x10.json") as file:
    config = json.load(file)

# Configure the game:
# gameEngine= moveit.GameEngine(
#     matrix= [ 
#               [00, 00, 00, 00, 00, 00, 00, 00],
#               [00, -1, -1, -1, -1, -1, -1, 00],
#               [00, 00, 00, 00, 00, 00, 00, 00],
#               [00, -1, -1, -1, -1, -1, -1, 00],
#               [00, 00, 00, 00, 00, 00, 00, 00],
#             ],
#     numberOfPlayers=1, numberOfRobot=2, tic=100,
#     missions= [(1, 29), (16, 24), (4, 21)]
# )

n_games = 1
total_score_bot1 = 0.0
total_score_bot2 = 0.0
total_diff = 0.0

for i in range(n_games):
    # Création d'une nouvelle configuration de jeu pour chaque partie
    gameEngine = moveit.GameEngine(
        matrix=config['matrix'],
        tic=config['tic'],
        numberOfPlayers=config['numberOfPlayers'],
        numberOfRobots=config['numberOfRobots'],
        numberOfPVips=config['numberOfPVips']
    )

    # Initialisation du GameMaster avec quelques missions aléatoires
    gameMaster = moveit.GameMaster(gameEngine, randomMission=10)

    # Instanciation des bots
    # Bot 1 (par exemple, le MultiPlayerBot classique)

    # Bots pour contrôler le VIP et l'ennemi (utilisés par le MonteCarloBot)
    vip_bot = VIPBot.VIPBot()  # Bot VIP prédictif
    enemy_bot = multiplayerbot.MultiPlayerBot()  # Bot ennemi prédictif

    # Bot 2 (MonteCarloBot) qui reçoit en argument le VIPBot et le bot ennemi
    bot2 = multiplayerbot.MultiPlayerBot()
    #bot2 = MultiDangerPathMCBot.MultiDangerPathMCBot(vip_bot, enemy_bot, debug=False)
    bot1 = MonteCarloBot.PathMCBot(vip_bot, enemy_bot, debug=True)

    #Bot 3 
    bot3 = multiplayerbotennemy.MultiPlayerBotEnnemy()

    # Lancer la partie
    gameMaster.launch([bot3, bot2], gameEngine.numberOfPlayers())
    

    # Récupérer les scores finaux pour chaque bot
    # Supposons que le bot 1 a l'ID 1 et bot 2 a l'ID 2
    score_bot1 = gameEngine.score(1)
    score_bot2 = gameEngine.score(2)
    diff = score_bot2 - score_bot1

    total_score_bot1 += score_bot1
    total_score_bot2 += score_bot2
    total_diff += diff

    print(f"Game {i + 1}: Bot1 score = {score_bot1}, Bot2 score = {score_bot2}, Diff = {diff}")

print(f"----- Résultats sur {n_games} parties -----")
print("Average Bot1 score:", total_score_bot1 / n_games)
print("Average Bot2 score:", total_score_bot2 / n_games)
print("Average diff (Bot2 - Bot1):", total_diff / n_games)