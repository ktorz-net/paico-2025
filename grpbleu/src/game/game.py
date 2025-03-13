from hacka.games.moveit import GameEngine

from ..player import Player
from ..robot import VIP
from ..utils import Display
from ..mission import MissionManager
from ..path.path import Path

class Game:
    def __init__(self):
        self._model = GameEngine()
        self.missionManager = MissionManager()
        self._players = []
        self._vip = VIP()

        self._distances = None
        self._visited_tiles = set()

    def buildModel(self, matrix, numberOfPlayers, numberOfRobots,numberOfVips,tic,missions):
        self._model= GameEngine(
                matrix=matrix,
                numberOfPlayers=numberOfPlayers,
                numberOfRobots=numberOfRobots,
                numberOfPVips=numberOfVips,
                tic=tic,
                missions=missions
            )

    def initModel(self, gameConfiguration):
        self._model.fromPod(gameConfiguration)

    def initVIP(self):
        if self._vip.checkVip(self._model):
            self._vip.setVipPosition(self._model.mobilePosition(0, 0))

    def initPlayers(self, numberOfPlayer, playerId):
        player1 = Player(playerId)
        if numberOfPlayer > 1:
            opponentId = playerId % 2 + 1
            player2 = Player(opponentId)
            if player1.getId() == 1:
                self._players.append(player1)
                self._players.append(player2)
            else:
                self._players.append(player2)
                self._players.append(player1)
        else:
            self._players.append(player1)

    def initRobots(self):
        for player in self._players:
            player.initRobots(self._model)

    def initRobotPriority(self, numberOfRobotsInGame):
        for player in self._players:
            for robot in player.getRobots():
                robot.initPriority(numberOfRobotsInGame)


    def getAllRobots(self):
        robots = []
        for player in self._players:
            for robot in player.getRobots():
                robots.append(robot)

        return robots

    def getModel(self):
        return self._model

    def getMissionManager(self):
        return self.missionManager

    def setModel(self, model):
        self._model = model

    def getPlayer(self, id):
        # id must be 1 or 2 (for the setting of 2 players)
        return self._players[id - 1]

    def getPlayers(self):
        return self._players

    def getDistances(self):
        return self._distances

    def getDistance(self, x, y=None):
        if y is None:
            return self._distances[x]
        return self._distances[x][y]

    def setDistances(self, x):
        # TODO: à voir s'il y a un intéret à implémenter cette méthode
        pass

    def getVIP(self):
        return self._vip

    def getVisitedTiles(self):
        return self._visited_tiles

    def addTile(self, robot_position):
        self._visited_tiles.add(robot_position)
    
    def path(self, iTile, iTarget):
        """Calcule le chemin optimal entre deux tuiles

        Args:
            iTile (int): Position de départ
            iTarget (int): Position cible

        Returns:
            tuple (list[int], list[int]): Liste des mouvements et chemin
        """
        pathClass = Path(iTile, iTarget, self)
        return pathClass.findDaWay()

    def computeDistances(self, iTile):
        """Calcule les distances depuis une tuile donnée vers toutes les autres tuiles.

        Utilise un algorithme de type "flood fill" (remplissage par propagation)
        pour déterminer la distance de chaque tuile par rapport à la tuile de départ.

        Args:
            iTile (int): L'indice de la tuile de départ.

        Returns:
            list: Une liste d'entiers où l'élément à l'indice 'i' représente la distance
                  entre la tuile 'i' et la tuile 'iTile'.
        """
        # Initialise la liste des distances avec la tuile de départ et des 0 pour les autres tuiles.
        dists = [iTile] + [0 for i in range(self._model.map().size())]
        # logging.debug(f"Distances initialisées : {dists}")

        # Récupère les voisins de la tuile de départ (distance 1).
        ringNodes = self._model.map().neighbours(iTile)
        ringDistance = 1

        # Tant qu'il y a des tuiles à explorer.
        while len(ringNodes) > 0:
            nextNodes = []
            # Pour chaque tuile de la couronne actuelle.
            for node in ringNodes:
                # Met à jour la distance de la tuile courante.
                dists[node] = ringDistance
                # Récupère les voisins de la tuile courante.
                neighbours = self._model.map().neighbours(node)
                # Pour chaque voisin.
                for candidate in neighbours:
                    # Si le voisin n'a pas encore été visité.
                    if dists[candidate] == 0:
                        # Ajoute le voisin à la liste des tuiles à explorer.
                        nextNodes.append(candidate)

            # Passe à la couronne suivante.
            ringNodes = nextNodes
            ringDistance += 1

        # Corrige la distance de la tuile de départ à 0.
        dists[iTile] = 0
        return dists

    def computeAllDistances(self):
        """Précaculate toutes les distances entre les cases de la carte.

        Cette fonction calcule et stocke une matrice complète des distances
        entre chaque paire de cases accessibles sur la carte.
        """
        matrix_size = self._model.map().size()
        # Initialise la matrice des distances avec des valeurs par défaut.
        self._distances = [[i for i in range(matrix_size + 1)]]
        # Calcule les distances pour chaque tuile de la carte.
        for i in range(1, self._model.map().size() + 1):
            self._distances.append(self.computeDistances(i))

        # Display.displayDistanceMatrix(self)
