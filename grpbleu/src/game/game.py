from hacka.games.moveit import GameEngine

from ..player import Player
from ..robot import VIP
from ..utils import Display
from ..mission import MissionManager
from ..path.path import Path

class Game:
    def __init__(self):
        self._model = GameEngine()
        # _mission_manager: assigns mission to robots
        self._mission_manager = MissionManager()
        # _players: List of players having robots
        self._players = []
        self._vip = VIP()
        # _distances: 2D array containing the cost (in number of cells) for all possible distances from point A to point B
        self._distances = None
        self._visited_tiles = set()

    # TODO: commentaire Elouan
    def buildModel(self, matrix, numberOfPlayers, numberOfRobots,numberOfVips,tic,missions):
        self._model= GameEngine(
                matrix=matrix,
                numberOfPlayers=numberOfPlayers,
                numberOfRobots=numberOfRobots,
                numberOfPVips=numberOfVips,
                tic=tic,
                missions=missions
            )

    # Model ------------------------------------------------------------------------------------------------------------
    def initModel(self, gameConfiguration):
        self._model.fromPod(gameConfiguration)

    def getModel(self):
        return self._model

    # Vip --------------------------------------------------------------------------------------------------------------
    def initVIP(self):
        if self._vip.checkVip(self._model):
            self._vip.setVipPosition(self._model.mobilePosition(0, 0))

    def getVIP(self):
        return self._vip

    # Player -----------------------------------------------------------------------------------------------------------
    def initPlayers(self, numberOfPlayer, playerId):
        """
            Initialize the list of players
            For this game, a maximum of 2 players is allowed
        """
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

    def getPlayers(self):
        return self._players

    def getPlayer(self, id):
        # id must be 1 or 2 (for the setting of 2 players)
        return self._players[id - 1]

    # Robots -----------------------------------------------------------------------------------------------------------
    def initRobots(self):
        for player in self._players:
            player.initRobots(self._model)

    def initRobotPriority(self, numberOfRobotsInGame):
        """" Init the priority of each robot depending on the number of robots in game """
        for player in self._players:
            for robot in player.getRobots():
                robot.initPriority(numberOfRobotsInGame)

    def getAllRobots(self):
        """ Retrieve all robots in the game, across all players """
        robots = []
        for player in self._players:
            for robot in player.getRobots():
                robots.append(robot)

        return robots

    # Mission Manager --------------------------------------------------------------------------------------------------
    def getMissionManager(self):
        return self._mission_manager

    # Distances --------------------------------------------------------------------------------------------------------
    def getDistances(self):
        return self._distances

    def getDistance(self, x, y=None):
        if y is None:
            return self._distances[x]
        return self._distances[x][y]

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

    # Visited Tiles ----------------------------------------------------------------------------------------------------
    def getVisitedTiles(self):
        return self._visited_tiles

    def addTile(self, robot_position):
        self._visited_tiles.add(robot_position)

    # Path -------------------------------------------------------------------------------------------------------------
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

    # Robot Priorities -------------------------------------------------------------------------------------------------
    def updateRobotBlockerAndPriorities(self):
        """
            Update the priority and the block robots list of each given robot
            (robots passed as parameters, including robots from the other player)
        """
        robots = self.getAllRobots()
        self.resetRobotPriorities(robots)
        self.updateBlockRobotsAndCommonCells(robots)
        self.updatePriorities(robots)

    def updatePriorities(self, robots):
        """
            Update the priorities of each robot

            The priority is computed based on the distance to common cells with the blocking robots
            The robot with the shortest distance to the common cells gets the highest priority
            After that, for robots with the same priority, the priority is updated based on the score of each robot's selected mission
        """
        # Update priorities depending on the shortest distance between the position of the robot and the common cell
        for robot in robots:
            block_robots = robot.getBlockRobots()
            if len(block_robots) > 0:
                common_cells = robot.getCommonCells()
                for common_cells_robot, block_robot in zip(common_cells, block_robots):
                    for common_cell in common_cells_robot:
                        distance_robot = self._distances[robot.getPosition()][common_cell]
                        distance_block_robot = self._distances[block_robot.getPosition()][common_cell]
                        if distance_robot > distance_block_robot:
                            robot.setPriority(block_robot.getId() - 1, False)

        # Update priorities based on the score of the selected mission for robots with the same priority value.
        for robot_a in robots:
            for robot_b in robots:
                if robot_a.getId() != robot_b.getId():
                    if robot_a.getSumPriority() == robot_b.getSumPriority():
                        # TODO: Area for improvement: Consider the remaining number of turns and the number of cells
                        #  to traverse to reach the mission, as these factors can influence the priority choice.
                        if robot_a.getScoreSelectedMission() < robot_b.getScoreSelectedMission():
                            robot_a.setPriority(robot_b.getId() - 1, False)

    def resetRobotPriorities(self, robots):
        for robot in robots:
            robot.resetPriority()
            robot.resetBlockRobots()
            robot.resetCommonCells()

    def updateBlockRobotsAndCommonCells(self, robots):
        """ Update/add robots found along the path of another robot """
        for i in range(len(robots)):
            for j in range(len(robots)):
                if i != j:
                    robot_a = robots[i]
                    robot_b = robots[j]
                    path_a = robot_a.getPath()
                    path_b = robot_b.getPath()
                    common_cell = set(path_a) & set(path_b)
                    if common_cell:
                        robot_a.addBlockRobot(robot_b)
                        robot_a.addCommonCells(common_cell)

