import unittest
import hacka.games.moveit as moveit
from bot import MultiBot

class TestMissionMultiPlayers(unittest.TestCase):

    def setUp(self):
        """
        Initialise un bot avec un modèle de test avant chaque test.
        """
         # Initialisation du moteur du jeu
        self.gameEngine = moveit.GameEngine(
            matrix= [
                [00, 00, 00, -1, 00, 00, 00, 00, 00, 00],
                [00, -1, 00, 00, 00, -1, 00, -1, -1, 00],
                [00, 00, 00, -1, 00, 00, 00, -1, 00, 00],
                [00, 00, 00, -1, 00, 00, 00, 00, 00, 00],
                [00, -1, 00, 00, 00, -1, 00, -1, -1, -1],
                [00, -1, 00, -1, 00, 00, 00, -1, -1, -1],
                [00, 00, 00, 00, 00, -1, 00, -1, -1, -1]
            ],
            tic=20,
            numberOfPlayers=2,
            numberOfRobots=2,
        )

        # Initialisation du bot player 1
        self.player = MultiBot()
        self.player._id = 1
        self.player._model = self.gameEngine
        self.player._model.render()

        # Initialisation du bot player 2
        self.player2 = MultiBot()
        self.player2._id = 2
        self.player2._model = self.gameEngine
        self.player2._model.render()

        self.player.initGame()
        self.player2.initGame()
        

    def perceive(self):
        self.player._free_missions = self.player._model.freeMissions()
        self.player._enemyPositions = self.player.getEnemyPosition()
        self.player2._free_missions = self.player._model.freeMissions()
        self.player2._enemyPositions = self.player.getEnemyPosition()

    def test_single_mission(self):
        """ Test avec une seule mission disponible. """
        assert self.player._id == 1
        assert self.player2._id == 2
        assert self.player._model == self.gameEngine
        assert self.player2._model == self.gameEngine
        assert self.player._model.addMission(11, 13, 10) == 1
        assert len(self.player._model.freeMissions()) == 1
        assert len(self.player2._model.freeMissions()) == 1
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 1, "Le bot devrait choisir la seule mission disponible")

    def test_multiple_missions_deux_players(self):
        """ Test avec plusieurs missions disponibles. """
        assert self.gameEngine.addMission(11, 13, 10) == 1  # Récompense de 10
        assert self.gameEngine.addMission(3, 5, 15) == 2 # Récompense de 15
        assert self.gameEngine.addMission(4, 8, 5) == 3  # Récompense de 5
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 2, "Le bot du premier joueur doit prendre la deuxième mission")

    def test_multiple_missions_deux_players_bot2(self):
        """ Test avec plusieurs missions disponibles. """
        assert self.gameEngine.addMission(11, 13, 10) == 1  # Récompense de 10
        assert self.gameEngine.addMission(3, 5, 15) == 2 # Récompense de 15
        assert self.gameEngine.addMission(4, 8, 15) == 3  # Récompense de 5
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=2)
        self.assertEqual(best_mission, 1, "Le bot du premier joueur doit prendre la première mission")

    def test_multiple_missions_deux_bots(self):
        """ Test avec plusieurs missions disponibles. """
        assert self.gameEngine.addMission(11, 13, 10) == 1 # Récompense de 10
        assert self.gameEngine.addMission(3, 5, 15) == 2 # Récompense de 15
        assert self.gameEngine.addMission(4, 8, 5) == 3  # Récompense de 5
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player2._id, id_bot=1)
        self.assertEqual(best_mission, 2, "Le premier bot du joueur 2 doit prendre la mission sur laquelle il est")

    def test_multiple_missions_deux_bots_v2(self):
        """ Test avec plusieurs missions disponibles. """
        assert self.gameEngine.addMission(11, 13, 10) == 1 # Récompense de 10
        assert self.gameEngine.addMission(3, 5, 15) == 2 # Récompense de 15 (meilleure)
        assert self.gameEngine.addMission(5, 8, 5) == 3  # Récompense de 5
        self.perceive()
        assert self.player2._id == 2
        best_mission = self.player.minDistanceToMission(id_player=self.player2._id, id_bot=2)
        self.assertEqual(best_mission, 3, "Le bot devrait choisir la troisième mission")
    
if __name__ == '__main__':
    unittest.main()
