import unittest
import hacka.games.moveit as moveit
from bot import MultiBot

class TestMissionSelection(unittest.TestCase):

    def setUp(self):
        """
        Initialise un bot avec un modèle de test avant chaque test.
        """
         # Initialisation du moteur du jeu
        self.gameEngine = moveit.GameEngine(
            matrix=[
                [-1, 0, -1,  0, -1],
                [ 0, 0,  0,  0,  0],
                [ 0,-1,  0, -1,  0],
                [ 0, 0,  0,  0,  0],
                [ 0, 0, -1,  0,  0],
                [ 0, 0,  0,  0,  0]
            ],
            tic=20,
            numberOfPlayers=1,
            numberOfRobots=2,
        )

        # Initialisation du bot player 1
        self.player = MultiBot()
        self.player._id = 1
        self.player._model = self.gameEngine
        self.player._model.render()
        self.player.initGame()
        
    def perceive(self):
        self.player._free_missions = self.player._model.freeMissions()
        self.player._enemyPositions = self.player.getEnemyPosition()

    def test_single_mission(self):
        """ Test avec une seule mission disponible. """
        
        assert self.player._id == 1
        assert self.player._model == self.gameEngine
        assert self.player._model.addMission(11, 13, 10) == 1
        assert len(self.player._model.freeMissions()) == 1
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 1, "Le bot devrait choisir la seule mission disponible")

    def test_equal_distance_different_rewards(self):
        """ Test avec 2 missions ayant la même distance mais différentes récompenses. """
        assert self.gameEngine.addMission(3, 11, 15)  # Récompense de 15 (meilleure)
        assert self.gameEngine.addMission(5, 13, 5)   # Récompense de 5
        assert len(self.player._model.freeMissions()) == 2
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 1, "Le bot devrait choisir la mission avec la meilleure récompense")

    def test_equal_distance_equal_rewards(self):
        """ Test avec 2 missions ayant la même distance mais différentes récompenses. """
        assert self.gameEngine.addMission(3, 11, 15) == 1 # Récompense de 15 (meilleure)
        assert self.gameEngine.addMission(5, 13, 15) == 2 # Récompense de 15
        self.perceive()
        assert len(self.player._model.freeMissions()) == 2
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 1, "Le bot devrait choisir la mission avec la meilleure récompense")

    def test_total_distance_different(self):
        """ Test avec 2 missions ayant la même distance mais différentes récompenses. """
        assert self.gameEngine.addMission(3, 16, 15) == 1 # Récompense de 15 (meilleure)
        assert self.gameEngine.addMission(5, 13, 15) == 2 # Récompense de 15  
        self.perceive()
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 2, "Le bot devrait choisir la mission avec la meilleure récompense")

    def test_different_ratios(self):
        """ Test avec 2 missions ayant des ratios reward/distance différents. """
        assert self.gameEngine.addMission(9, 11, 30) == 1 # Récompense de 15 (meilleure) 30/6 (3+3)
        assert self.gameEngine.addMission(10, 24, 50) == 2  # Récompense de 5 50/8 (5+3)  
        
        self.perceive() # Distance 3 pour mission 1, distance 6 pour mission 2

        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 2, "Le bot devrait choisir la mission avec le meilleur ratio reward/distance")

    def test_high_reward_far_away(self):
        """ Test avec une mission très lointaine mais avec une grosse récompense. """
        assert self.gameEngine.addMission(5, 11, 10) == 1 # Récompense de 15 (meilleure)
        assert self.gameEngine.addMission(14, 20, 80) == 2 # Récompense de 5

        self.perceive() # Distance 2 pour mission 1, distance 10 pour mission 2
    
        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 2, "Le bot devrait choisir la mission plus rentable malgré la distance")

    def test_mission_already_on_site(self):
        """ Test si le bot choisit une mission où il est déjà présent. """
        assert self.gameEngine.addMission(1, 11, 40) == 1
        assert self.gameEngine.addMission(3, 20, 50) == 2
        
        self.perceive() # Déjà sur place pour la mission 1

        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 1, "Le bot devrait prioriser la mission déjà sur place")

    def test_multiple_missions(self):
        """ Test avec plusieurs missions disponibles. """
        
        assert self.gameEngine.addMission(11, 13, 10) == 1  # Récompense de 10
        assert self.gameEngine.addMission(12, 5, 15) == 2 # Récompense de 15 (meilleure)
        assert self.gameEngine.addMission(4, 8, 5) == 3 # Récompense de 5
        self.player._free_missions = self.player._model.freeMissions()

        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertEqual(best_mission, 2, "Le bot devrait choisir la mission avec le meilleur ratio reward/distance")

    def test_no_mission_available(self):
        """ Test si aucune mission n'est disponible. """
        self.player._free_missions = set()  # Aucune mission disponible

        best_mission = self.player.minDistanceToMission(id_player=self.player._id, id_bot=1)
        self.assertIsNone(best_mission, "Le bot ne devrait choisir aucune mission si rien n'est disponible")
    
if __name__ == '__main__':
    unittest.main()
