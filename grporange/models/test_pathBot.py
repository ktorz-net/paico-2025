import pytest
import math
import random
from collections import deque
from MonteCarloBot import PathMCBot

class FakeCenter:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    def x(self):
        return self._x
    def y(self):
        return self._y

class FakeTile:
    def __init__(self, tile_id, center):
        self.tile_id = tile_id
        self._center = center
    def center(self):
        return self._center

class FakeMap:
    def __init__(self):
        self.size_val = 10
        self.tiles = {}
    def size(self):
        return self.size_val
    def tile(self, tile):
        if tile not in self.tiles:
            self.tiles[tile] = FakeTile(tile, FakeCenter(tile, tile))
        return self.tiles[tile]
    def completeClock(self, tile):
        if tile == 5:
            return [5,5,5,8,5,5,5,5,5,7,5,5,5]
        arr = [tile] * 13
        arr[3] = tile + 1 if tile + 1 <= self.size_val else tile
        arr[9] = tile - 1 if tile - 1 >= 1 else tile
        return arr
    def clockBearing(self, tile):
        nbrs = self.neighbours(tile)
        if len(nbrs) == 2:
            return [9, 3]
        elif len(nbrs) == 1:
            return [9] if nbrs[0] < tile else [3]
        else:
            return []
    def clockposition(self, current_tile, direction):
        if direction == 3:
            return current_tile + 1 if current_tile + 1 <= self.size_val else current_tile
        elif direction == 9:
            return current_tile - 1 if current_tile - 1 >= 1 else current_tile
        else:
            return current_tile
    def neighbours(self, tile):
        nbrs = []
        if tile - 1 >= 1:
            nbrs.append(tile - 1)
        if tile + 1 <= self.size_val:
            nbrs.append(tile + 1)
        return nbrs

class FakeMission:
    def __init__(self, start, final, reward=10, owner=0):
        self.start = start
        self.final = final
        self.reward = reward
        self.owner = owner

class FakeGameEngine:
    def __init__(self):
        self._tic = 50
        self._score = {1: 0, 0: 0, 2: 0}
        # Pour player 1, 2 robots aux positions 3 et 4; pour VIP (player 0) position 5; pour adversaire (player 2) position 7.
        self._mobiles = {1: [3, 4], 0: [5], 2: [7]}
        self._map = FakeMap()
        # Deux missions : mission 1 part de 3 vers 8, mission 2 part de 4 vers 2.
        self._missions = [FakeMission(3, 8), FakeMission(4, 2)]
    def fromPod(self, pod):
        return self
    def asPod(self):
        return {}
    def tic(self):
        return self._tic
    def score(self, player):
        return self._score.get(player, 0)
    def numberOfPlayers(self):
        return 2
    def numberOfMobiles(self, player):
        return len(self._mobiles.get(player, []))
    def mobilePosition(self, player, robot):
        return self._mobiles[player][robot - 1]
    def mobileMission(self, player, robot):
        # Pour les tests, on considère qu'aucun robot n'a de mission en cours (0).
        return 0
    def map(self):
        return self._map
    def missionsList(self):
        return list(range(1, len(self._missions) + 1))
    def mission(self, mid):
        return self._missions[mid - 1]
    def render(self):
        pass
    def setOnState(self, state):
        pass
    def applyMoveActions(self):
        pass

class FakeBot:
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId
    def decide(self):
        return "move 1 3"
    def predict_next_move_distribution(self, current_tile, map_instance):
        return {3: 0.6, 9: 0.4, 0: 0.0}

@pytest.fixture
def fake_engine():
    return FakeGameEngine()

@pytest.fixture
def fake_vip_bot():
    return FakeBot()

@pytest.fixture
def fake_enemy_bot():
    return FakeBot()

@pytest.fixture
def path_mc_bot(fake_vip_bot, fake_enemy_bot):
    bot = PathMCBot(fake_vip_bot, fake_enemy_bot, debug=True)
    bot._model = FakeGameEngine()
    bot._id = 1
    return bot

def test_available_moves(path_mc_bot):
    engine = path_mc_bot._model
    moves = path_mc_bot.available_moves(engine, 1, 1)
    expected = {"move 1 9", "move 1 3"}
    assert set(moves).intersection(expected) == expected

def test_bfs_path(path_mc_bot):
    engine = path_mc_bot._model
    path = path_mc_bot.bfs_path(3, 8)
    assert path is not None
    assert path[0] == 3 and path[-1] == 8

def test_next_direction(path_mc_bot):
    direction = path_mc_bot.next_direction(3, 4)
    assert direction == 3

def test_basic_path_decision(path_mc_bot):
    decision = path_mc_bot.basic_path_decision(1, blocked_tiles=set())
    assert decision.startswith("mission") or decision.endswith("0")

def test_predict_vip_distribution(path_mc_bot):
    vip_pos = path_mc_bot._model.mobilePosition(0, 1)
    dist = path_mc_bot.predict_vip_distribution(vip_pos)
    assert vip_pos in dist
    assert any(tile != vip_pos for tile in dist)

def test_predict_adversary_positions(path_mc_bot):
    adv = path_mc_bot.predict_adversary_positions()
    assert len(adv) > 0

def test_decide(path_mc_bot):
    decision = path_mc_bot.decide()
    assert decision != ""
    assert decision.startswith("move") or decision.startswith("mission")