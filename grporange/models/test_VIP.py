import pytest
from VIPMoveTracker import VIPMoveTracker

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
        self.tiles = {}
    def completeClock(self, tile):
        if tile == 5:
            return [5, 5, 5, 8, 5, 5, 5, 5, 5, 7, 5, 5, 5]
        base = tile
        arr = [base] * 13
        arr[3] = base + 3
        arr[9] = base - 1
        return arr
    def tile(self, tile):
        if tile not in self.tiles:
            self.tiles[tile] = FakeTile(tile, FakeCenter(tile, tile))
        return self.tiles[tile]
    def clockposition(self, current_tile, direction):
        return self.completeClock(current_tile)[direction]

@pytest.fixture
def tracker():
    return VIPMoveTracker()

@pytest.fixture
def fake_map():
    return FakeMap()

def test_get_valid_cardinal_directions(tracker, fake_map):
    valid = tracker.get_valid_cardinal_directions(5, fake_map)
    assert valid == {3, 9}

def test_get_possible_moves(tracker, fake_map):
    moves = tracker.get_possible_moves(5, fake_map)
    assert moves == [0, 3, 9]

def test_get_markov_prediction(tracker):
    tracker.moves = [3, 3, 9, 3, 6]
    probs = tracker.get_markov_prediction()
    assert abs(probs[3] - 0.6) < 1e-6
    assert abs(probs[9] - 0.2) < 1e-6
    assert abs(probs[6] - 0.2) < 1e-6

def test_predict_next_move_distribution(tracker, fake_map):
    tracker.moves = [3, 3, 9, 3, 6]
    dist = tracker.predict_next_move_distribution(5, fake_map)

    assert 3 in dist
    assert 9 in dist
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-6

def test_predict_next_move(tracker, fake_map):
    tracker.moves = [3, 3, 9, 3, 6]
    pred = tracker.predict_next_move(5, fake_map)
    assert pred[0] == 0
    for move in pred[1:]:
        assert move in {3, 6, 9, 12}