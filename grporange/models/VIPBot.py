from hacka.games.moveit import GameEngine
from hacka import AbsPlayer
from models.VIPMoveTracker import VIPMoveTracker

class VIPBot(AbsPlayer):
    def __init__(self):
        super().__init__()
        self._model = GameEngine()
        self._id = 0
        self._vip_tracker = VIPMoveTracker()
        self._prev_position = None
        self._other_robots_prev = {}
    def wakeUp(self, playerId, numberOfPlayers, gamePod):
        self._id = playerId
        self._model.fromPod(gamePod)
        self._model.render()
        self._prev_position = self._model.mobilePosition(self._id, 1)
        self._other_robots_prev = {}
    def perceive(self, state):
        self._model.setOnState(state)
        self._model.render()
        current_position = self._model.mobilePosition(self._id, 1)
        other_positions = {}
        for player in range(1, self._model.numberOfPlayers()+1):
            if player == self._id:
                continue
            num = self._model.numberOfMobiles(player)
            for robot_id in range(1, num+1):
                other_positions[(player, robot_id)] = self._model.mobilePosition(player, robot_id)
        self._vip_tracker.update(self._prev_position, current_position, self._model.map(), self._other_robots_prev, other_positions)
        self._prev_position = current_position
        self._other_robots_prev = other_positions
    def decide(self):
        current_position = self._model.mobilePosition(self._id, 1)
        distribution = self._vip_tracker.predict_next_move_distribution(current_position, self._model.map())
        best_move = max(distribution, key=distribution.get)
        command = f"move 1 {best_move}"
        return command
    def sleep(self, result):
        print(f"VIPBot: Game finished with result: {result}")