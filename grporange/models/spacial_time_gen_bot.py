from hacka.games.moveit import GameEngine
from collections import deque
import time
import os
import csv

class SpacialTimeGenBot():
    def __init__(self, history_length=5, render=False):
        self.history_length = history_length
        self.render = render

    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self.player_id = playerId
        self.game_engine = GameEngine()
        self.game_engine.fromPod(gameConfiguration)  # Load the model from gameConfiguration

        if self.game_engine._map._size < 15:
            self.map_size = "small"
        elif self.game_engine._map._size < 40:
            self.map_size = "medium"
        else:
            self.map_size = "large"
        
        if self.render:
            self.game_engine.render()

        self.previous_position = -1
        self.move_history = deque([-1] * self.history_length, maxlen=self.history_length)
        self.training_data = []

    def perceive(self, gameState):
        self.game_engine.setOnState(gameState)
        if self.render:
            self.game_engine.render()
            time.sleep(0.2)

    def decide(self):
        current_position = self.game_engine.mobilePosition(0, 1)
        sub_matrice = self.get_area_around(current_position)
        
        if self.previous_position >= 0:
            possible_moves = self.game_engine._map.clockBearing(self.previous_position)
            adjacent_tiles = self.game_engine._map.neighbours(self.previous_position)

            for move, tile in zip(possible_moves, adjacent_tiles):
                if tile == current_position:
                    aplatie = [element for ligne in sub_matrice for element in ligne]
                    movement_sequence = [str(move) for move in self.move_history]  # Convert history to strings
                    movement_sequence.append(str(move))  # Append current move
                    aplatie.extend(movement_sequence)
                    self.training_data.append(aplatie)  # Store training data
                    self.move_history.append(move)  # Update history
                    break
        
        self.previous_position = current_position
        return ""

    def sleep(self, result):
        csv_file_path = f"./data-map-{self.map_size}-history{self.history_length}.csv"
        file_exists = os.path.exists(csv_file_path)

        with open(csv_file_path, "a", newline="") as file:
            writer = csv.writer(file)

            # Add header if the file is new
            if not file_exists:
                header = [f"tile_{i}" for i in range(9)] + [f"move_{i}" for i in range(self.history_length)] + ["next_move"]
                writer.writerow(header)

            # Write recorded training data
            for movement_sequence in self.training_data:
                writer.writerow(movement_sequence)

        print(f"End of session: {result}")
    
    def get_area_around(self, i_tile):
        tile = self.game_engine._map.tile(i_tile)
        sub_matrice = [[0 for _ in range(3)] for _ in range(3)]
        adjencies = tile.adjacencies()
        directions = self.game_engine._map.clockBearing(i_tile)
        if 12 not in directions or 9 not in directions or 3 not in directions or 6 not in directions:
            if 12 not in directions:
                sub_matrice[0][1] = -1
            else:
                top_i = adjencies[directions.index(12)]
                top_directions = self.game_engine._map.clockBearing(top_i)
                if 9 not in top_directions:
                    sub_matrice[0][0] = -1
                if 3 not in top_directions:
                    sub_matrice[0][2] = 0
            
            if 9 not in directions:
                sub_matrice[1][0] = -1
            else:
                left_i = adjencies[directions.index(9)]
                left_directions = self.game_engine._map.clockBearing(left_i)
                if 12 not in left_directions:
                    sub_matrice[0][0] = -1
                if 6 not in left_directions:
                    sub_matrice[2][0] = -1
            
            if 3 not in directions:
                sub_matrice[1][2] = -1
            else:
                right_i = adjencies[directions.index(3)]
                right_directions = self.game_engine._map.clockBearing(right_i)
                if 12 not in right_directions:
                    sub_matrice[0][2] = -1
                if 6 not in right_directions:
                    sub_matrice[2][2] = -1
            
            if 6 not in directions:
                sub_matrice[2][1] = -1
            else:
                bot_i = adjencies[directions.index(6)]
                bot_directions = self.game_engine._map.clockBearing(bot_i)
                if 9 not in bot_directions:
                    sub_matrice[2][0] = -1
                if 3 not in bot_directions:
                    sub_matrice[2][2] = -1
            
            if 12 not in directions and 3 not in directions:
                sub_matrice[0][2] = -1
            if 12 not in directions and 9 not in directions:
                sub_matrice[0][0] = -1
            if 6 not in directions and 3 not in directions:
                sub_matrice[2][2] = -1
            if 6 not in directions and 9 not in directions:
                sub_matrice[2][0] = -1

        return sub_matrice
