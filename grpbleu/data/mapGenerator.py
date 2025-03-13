import random
import json
import os
from typing import Dict, List, Tuple
from collections import deque

class MapGenerator:
    def __init__(self):
        self.existing_maps = self._load_existing_maps()
        
    def _load_existing_maps(self) -> dict:
        """Charge les maps existantes depuis le fichier JSON"""
        if os.path.exists('maps.json'):
            with open('maps.json', 'r') as f:
                return json.load(f)
        return {}

    def _generate_map_id(self, matrix: List[List[int]], num_players: int, num_robots: int, num_vips: int) -> str:
        """Génère un ID unique pour la map basé sur sa configuration complète"""
        rows, cols = len(matrix), len(matrix[0])
        # Encode les positions des murs
        wall_positions = []
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == -1:
                    wall_positions.append(f"{i},{j}")
        
        # Crée une chaîne unique pour les murs pour s'assurer de générer que de nouvelles maps + apprentissage des meilleurs stratégies selon la map 
        walls_str = "_w" + "-".join(wall_positions) if wall_positions else ""
        
        return f"map_{rows}x{cols}_p{num_players}r{num_robots}v{num_vips}{walls_str}"

    def _is_valid_position(self, matrix: List[List[int]], row: int, col: int) -> bool:
        """Vérifie si une position est valide dans la matrice"""
        return 0 <= row < len(matrix) and 0 <= col < len(matrix[0]) and matrix[row][col] == 0

    def _calculate_player_limits(self, available_space: int) -> Tuple[int, int, int]:
        """Calcule les limites de joueurs/robots/VIP en fonction de l'espace disponible"""
        # Maximum 2 joueurs, mais réduit si pas assez d'espace
        max_players = min(2, available_space // 4)
        
        # Maximum 3 robots
        remaining_space = available_space - (max_players * 4)
        max_robots = min(3, remaining_space // 4)
        
        if available_space < 16:
            max_robots = min(max_robots, 1)
        
        # VIP: 0 à 1 selon l'espace restant
        remaining_space -= (max_robots * 4)
        max_vips = min(1, remaining_space // 4)
    
        return max_players, max_robots, max_vips

    def _is_connected(self, matrix: List[List[int]]) -> bool:
        """Vérifie si toutes les cases accessibles sont connectées en utilisant BFS"""
        if not matrix or not matrix[0]:
            return False

        rows, cols = len(matrix), len(matrix[0])
        visited = [[False] * cols for _ in range(rows)]
        
        # Trouve la première case accessible
        start = None
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    start = (i, j)
                    break
            if start:
                break
        
        if not start:
            return False

        queue = deque([start])
        visited[start[0]][start[1]] = True
        accessible_count = 1

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while queue:
            current_row, current_col = queue.popleft()
            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                if (self._is_valid_position(matrix, new_row, new_col) and 
                    not visited[new_row][new_col]):
                    queue.append((new_row, new_col))
                    visited[new_row][new_col] = True
                    accessible_count += 1

        return accessible_count == sum(row.count(0) for row in matrix)

    def _generate_missions(self, rows: int, cols: int, num_missions: int) -> List[List[int]]:
        """Génère des missions valides dans les limites de la map"""
        missions = []
        for _ in range(num_missions):
            x = random.randint(1, cols)
            y = random.randint(1, rows)
            missions.append([x, y])
        return missions

    def generate_random_map(self, rows: int, cols: int, num_players: int, 
                          num_robots: int, num_vips: int) -> dict:
        """Génère une map aléatoire avec les paramètres spécifiés"""
        max_attempts = 10  # Nombre maximum de tentatives pour générer une map valide
        
        for _ in range(max_attempts):
            matrix = [[0 for _ in range(cols)] for _ in range(rows)]
            
            # Ajuste la densité des murs selon la taille
            min_density = max(0.15, 0.25 - (rows * cols) / 400)
            max_density = min(0.35, 0.4 - (rows * cols) / 400)
            
            wall_density = random.uniform(min_density, max_density)
            walls_to_add = int(rows * cols * wall_density)
            
            # Distribution des murs
            if random.random() < 0.6:  # 60% chance de pattern structuré
                for _ in range(walls_to_add):
                    if random.random() < 0.7:
                        r = random.randint(0, rows-1)
                        c = random.randint(0, cols-1)
                        if matrix[r][c] == 0:
                            matrix[r][c] = -1
                            directions = [(0,1), (1,0), (0,-1), (-1,0)]
                            for dr, dc in random.sample(directions, 2):
                                new_r, new_c = r + dr, c + dc
                                if (0 <= new_r < rows and 0 <= new_c < cols and 
                                    matrix[new_r][new_c] == 0 and random.random() < 0.4):
                                    matrix[new_r][new_c] = -1
                    else:
                        r = random.randint(0, rows-1)
                        c = random.randint(0, cols-1)
                        matrix[r][c] = -1
            else:
                for _ in range(walls_to_add):
                    r = random.randint(0, rows-1)
                    c = random.randint(0, cols-1)
                    matrix[r][c] = -1

            if not self._is_connected(matrix):
                continue

            map_id = self._generate_map_id(matrix, num_players, num_robots, num_vips)
            
            if map_id in self.existing_maps:
                continue

            missions = self._generate_missions(rows, cols, random.randint(3, 6))
            
            map_object = {
                map_id: {
                    "matrix": matrix,
                    "numberOfPlayers": num_players,
                    "numberOfRobots": num_robots,
                    "numberOfVips": num_vips,
                    "tic": random.randint(100, 1000),
                    "missions": missions
                }
            }
            
            return map_object
        
        return None 

    def generate_all_configurations(self, min_size: int = 3, max_size: int = 10) -> List[dict]:
        """Génère toutes les configurations possibles pour différentes tailles de maps"""
        configurations = []
        
        for rows in range(min_size, max_size + 1):
            for cols in range(min_size, max_size + 1):
                if rows * cols > 100:  # Limite pour les maps 10x10
                    continue
                    
                available_space = rows * cols
                if available_space < 9:
                    continue
                    
                max_players, max_robots, max_vips = self._calculate_player_limits(available_space)
                
                # Augmente le nombre de variations pour plus de diversité
                num_variations = max(2, min(8, (rows * cols) // 15))
                
                for variation in range(num_variations):
                    for players in range(1, max_players + 1):
                        for robots in range(1, max_robots + 1):
                            for vips in range(max_vips + 1):
                                new_map = self.generate_random_map(rows, cols, players, robots, vips)
                                if new_map:
                                    configurations.append(new_map)
        
        return configurations

    def save_maps(self, maps: List[dict]):
        """Sauvegarde les nouvelles maps dans le fichier JSON"""
        for map_obj in maps:
            self.existing_maps.update(map_obj)
            
        with open('./data/maps.json', 'w') as f:
            json.dump(self.existing_maps, f, indent=4)

if __name__ == "__main__":
    generator = MapGenerator()
    
    # Génère des configurations pour des maps de 3x3 à 7x7 (20x20 non exploitable pour le moment) 
    configurations = generator.generate_all_configurations(3, 7)
    
    # Sauvegarde les nouvelles maps
    generator.save_maps(configurations)
    
    print(f"Nombre total de configurations générées : {len(configurations)}")
    print("Maps sauvegardées dans maps.json")