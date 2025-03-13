import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import heapq

# --- Définition du réseau DQN ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)  # output_size = 5 actions
        )

    def forward(self, x):
        return self.network(x)

# --- Agent d'apprentissage par Deep Q-Learning pour la résolution de blocages ---
class DeadlockSolver:
    def __init__(self, map_size=9, extra_features=1):
        # La taille d'état inclut désormais une feature supplémentaire (ex: compteur de répétition)
        self.state_size = map_size + 4 + extra_features  # pour map_size=9, state_size = 14
        self.action_size = 5  # 0: haut, 1: bas, 2: droite, 3: gauche, 4: rester sur place
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.episode_rewards = []  # pour suivre les récompenses
        print("Initialisation du DeadlockSolver avec state_size =", self.state_size)

    def a_star(self, grid, start, goal):
        """
        Vérifie l'existence d'un chemin entre start et goal dans une grille 3x3 en utilisant A*.
        - grid est une liste de 9 éléments (0 = obstacle, 1 = praticable)
        - start et goal sont des indices de 0 à 8.
        """
        def heuristic(a, b):
            ax, ay = a % 3, a // 3
            bx, by = b % 3, b // 3
            return abs(ax - bx) + abs(ay - by)
        
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), start))
        g_score = {i: float('inf') for i in range(9)}
        g_score[start] = 0
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == goal:
                return True
            if current in closed_set:
                continue
            closed_set.add(current)
            # Calculer les voisins (haut, bas, gauche, droite)
            row, col = current // 3, current % 3
            neighbors = []
            if row > 0: 
                neighbors.append(current - 3)
            if row < 2: 
                neighbors.append(current + 3)
            if col > 0: 
                neighbors.append(current - 1)
            if col < 2: 
                neighbors.append(current + 1)
            for neighbor in neighbors:
                if grid[neighbor] == 0:  # cellule bloquée
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return False

    def train_on_deadlocks(self, num_episodes=1000000, max_steps_per_episode=10000):
        print(f"Début de l'entraînement sur {num_episodes} épisodes")
        total_steps = 0
        for episode in range(num_episodes):
            state = self.generate_random_deadlock()
            episode_reward = 0
            done = False
            steps = 0

            while not done:
                # Stratégie epsilon-greedy
                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(self.action_size)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = torch.argmax(self.model(state_tensor)).item()

                next_state, reward, done = self.simulate_action(state, action)
                self.memory.append((state, action, reward, next_state, done))
                episode_reward += reward
                steps += 1
                total_steps += 1
                if steps >= max_steps_per_episode:
                    print(f"Épisode terminé : max steps atteint ({max_steps_per_episode})")
                    done = True

                if len(self.memory) >= 32:
                    loss = self.replay(batch_size=32)
                    if steps % 100 == 0:
                        print(f"Episode {episode}/{num_episodes}, Step {steps}, Loss: {loss:.4f}, Epsilon: {self.epsilon:.4f}")

                state = next_state
            self.episode_rewards.append(episode_reward)
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"\nEpisode {episode}/{num_episodes} terminé:")
                print(f"Récompense moyenne (100 derniers épisodes): {avg_reward:.2f}")
                print(f"Epsilon actuel: {self.epsilon:.4f}")
                print(f"Total steps: {total_steps}")
                print(f"Taille mémoire: {len(self.memory)}\n")
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate_random_deadlock(self):
        """
        Génère aléatoirement un état pour une carte 3x3.
        L'état est un vecteur de taille state_size, sous la forme :
        [pos_robot1, pos_robot2, obj_robot1, obj_robot2, case0, ..., case8, repeat_count]
        
        La grille est générée de manière aléatoire avec des obstacles,
        tout en garantissant que la case de départ (state[0]) et celle de l'objectif (state[2])
        sont praticables et qu'un chemin existe entre elles (vérifié par A*).
        """
        while True:
            state = np.zeros(self.state_size)
            # Positions aléatoires pour le robot et un second robot (utilisé ici pour information)
            state[0] = np.random.randint(0, 9)
            state[1] = np.random.randint(0, 9)
            # Définir des objectifs différents
            available = [i for i in range(9)]
            state[2] = random.choice(available)
            available.remove(state[2])
            state[3] = random.choice(available)
            
            # Génération de la grille : indices 4 à 12
            # On force la praticabilité des cases de départ (state[0]) et d'objectif (state[2])
            grid = []
            for i in range(9):
                if i == int(state[0]) or i == int(state[2]):
                    cell = 1
                else:
                    cell = np.random.choice([0, 1], p=[0.3, 0.7])
                grid.append(cell)
            
            # Intégration de la grille dans l'état
            for i in range(9):
                state[i+4] = grid[i]
            # Initialisation du compteur de répétition
            state[-1] = 0

            # Vérification qu'un chemin existe entre la position de départ et l'objectif via A*
            if self.a_star(grid, int(state[0]), int(state[2])):
                return state
            # Sinon, générer une nouvelle configuration

    def simulate_action(self, state, action):
        """
        Simule l'action sur la carte 3x3.
        Met à jour la position du robot et incrémente le compteur de répétition
        si le robot ne bouge pas.
        """
        action_mapping = {0: 12, 1: 6, 2: 3, 3: 9, 4: 0}
        move_code = action_mapping[action]

        next_state = state.copy()
        robot1_pos = int(state[0])
        goal1 = int(state[2])

        # Conversion position -> coordonnées (3x3)
        x = robot1_pos % 3
        y = robot1_pos // 3

        if move_code == 12:      # haut
            x_new, y_new = x, y - 1
        elif move_code == 6:     # bas
            x_new, y_new = x, y + 1
        elif move_code == 3:     # droite
            x_new, y_new = x + 1, y
        elif move_code == 9:     # gauche
            x_new, y_new = x - 1, y
        else:                    # rester sur place
            x_new, y_new = x, y

        if 0 <= x_new < 3 and 0 <= y_new < 3:
            new_pos = y_new * 3 + x_new
        else:
            new_pos = robot1_pos

        if state[new_pos + 4] == 1:
            next_state[0] = new_pos

        # Calcul de la récompense
        if next_state[0] == goal1:
            reward = 1
            done = True
        elif next_state[0] == state[1]:
            reward = -1
            done = True
        else:
            old_dist = abs(robot1_pos - goal1)
            new_dist = abs(next_state[0] - goal1)
            reward = old_dist - new_dist
            done = False

        # Mise à jour du compteur de répétition
        if next_state[0] == robot1_pos:
            next_state[-1] = state[-1] + 1
        else:
            next_state[-1] = 0

        return next_state, reward, done

    def get_action(self, state):
        # Avec une petite probabilité, l'agent choisit une action aléatoire (exploration)
        if np.random.rand() <= self.epsilon_min:
            action = np.random.randint(self.action_size)
            print(f"[DeadlockSolver] Action aléatoire choisie: {action} pour l'état: {state}")
            return action
        # Sinon, on utilise le modèle DL pour choisir l'action (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(self.model(state_tensor)).item()
        print(f"[DeadlockSolver] Action DL choisie: {action} pour l'état: {state}")
        return action


    def save_model(self, path='dqn_model.h5'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards
        }, path)
        print(f"Modèle sauvegardé dans {path}")

    def load_model(self, path='dqn_model.h5'):
        print(f"Chargement du modèle depuis {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.model.eval()
        print("Modèle chargé avec succès")
