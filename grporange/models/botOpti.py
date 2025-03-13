from collections import deque

class MultiPlayerBotEnnemy:
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration, width=5, height=3):
        # Dimensions du terrain de jeu
        self.width = width
        self.height = height
        # Grille du jeu : 0 = libre, 1 = occupé (mur/trace de bot)
        self.grid = [[0 for _ in range(height)] for _ in range(width)]
        # Tableau du nombre de voisins libres pour chaque case (pour détection de tunnels)
        self.voisins_libres = [[0 for _ in range(height)] for _ in range(width)]
        # Initialisation du tableau voisins_libres
        for x in range(width):
            for y in range(height):
                # Compter les voisins initiaux (dans la grille vide au départ)
                count = 0
                if x > 0: count += 1
                if x < width - 1: count += 1
                if y > 0: count += 1
                if y < height - 1: count += 1
                self.voisins_libres[x][y] = count
        # État du bot
        self.my_position = None
        self.enemy_position = None
        # Mode du bot (normal ou en résolution de deadlock)
        self.en_deadlock = False
        self.plan_actions = []  # actions planifiées par DeadlockSolver le cas échéant

    def maj_positions(self, my_pos, enemy_pos):
        """Met à jour la position du bot et de l'ennemi, et marque l’ancienne position du bot comme occupée."""
        if self.my_position:
            # Marquer l'ancienne position du bot comme occupée dans la grille
            old_x, old_y = self.my_position
            self.grid[old_x][old_y] = 1
            # Mettre à jour les voisins libres autour de l'ancienne position du bot
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = old_x + dx, old_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Décrémenter le voisin libre de la case voisine si elle n'est pas déjà un mur
                    if self.grid[nx][ny] == 0:
                        self.voisins_libres[nx][ny] -= 1
            # Marquer aussi la position de l'ennemi précédente comme occupée
            if self.enemy_position:
                ex, ey = self.enemy_position
                self.grid[ex][ey] = 1
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[nx][ny] == 0:
                            self.voisins_libres[nx][ny] -= 1
        # Mettre à jour les nouvelles positions
        self.my_position = my_pos
        self.enemy_position = enemy_pos

    def est_deplacement_valide(self, x, y):
        """Vérifie si la case (x,y) est libre et dans les limites de la grille."""
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[x][y] == 0

    def get_voisins(self, x, y):
        """Retourne la liste des cases voisines libres de (x,y)."""
        voisins = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if self.est_deplacement_valide(nx, ny):
                voisins.append((nx, ny))
        return voisins

    def detecter_deadlock(self):
        """
        Détecte si le bot et l'ennemi sont dans une situation de blocage (zones séparées ou face-à-face dans un couloir).
        Retourne True si un deadlock est détecté.
        """
        # 1. Vérifier si les deux joueurs sont isolés dans des régions différentes (plus de chemin entre eux).
        # On fait un BFS depuis la position du bot et on voit si on peut atteindre l'ennemi.
        visited = set()
        queue = deque([self.my_position])
        visited.add(self.my_position)
        reachable_enemy = False
        while queue:
            cx, cy = queue.popleft()
            # Si on atteint la position de l'ennemi, ils sont dans la même région (pas de deadlock territorial)
            if (cx, cy) == self.enemy_position:
                reachable_enemy = True
                break
            for nx, ny in self.get_voisins(cx, cy):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        if not reachable_enemy:
            # Ennemi non atteignable : chacun est confiné dans sa région -> deadlock territorial
            return True

        # 2. Vérifier un face-à-face dans un couloir étroit.
        # Si les deux joueurs sont adjacents ou très proches dans un corridor (peu de voisins libres autour), c'est un cas de blocage potentiel.
        mx, my = self.my_position
        ex, ey = self.enemy_position
        # Distance de Manhattan
        manhattan_dist = abs(mx - ex) + abs(my - ey)
        if manhattan_dist <= 2:
            # S'ils sont très proches, vérifier le contexte de couloir
            # Un couloir étroit se caractérise par un faible nombre de voisins libres autour des deux positions
            if self.voisins_libres[mx][my] <= 1 and self.voisins_libres[ex][ey] <= 1:
                return True

        return False

    def est_tunnel(self, x, y):
        """
        Détermine si la case (x, y) se trouve à l'entrée d'un tunnel (c'est-à-dire qu'elle n'a qu'un seul voisin libre accessible).
        """
        # On considère qu'une case libre est un "tunnel" si elle a exactement 1 voisin libre.
        # (Cas particulier : si c'est la case actuelle du bot, avoir 1 voisin libre signifie que le seul voisin libre est possiblement la case d'où il vient)
        return self.voisins_libres[x][y] == 1

    def calculer_espace_accessible(self, start):
        """
        Calcule le nombre de cases libres accessibles à partir de la position start (BFS).
        Utile pour estimer la taille de la région disponible.
        """
        visited = set([start])
        queue = deque([start])
        count = 0
        while queue:
            cx, cy = queue.popleft()
            count += 1
            for nx, ny in self.get_voisins(cx, cy):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return count

    def prochaine_action(self):
        """
        Détermine la prochaine action du bot (direction du déplacement) en fonction de l'état actuel.
        """
        # Si un plan d'actions (DeadlockSolver) est en cours et toujours valable, suivre ce plan
        if self.plan_actions:
            prochaine_direction = self.plan_actions.pop(0)
            # Une fois l'action extraite du plan, on peut la retourner directement
            return prochaine_direction

        # Vérifier la situation de deadlock éventuelle
        self.en_deadlock = self.detecter_deadlock()

        if self.en_deadlock:
            # Si on est en situation de blocage, on fait appel au DeadlockSolver pour décider des actions
            self.plan_actions = self.resoudre_deadlock()
            # Après avoir obtenu un plan (liste de mouvements) du DeadlockSolver, on l'exécute dès les prochains tours.
            if self.plan_actions:
                return self.plan_actions.pop(0)
            # Si pas de plan particulier, on continue avec la logique normale comme secours.

        # Logique standard (mode normal) : évaluer les mouvements possibles et choisir le meilleur
        x, y = self.my_position
        mouvements_possibles = []  # liste de tuples (direction, score)
        # Définition des directions avec leur effet sur (x,y)
        directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}

        # Calcule une fois l'espace accessible actuel du bot et de l'ennemi pour référence
        espace_bot = self.calculer_espace_accessible(self.my_position)
        espace_ennemi = self.calculer_espace_accessible(self.enemy_position)

        for direction, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if not self.est_deplacement_valide(nx, ny):
                continue  # mouvement non valide, passer
            # Éviter de foncer dans un cul-de-sac si d'autres options existent
            if self.est_tunnel(nx, ny):
                # Si c'est un tunnel et qu'on a d'autres mouvements possibles non tunnels, on évite celui-ci
                # (On vérifie après la boucle s'il existait au moins un mouvement non-tunnel)
                mouvements_possibles.append((direction, -float('inf')))
                continue

            # Évaluer l'espace accessible si on se déplace dans cette direction
            espace_apres = self.calculer_espace_accessible((nx, ny))
            # Critère principal : maximiser l'espace accessible restant pour le bot, éventuellement en comparaison de l'ennemi
            score = espace_apres
            # On peut affiner le score en pénalisant la proximité de l'ennemi ou en favorisant l'éloignement de murs, etc.
            # Par exemple, on pénalise légèrement si l'ennemi aurait plus d'espace que nous après ce move
            if espace_ennemi > espace_apres:
                score -= 0.5  # pénalité si l'ennemi semble avoir plus de place
            mouvements_possibles.append((direction, score))

        # S’il ne reste que des tunnels en options (score -inf affecté), on les considère quand même par nécessité
        if all(score == -float('inf') for _, score in mouvements_possibles):
            # Réinitialiser les scores de tunnel à 0 pour choisir le "moins pire"
            mouvements_possibles = [(direction, 0) for direction, score in mouvements_possibles]

        # Choisir la direction avec le meilleur score
        if mouvements_possibles:
            mouvements_possibles.sort(key=lambda x: x[1], reverse=True)
            meilleure_direction = mouvements_possibles[0][0]
        else:
            # Aucune direction possible (complètement bloqué)
            meilleure_direction = 'UP'  # choix par défaut (mais en principe ne devrait pas arriver)

        return meilleure_direction

    def resoudre_deadlock(self):
        """
        Appelle le module DeadlockSolver pour obtenir un plan d'actions en situation de blocage.
        Cette fonction est un placeholder représentant l'appel à un modèle externe ou une logique avancée.
        """
        # *** Cette fonction devrait intégrer le modèle ou l'algorithme de résolution de deadlock réel. ***
        # Pour l'exemple, on simule une stratégie simple : 
        # si bloqué dans un couloir face à l'ennemi, faire demi-tour ou essayer de contourner.
        plan = []
        mx, my = self.my_position
        ex, ey = self.enemy_position
        if abs(mx - ex) <= 1 and abs(my - ey) <= 1:
            # Si l'ennemi est adjacent (face-à-face), tenter de s'écarter : 
            # on cherche un mouvement latéral qui n'est pas un tunnel si possible.
            for direction, (dx, dy) in {'UP': (0,-1),'DOWN':(0,1),'LEFT':(-1,0),'RIGHT':(1,0)}.items():
                nx, ny = mx + dx, my + dy
                if self.est_deplacement_valide(nx, ny) and not self.est_tunnel(nx, ny):
                    plan.append(direction)
                    break
        # D'autres logiques du modèle pourraient être ajoutées ici.
        return plan
