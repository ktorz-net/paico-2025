# PAICO Groupe ORANGE

## IMPORTANT

Mise à jour, il faut bien récupérer les dernières sources avant de lancer la compétition.

## Introduction 
Artificial Intelligence and Combinatory Opimisation Projet repository

## Dependencies...

Pour installer les dépendances requises afin de lancer les bots : `pip install -r requirements.txt`

## Organisation du repo
```bash
.
├── README.md
├── configs             # Contient les configurations de parties
├── grporange           # Contient les livraisons
├── models              # Contient l'ensemble de nos itérations
├── launch-jawad.py     # Simple launcher
├── requirements.txt    # Contient les dépendances à télécharger
└── utils               # Utilitaires (notamment quelques launcher)
```

## Les bots livrés

|Bot|Allié|Adversaires|VIP|
|:--|:---:|:---------:|:-:|
|TspBot|❌|❌|❌|
|MultiBot|✅|❌|❌|
|MultiPlayerBot|✅|❌|❌|
|CompleteBot|✅|✅|✅|

### TspBot
#### Fonctionnement

Le `TspBot` applique l'un des algorithmes utlisé pour résoudre le problème du voyageur de commerce :

1. Une première version du chemin optimale est réalisée de manière heuristique en prenant les missions les plus proches les unes des autres.

2. Deux missions sont aléatoirement échangées dans l'ordre défini dans l'étape précédente.

3. Le résultat est gardé s'il est meilleur que celui que l'on avait précédemment.

4. Au bout de $n$ itérations sans amélioration, on sort de la boucle d'exploration.

Une fois le chemin potentiel optimal obtenu, il l'éxecute en passant par les chemins les plus rapides.

#### Structure de données utilisée

Pour modéliser le chemin et faciliter le changement aléatoire nous sommes partis sur une liste de tuples. `[(1, 1), (2, 4), ...,  (6, 2)]`

Un tuple est composé du début de la mission et de la fin de la mission tel que `(debut, fin)`.

Le premier élément de la liste correspond à la position du robot qui effectue la recherche.

De cette manière lorsque l'on souhaite échanger deux index de la liste on déplace le couple `(debut, fin)` de mission de manière séparée.

#### Optimisation combinatoire

Plutôt que parcourir la liste pour calculer le coût total de déplacement du chemin construit ce qui aurait aurait une complexité de `O(n)`.

On décroche seulement le tuple à déplacer du tuple précédent et du suivant avant de le raccrocher ailleurs. De cette manière on obtient une complexité de `O(1)`.

### Multibot
#### Fonctionnement

Le `Multibot` va répartir les missions entre chacun des robots selon la politique suivante :
1. Chaque robot prend la mission qui lui ai la plus proche
2. Si la mission est plus proche d'un autre robot alors on la lui laisse
3. On itère jusqu'à convergence

Concernant la politique d'évitement, le `Multibot` applique une politique d'évitement simple.

### CompleteBot
#### Fonctionnement

##### Réservation de  mission
Le `CompleteBot` va répartir les missions entre chacun de ses robots selon la politique suivante :
1. Chaque robot prend la mission qui lui ai la plus proche
2. Si la mission est plus proche d'un autre robot alors on la lui laisse
3. On itère jusqu'à convergence

##### Collisions entre alliés
Concernant la politique d'évitement des alliés, le `CompleteBot` intègre une notion de priorité sur ses robots. Elle compare un score de priorité qui se calcule de la manière suivante : $score = 10*(\frac{reward}{distance}) + id$

De cette manière on applique un poid de priorité au robot qui a la mission que l'on estime la plus rentable dans le temps imparti, sinon par défaut c'est l'id le plus élevé qui prime.

En cas de confrontation, le robot non prioritaire laisse la place à l'autre robot en acceptant de s'éloigner temporairement de sa cible s'il n'a pas le choix.

##### Collisions avec le VIP
Concernant la politique d'évitement du VIP, le `CompleteBot` utilise trois réseaux de neurones pour déterminer son prochain mouvement le plus probable.

Il y a un réseau de neurone par taille de map (small, medium, large).

Ils prennent en entrée l'environnement spatiale du VIP à un rayon de 1 (toutes les cases autour de lui) et ses 5 derniers déplacements.

Nous itérons la prédiction $n$ fois, pour estimer ses déplacements à $t+n$ avec $n \in \mathbb{N}$.
Sachant que $\forall i \in [0; n], \lim_{i \to n} pred(i) = 0$.

Si on détecte une collision avec les potentiels déplacements futurs du VIP alors on ne passe pas par ce chemin.

Dans le cas où le VIP est trop proche, une sécurité est mise en place qui consiste à s'éloigner de celui-ci en selectionnant la case telle que `self._distances[current_robot_position][vip_position] < self._distances[case_position][vip_position]`.

De cette manière on s'assure d'éviter au maximum (sauf cas particulier --> impasse) les collisions avec le VIP.

##### Collisions avec les ennemies

Pour savoir où l'aversaire souhaite aller, nous avons essayé de cracker le répertoire des adversaires et de les soudoyer. Après échec, nous avons essayé de mettre en oeuvre une heuristique qui fonctionne de la manière suivante :

1. Récupérer les missions jugées intéressantes. Pour cela on récupère les missions les plus proches, les missions qui rapportent le plus de récompenses, les missions qui ont le meilleur rapport récompense/distance.

2. Tracer tous les chemins les plus courts vers ces missions et calcul de probabilités de passage.

3. Identifier de potentielles collisions entre les adversaires et nos robots et les éviter si c'est le cas.

##### Implémentation

```python
def get_next_steps_better(self, reservations=None):
    # Selection des missions
    if reservations is None:
        reservations = self.assign_missions()
    
    # Initialisation du tableau de prochains mouvements
    # Ex: [[], [("move", 0, 1)], [("move", 0, 14), ("move", 3, 15)]]
    next_moves=[[] for _ in range(self._nb_robots + 1)]
    next_moves=self.get_optimal_moves(next_moves, reservations=reservations)

    # Application des filtres
    next_moves=self.apply_priority(next_moves)
    next_moves=self.apply_vip_policies(next_moves)
    next_moves=self.filter_by_willingess(next_moves)
    next_moves=self.filter_by_any_presence(next_moves)
    next_moves=self.apply_default_move(next_moves)

    return next_moves
```

On aurait pu mettre les fonctions de filtre dans un tableau `self._filtres` et itérer pour appliquer l'ensemble des filtres avant de renvoyer `next_moves`. 

Cela dit on trouvait que de cette manière le code était plus lisible pour tous.

## Résultats

Après mise à jour du code, voici les résultats :

### Small maps with 0 vip
![Small with 0 vip](/assets/small-0-vip.webp)

### Small maps with 1 vip
![Small with 1 vip](/assets/small-1-vip.webp)

### Medium maps with 0 vip
![Medium with 0 vip](/assets/medium-0-vip.webp)

### Medium maps with 1 vip
![Medium with 1 vip](/assets/medium-1-vip.webp)

### Large maps with 0 vip
![Large with 0 vip](/assets/large-0-vip.webp)

### Large maps with 1 vip
![Large with 1 vip](/assets/large-1-vip.webp)

## Chalendgers


orange: (4 bots)

challengers: ['orange-0', 'orange-1', 'orange-2', 'orange-3']

- test: orange-0
  * small-1: (0.318)
  * medium-2: (0.345)
  * medium-3: (0.318)
  * large-1: (0.364)
  * large-2: (0.373)
- test: orange-1
  * small-1: (0.314)
  * medium-2: (0.32)
  * medium-3: (0.318)
  * large-1: (0.348)
  * large-2: (0.365)
- test: orange-2
  * small-1: (0.359)
  * medium-2: (0.381)
  * medium-3: (0.377)
  * large-1: (0.474)
  * large-2: (0.477)
- test: orange-3
  * small-1: (0.315)
  * medium-2: (0.323)
  * medium-3: (0.319)
  * large-1: (0.352)
  * large-2: (0.369)

Pour plus de détail voir le fichier `./assets/log-eval-orange-test.md`.