# MOVE-IT

| Fonctions / Bots | UtimateBot | SoloBot | MultiBot |
|------------------|------------|---------|----------|
|     MultiBot     |     ✅    | ✅  |     ✅     |
|       VIP        | ✅  |  ❌ |  ❌  |
|    Multijoueur   | ✅  | ❌ |   ✅ |

## Contributeurs : CATTEAU Julie, FATHOUNE Salma, FOULON Benjamin, OGLIALORO Ugo

1er scénario: Contrôler l'environnement avec plusieurs robots
- Faire effectuer les actions des robots en même temps ( concatener en une commande )
- Qui fait quoi ? Dans quel ordre ? Réduire les collisions ?
- Optimiser l'ordre des missions ( chemin le plus court entre la première et la deuxième )
- Optimiser par zone ( déterminer les zones suivant les missions, et déterminer des robots par zone )
- Simuler tous les chemins possibles avant chaque target pour éviter les collisions
- Choisir le meilleur chemin possible où il n'y a pas de collisions

2ème scénario : Eviter le VIP
Le VIP est un robot qui bouge aléatoirement et qui est prioritaire.

3ème scénario : Contrôler la situation avec plusieurs robots et un VIP

## Bots disponibles

### Bot : UltimateBot

#### Fonctionalités :
- MultiBot : ✅
- VIP : ✅
- Multijoueur : ✅

#### Stratégie adoptée :

Adapte le plus court chemin selon la situation actuelle (instant t) et la situation au tour suivant (intant t+1).

#### Décision des bots

1. Pour un instant t donné, un tri est d'abord effectué pour déterminer quel bot va prendre une décision en premier. C'est le rôle de la méthode `sort_robots_by_remain_dirs` qui permet de trier les robots en prenant en compte leurs directions possibles.
C'est alors le robot le plus en difficulté (ayant le moins de cases disponibles) qui jouera le premier.

2. La prise de décision est ensuite réalisé par la méthode `takeDecision`. Cette dernière analyse d'abord les cases disponibles restantes avec la méthode `remove_avoid_tiles` (voir **Cases exclues**). Ensuite il choisi selon la situation entre prendre une mission, terminer une mission ou se déplacer. 

3. S'il se déplace, un nouveau calcul du meilleur chemin est généré en excluant l'action de ne pas bouger dans le cas où on a plusieurs directions possibles. Cela permet de faire de l'exploration et évite la génération de bouchons. Cette solution est retourné par la méthode `moveToward`.

4. Enfin, certaines conditions particulières sont gérées :

- La fin de jeu où certains robots peuvent rester sans activités. Dans ce cas, s'ils bloquent le passage, ils prennent une direction aléatoire dans les directions possibles pour éviter les collisions (c'est le rôle de la méthode `randomMove`). 

- Lorsque les robots prennent les mêmes décisions à répétition, ce cycle est brisé par la méthode `detect_cycle` pour que les robots se débloquent.

#### Cases exclues

- Cases du vip et ses voisines
- Cases des autres robots
- Cases réservées au tour suivant

---

### Bot : SoloBot

#### Fonctionnalités :
- **MultiBot** : ✅  
- **VIP** : ❌  
- **Multijoueur** : ❌  

#### Stratégie adoptée :
Adopte le plus court chemin, basé sur le calcul de plusieurs chemins alternatifs, afin de choisir l'option la plus optimale tout en prévoyant et évitant les collisions futures.

#### Décision des bots

1. Attribution des missions :
- Chaque robot se voit attribuer une mission en fonction de la distance minimale entre sa position actuelle et la mission la plus proche (`assignOptimalMissions`).
- Le bot maximise le ratio **récompense/distance** pour prioriser les missions les plus rentables (`minDistanceToMission`).

2. Prise de décision pour chaque robot :
- Si le robot a une mission en cours et atteint sa destination, il valide la mission.
- S'il n'a pas de mission, il en cherche une nouvelle.
- Si un bot n'a plus de mission mais qu'il se trouve sur une case bloquant un autre robot en mission, il se déplace vers un voisin valide pour libérer le chemin.

3. Gestion des collisions et des obstacles :
- Avant chaque déplacement, le bot simule différents plus courts chemins possibles pour éviter les autres robots et trouver un passage sécurisé (`path`).
- Un système de détection de collision empêche les robots d’entrer en conflit (`detectCollision`).

4. Optimisation des déplacements :
- Chaque robot choisit son itinéraire en fonction du chemin le plus court disponible.
- Les robots sont priorisés selon leur marge de manœuvre : ceux avec le moins d’options de déplacement jouent en premier (`sortedBotsByPriority`).



