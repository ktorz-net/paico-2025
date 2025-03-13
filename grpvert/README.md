# MOVE-IT

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

### Bot : SoloBot

#### Fonctionalités :
- MultiBot : ✅
- VIP : ❌
- Multijoueur : ❌

#### Stratégie adoptée :

Adapte le plus court chemin selon une liste de chemins alternatifs.

### Bot : MultiBot

#### Fonctionalités :
- MultiBot : ✅
- VIP : ⚠️
- Multijoueur : ✅

#### Stratégie adoptée :

Adapte le plus court chemin selon une liste de chemins alternatifs et le bot minimise les dégâts.




