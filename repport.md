# Report

## Orange 

### Test Solo

- Erreur (`test_solo_bot_0`): `IndexError: list assignment index out of range` (`grporange/grporange/complete_bot.py:1570`)
- Errueur ( - ): `IndexError: list index out of range` (`grporange/grporange/tsp_bot.py:56`)

Au final on a 3 bots sur 4 fonctionnel (solo), basé sur de réelles différences dans l'implémentation.

Beaucoup de négatifs sur `orange-2`, ne supporte pas la montée en robots.


### Readme

Complet, mais relativement concis (inclusion d'image foireuse).

Introduction trop légère, il faut avoir suivi le module pour savoir où on met les pieds.

Process d'installs ok (mais hackagames et move-it sont inconnues à ce jour de pip)

L'intégration du ViP n'est pas évidente et des résultats rapidement négatifs, même sur de larges cartes.

- orange-0: _CompleteBot_ mais :  `index out of range`
- orange-1: _MultiBot_
- orange-2: _TspBot_
- orange-3: _MultiPlayerBot_

D'autre par c'est inquiétants que _CompleteBot_ soient présenté comme le bot le plus abouti et que se soit le seul à cracher sur le test...

à noté que  `models` qui 'contient l'ensemble de nos itérations' est superflu: c'est du bruit, et je suis capable de descendre dans les branches et les logs au besoin.


### Test Duo

Seul _Orange-3_ permet d'envisager des tests multi.


## Vert

### Test Solo

- Franchement plus lent (trop, j'ai coupé après 30 minutes, sur le test 3 (_vert-1_)).

- _vert-2_: ne passe pas la limite de temps sur large-1: (min: 26.463s vs 15.0s mais des valeurs de plus d'une minute observées)

Il ne reste donc que _vert-0_ mais avec des stats sympathiques. Seul quelque négatif sur certaine carte avec ViP.

### Readme Vert

Description des trois solutions proposée, mais rien sur leur prise en main ou sur la structuration des fichiers / du code.

Cependant, les descriptions présentent les méthodes implémentées.


### Test Duo

Effectuable uniquement sur _Vert-0_.

Les résultats sont globalement négatifs, même avec un unique robot par équipe.


## Bleu 

### Test Solo

Un seul robot _bleu-0_ mais il côtoie le temps max sur Large et finit par l'exploser sur 10 parties en large-1 (ViP).

Les temps sont maintenant similaires avec la version Ghost.

### Readme

Readme à l'opposé de vert. Complet sur les points d'entrée du projet et légère sur les implémentations/résultats obtenues.

Au vu de la complexité de la structure proposée (et il est très bien a priori de décomposer son projet), un/des diagramme(s) serait bienvenu (diagramme des classes par exemple) pour comprendre l'imbrication. 


### Test Duo

Les résultats sont globalement négatifs accepté sur des configurations monorobot. 
C'est d'autant plus décevant que les temps de calcul sont lourds.


## Fights

### Augmentation du nombre de robots

Cf. `eval-plot.py`

### Mode Duo

Orange: Vistoire par abandon.

Ni bleu ni vert ne sont capables de monter en multiplayer-multirobot.

## Conclusion

(Tous:)

1. Des débuts de tests, mais non fonctionnels chez moi.
2. Il n'est pas évident de rentrer dans le code, peu de portes d'entrée sont fournies ou la structuration est à revoir.
3. C'est à moitié buggé. Ce qui nécessite d'annuler plusieurs des bots. Est-ce que dans un cadre professionnel aussi vous fournissez du code non fonctionnel ?


### Orange (13)

Des résultats décevants au vu des promesses des stratégies et du code. 

La start _0_ buggé.

Une exploration assez large (apprentissage, optimisation) traduit par plusieurs bots implémentés.

Un Readme assez complet.



### Bleu  (11)

Des résultats décevants au vu des promesses des stratégies et du code. 

Des efforts de développement de la stratégie proposé qui se traduit par un unique bots.

Des approches très heuristiques, mais poussées sur des cas particuliers. Cependant, les notions présentées en AICO-I restent assez absentes des stratégies proposées.

Un Readme axé prise en main du code.

Une bonne structuration du code, mais ne passe pas le multiployer malgré de lourd temps de calcul.


### Vert (11)

Stratégie développée sur deux voix, qui se traduit par deux familles de bot.

Dommage qu'il n'en découle pas plus d'analyse sur la confrontation de bots.

D'autre part, seule _Bot 0_ présente des stats fonctionnelles sur des cartes larges.

Il sort largement en tête sur des scénarios solos, en augmentant le nombre de robots, mais s'écrase complétement sur du 2-joueurs.

Les approches très heuristiques qui ne font pas la part belle aux notions présentées en AICO-I.

Un Readme axés prise en main du code, et aucune structuration du code.
