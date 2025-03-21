# Report

## Orange 

### Test Solo

- Erreur (2 - `test_solo_bot_0`): `IndexError: list assignment index out of range` (`grporange/grporange/complete_bot.py:1570`)

Au final on a 3 bots sur 4 fonctionnal, basé sur de réel différence dans l'implémentation.

Beaucoup de négatif sur `orange-2`, ne supporte pas la monté en robots.


### Readme

Complet, mais relativement consis (inclusion d'image foireuse).

Introduction trop légére, il faut avoir suivi le module pour savoir ou on met les pieds.

Precess d'install ok (mais hackagames et move-it sont inconue à ce jour de pip)

L'intégration du ViP n'est pas évident au des résultat rapidement negatif, même sur de large carte.

- orange-0: _CompleteBot_ mais :  `index out of range`
- orange-1: _MultiBot_
- orange-2: _TspBot_
- orange-3: _MultiPlayerBot_

D'autre par c'est inquitant _CompleteBot_ soient présenté comme le bot le plus abouti et que se soit le seul à cracher sur le test...


### Test Duo


## Vert

### Test Solo

- Franchement plus lent (trop, j'ai coupé aprés 30 minutes, sur le test 3 (_vert-1_)).

- _vert-2_: ne passe pas la limite de temps sur large-1: (26.463s vs 15.0s)

Il ne reste donc que _vert-0_ mais avec des stats sympathique. Seul quelque negatif sur certaine carte avec ViP.

### Readme Vert

Description des trois solutions proposé mais rien sur leur prise en main ou sur la structuration des fichier / du code.

Cependant, les descriptions présente les methodes implémentés.


## Bleu 

### Test Solo

Un seul robot _bleu-0_ qui cotoye le temps max sur Large et finit par l'exploser sur 10 parties en large-1 (ViP).

Les temps sont maintenant similaire avec la version Ghost.

### Readme

Readme à l'oposé de vert. Complet sur les points d'entrée du projet et légé sur les implémentations/resultats optenue.

Au vue de la complexité de la structure proposé (et il est trés bien à priori de décomposé sont projet), un/des diagrame(s) serais bienvenue (diagrame des classes par exemple) pour comprendre l'imbrication. 


## Fights

### Augmentation du nom de robots


### 1v1


### 3v3


### 1v1 + ViP


### 3v3 + ViP



## Conclusion

(Tous:)

1. Des début de tests, mais non fonctionnel chez moi.


### Orange 

Des résultat décevant au vu des promesses des stratégies et du code. 

La start _0_ beuggé.

Une exploration assez large (apprentissage, optimisation) traduit par plusieurs bots implémentés.

Un Readme assez complet.



### Bleu 

Des résultats décevant au vu des promesses des stratégies et du code. 

Des effort de dévelopement de la stratégie proposé qui se traduit par un unique bots.

Des approches trés heuristiques, mais poussé sur des cas particuliés. Cependant, les notions présenté en AICO-I reste assez absente des stratégies proposé.

Un Readme axés prise en main du code.

Une structuration du code 


### Vert 

Stratégie dévelopé sur deux voix, qui se traduit par deux famille de bot.

Dommage qu'il n'en découle pas plus d'annalyse sur la confrontation de bots.

D'autre part, seule _Bot 0_ Presente des stats fonctionnel sur des carte larges.

Les approches trés heuristiques qui ne fait pas la part belle aux notions présenté en AICO-I.

Un Readme axés prise en main du code.

Une structuration du code 