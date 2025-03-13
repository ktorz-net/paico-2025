# Repos de test PAICO 2025

Le principe du repos est de tester les bots proposés par les équipes concurrentes.
La philosophie est d'avoir des scripts de test qui génère de la donnée (résultat de parties) et d'autre pour analyser ces résultats.


## Tester une équipe :

Les scripts `test_grpcolor` permettent de tester la valider du code d'un robot. 
Pour l'instant les tests se contentent de tester si les temps max ne sont pas dépassés.
Pour bien faire, il faudrait inclure aussi des scores minimums à atteindre.

Chaque script charge tout les bots de l'équipe et s'en débarrasse au fur et à mesure qu'ils ne passent pas les tests.

Pour le lancer par exemple avec `red`:

```sh
pytest test_grpred
```

Trois fichiers sont créés: 

- `log-eval-red-test.md` qui propose un rapport des tests effectués.
- `log-solo-red-test.log` qui regroupe les résultats de toutes les parties lancées.
- `results-red.json` qui structure en json les résultats dans `log-solo-red-test.log`.

À noter que `results-red.json` est persistant la ou `log-solo-red-test.log` et `log-eval-red-test.md` seront écrasés à chaque exécution des tests.


## Générer l'analyse Solo :




## En mode Duo :