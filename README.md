A expliquer ici :
Les chemins / la nomenclature
Les formats des fichiers en entrée
Les commandes dans le terminal

# Description de l'arborescence

Tous les fichiers de résulstats sont enregistrés dans le dossier `outfiles`.

Un dossier `X_sentences` est créé pour chaque taille d'échantillon testée. Dans chaque dossier `X_sentences` se trouve un dossier par méthode de réduction de dimensionalité utilisée. 

Les fichiers avec les voisins trouvés se trouvent dans le dossier `outifles/knn`. 

Le corpus se trouve dans le dossier `corpus`.

## Nomenclature des fichiers

### Fichiers des voisins trouvés

Pour chaque fichier de résultat, le mot cible et la taille de l'échantillon utilisée se trouve dans le nom du fichier. 

### Fichiers des matrices réduites

Chaque fichier de matrice réduite porte dans son nom le nom de la méthode et les paramètres utilisés. En plus de cela, quand on crée la matrice terme-terme, il est possible d'apliquer ou non une PPMI et un lissage laplacien. Le fichier portera donc également la mention `PPMI` si on applique une PPMI et `addX` si on applique un lissage laplacien. 

Par exemple, le fichier de la matrice réduite avec PCA (`n_components=100`), avec PPMI et lissage laplacien + 3 aura comme nom : `PPMI_add3_PCA_ncomponents_100.tsv`.

### Fichiers d'évaluation et de comparaison

# Lancement des scripts

Pour constituer le corpus : 

```bash
python3 wikiScrap.py
```

Pour prétraiter le corpus :

```bash
python3 traitements.py 
```

Pour appliquer les méthodes de réduction de dimensionalité : 

```bash
python3 downsizing.py sample_size ppmi n_laplace
```

Il n'est pas possible d'utiliser laplace sans faire de PPMI mais on peut faire une PPMI sans lissage.