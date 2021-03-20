source("./fonctions_exam_tp.R")
source("./script_RN.R")
source("./script_RegLog.R")
options(max.print = 50000)


# Chargement du jeu de donnÃ©es
dataset = read.table("./arabic_dataset.txt", header = T)
dataset$label = as.factor(dataset$label)

# Les lignes (1700 en tout) de ce jeu de donnÃ©es contiennent les pixels d'images en noir et blanc reprÃ©sentant des caractÃ¨res 
# de l'alphabet arabe (colonnes 2 Ã  1025, car 1024 pixels, image 32*32), ainsi que la colonne label (colonne 1) qui
# correspond au caractÃ¨re reprÃ©sentÃ© sur l'image.
# Dans ce jeu de donnÃ©es, il y a 7 caractÃ¨res diffÃ©rents (numÃ©rotÃ©s 1,2,6,12,16,18,22) comme expliquÃ© dans le sujet pdf
# de ce TP

# Pour afficher une image, il faut choisir un indice de ligne et utiliser la fonction display (que vous avez dans 
# le fichier fonctions_exam_tp.R):
idx_ligne = 1 # je choisis la ligne 1 (vous pourrez changer)
display(matrix(as.numeric(dataset[idx_ligne,2:1025]), ncol = 32))

# Le tableau competition contient 500 nouvelles images pour lesquelles on vous demande de prÃ©dire la classe
# (c'est Ã  dire de prÃ©dire quel caractÃ¨re est dessinÃ© sur chaque image),
# et d'aller soumettre vos prÃ©dictions sur la page Kaggle.
# Pour charger les donnÃ©es de competition: 
competition = read.table("./competition.txt", header = T)
# Vous pourrez remarquer que vous ne connaissez que les pixels de ces 500 images et pas le label

# Pour soumettre vos prÃ©dictions sur kaggle (Ã  faire aprÃ¨s Ã©tude de chaque famille de classifieur : arbre, svm, knn, etc) : 
# si p est un vecteur contenant vos prÃ©dictions sur le jeu de compÃ©tition, il faut procÃ©der comme suit pour faire une soumission :
write.csv(data.frame("Id"=c(1:500), "label"=p), file ="./mespredictions.csv", row.names = F)
# et ensuite, vous allez soumettre ce fichier sur kaggle
# Quand vous soumettez vos prÃ©dictions, n'oubliez pas de mettre en commentaire (sur le site kaggle), Ã  quoi correspondent ces 
# prÃ©dictions.
# Attention, le nombre de soumissions est limitÃ©, donc choisissez bien les prÃ©dictions que vous voulez soumettre.

# A rendre : 
# - ce fichier que vous complÃ©terez 
# - le fichier fonctions_exam_tp.R, si vous utilisez des fonctions annexes
# - un compte rendu (manuscrit ou pdf)



##################################################################
# Partie prÃ©liminaire : description et analyse du jeu de donnÃ©es #
##################################################################

# complÃ©ter ici
# resumé des données :  
str(dataset)
# repartition des labels
table(dataset$label)


###########################################################
####  PremiÃ¨re partie : utilisation des images brutes  ####
###########################################################

# Dans cette partie, vous allez travailler avec les images brutes (sans faire de traitement) pour crÃ©er 
# vos modÃ¨les de classification. Vous avez vu en cours (module INV) que travailler avec des images brutes 
# n'Ã©tait pas la meilleure solution, mais vous allez commencer comme Ã§a, avant de tenter de faire mieux.


# 1) Arbres de dÃ©cision
# Pour les arbres de dÃ©cision, commencer par:
library(rpart)

#séparation Apprentissage/Validation/test

set.seed(20) # (pour reproduire mes résultats d'une session à l'autre)
nall = nrow(dataset) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = dataset[index[1:ntrain],] # création jeu d'apprentissage
valid = dataset[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = dataset[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 1025 colonnes
dim(valid) #255 lignes et 1025 colonnes
dim(test) #255 lignes et 1025 colonnes

#definition du modèle sans elagage
tr = rpart(label~., data = train, control = list(minbucket = 1,cp = 0, minsplit = 1))

#prediction jeu de d'apprentissage 
sum(predict(tr, train, type = "class") == train$label) # 1190 bonnes prédictions 
(sum(predict(tr, train, type = "class") != train$label)/1190)*100 # soit 0% d'erreur

#prediction jeu de validation
sum(predict(tr, valid, type = "class") == valid$label) # 130 bonnes prédictions
(sum(predict(tr, valid, type = "class") != valid$label)/255)*100 # soit 49.01961% d'erreur


#essayon d'elaguer l'arbre pour améliorer les perf
# Les valeurs de cp permettant d'Ã©laguer l'arbre peuvent Ãªtre trouvÃ©es en regardant le rÃ©sultat de la commande suivante : 
valeurs_cp = rev(tr$cptable[,1])
#valeur possible de cp 
valeurs_cp


# Methode plus rapide pour choisir la valeur permerttant d'améliorer les performances en validation
v = seq(from=0.0005, to=0.0702274975, by=0.001)
res = c()
for (i in 1:length(v)){
  tr_elague = prune(tr, cp = v[i])
  err_app = sum(predict(tr_elague, train, type = "class") == train$label)
  err_val = sum(predict(tr_elague, valid, type = "class") == valid$label)
  res = cbind(res, c(err_app= err_app, err_val = err_val, c = v[i]))
  
}
res

# on prend le modèle pour laquel cp donne le meilleur compromis en apprentissage/validation
# c'est à dire : cp = which.max(res[2,]) # 0.0005
which.max(res[2,])
tr_elague = prune(tr, cp = 0.0005)
#prediction
sum(predict(tr_elague, train, type = "class") == train$label) # 1185 bonnes reponses en apprentissage
sum(predict(tr_elague, valid, type = "class") == valid$label) # 130 bonnes reponses en validation


#prediction jeu de test avec le modèle choisi
sum(predict(tr_elague, test, type = "class") == test$label)# 137 bonnes prédiction


#utilisation du modèle retenu pour prédire les données compétitions
pred_arbre = predict(tr_elague, competition, type = "class")
write.csv(data.frame("Id"=c(1:500), "label"=pred_arbre), file ="./mespredictions_arbre.csv", row.names = F)

# 2) Support Vector Machine (SVM) 
library(e1071)

#séparation Apprentissage/Validation/test

set.seed(20) # (pour reproduire mes résultats d'une session à l'autre)
nall = nrow(dataset) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = dataset[index[1:ntrain],] # création jeu d'apprentissage
valid = dataset[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = dataset[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 1025 colonnes
dim(valid) #255 lignes et 1025 colonnes
dim(test) #255 lignes et 1025 colonnes

# Methode linéaire

# on prend c= 50 pour tolérer 50 points mal placés. on reduira au fur et à mesure de l'étude
svm_model = svm(label~.,data = dataset, kernel='linear',cost=50,type='C-classification', scale = F)


# prédiction apprentissage
sum(train$label == predict(svm_model, train)) # 1190 bonnes prédictions
(sum(train$label != predict(svm_model, train))/1190)*100 # soit 0% d'erreur

# Prédiction validation
sum(valid$label == predict(svm_model, valid)) #255 bonnes predictions
(sum(valid$label != predict(svm_model, valid))/255)*100 # soit 0% d'erreur



#faisons varier c pour voir l'évolution 
c = seq(from=1,to=10,by=2)
result = c()
for(i in 1:length(c)){
  svm_model = svm(label~.,data = dataset, kernel='linear',cost=c[i],type='C-classification', scale = F)
  god_pred_app = sum(train$label == predict(svm_model, train))
  god_pred_valid = sum(valid$label == predict(svm_model, valid))
  result = cbind(result, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, cost = c[i]))
  
}
result # meilleur compromis apprentissage/validation avec cost = 1 


#Methode du noyau gaussien

# modèle avec gamma = 1 et cost = 1 pour commencer
svm_model_gaussien <- svm(label~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = 1)

# Performance en apprentissage
sum(train$label == predict(svm_model_gaussien, train)) #1190 bonnes prédiction en apprentissage
(sum(train$label != predict(svm_model_gaussien, train))/1190)*100 # soit 0% d'erreur

# Performance en validation
sum(valid$label == predict(svm_model_gaussien, valid)) #38 bonnes prédictions en validation
(sum(valid$label != predict(svm_model_gaussien, valid))/255)*100 # soit 85.09804% d'erreur



# modèle avec gamma = 0.2 et cost = 1 pour commencer
svm_model_gaussien <- svm(label~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = 0.2)

# Performance en apprentissage
sum(train$label == predict(svm_model_gaussien, train)) #1190 bonnes prédiction en apprentissage
(sum(train$label != predict(svm_model_gaussien, train))/1190)*100 # soit 0% d'erreur


# Performance en validation
sum(valid$label == predict(svm_model_gaussien, valid)) #60 bonnes prédictions en validation
(sum(valid$label != predict(svm_model_gaussien, valid))/255)*100 # soit 76.47059% d'erreur

# modèle avec gamma = 0.01 et cost = 1
svm_model_gaussien <- svm(label~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = 0.01)

# Performance en apprentissage
sum(train$label == predict(svm_model_gaussien, train)) #1092 bonnes prédiction en apprentissage
(sum(train$label != predict(svm_model_gaussien, train))/1190)*100 # soit 8.235294% d'erreur

# Performance en validation
sum(valid$label == predict(svm_model_gaussien, valid)) #188 bonnes prédictions en validation
(sum(valid$label != predict(svm_model_gaussien, valid))/255)*100 # soit 26.27451% d'erreur


# Méthode polynomiale

#modèle avec cost = 1 puis dégrés = 2 pour commencer.
svm_model_poly <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 2, coef0 = 1)

# performance apprentissage 
sum(train$label == predict(svm_model_poly, train)) #896 bonnes prédictions en apprentissage
(sum(train$label != predict(svm_model_poly, train))/1190)*100 # soit 24.70588% d'erreur

#performance en validation
sum(valid$label == predict(svm_model_poly, valid)) #162 bonnes prédiction en validation
(sum(valid$label != predict(svm_model_poly, valid))/255)*100 # soit 36.47059% d'erreur 



#essayons d'améliorer le modèle avec différente valeur de degree
# avec degree = 3
#modèle avec cost = 1 puis dégrés = 3 pour commencer.
svm_model_poly_3 <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 3, coef0 = 1)

# performance apprentissage 
sum(train$label == predict(svm_model_poly_3, train)) #1001 bonnes prédictions en apprentissage
(sum(train$label != predict(svm_model_poly_3, train))/1190)*100 # soit 15.88235% d'erreur

#performance en validation
sum(valid$label == predict(svm_model_poly_3, valid)) #171 bonnes prédiction en validation
(sum(valid$label != predict(svm_model_poly_3, valid))/255)*100 # soit 32.94118% d'erreur 

#essayons d'améliorer le modèle avec différente valeur de degree
# avec degree = 5
#modèle avec cost = 1 puis dégrés = 5 pour commencer.
svm_model_poly_5 <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 5, coef0 = 1)

# performance apprentissage 
sum(train$label == predict(svm_model_poly_5, train)) #1139 bonnes prédictions en apprentissage
(sum(train$label != predict(svm_model_poly_5, train))/1190)*100 # soit 4.285714% d'erreur

#performance en validation
sum(valid$label == predict(svm_model_poly_5, valid)) #170 bonnes prédiction en validation
(sum(valid$label != predict(svm_model_poly_5, valid))/255)*100 # soit 33.33333% d'erreur

#Essaie de masse 
c = seq(from=5,to=10,by=2)
result = c()
for(i in 1:length(c)){
  svm_model_poly_x <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = c[i], coef0 = 1)
  god_pred_app = sum(train$label == predict(svm_model_poly_x , train))
  god_pred_valid = sum(valid$label == predict(svm_model_poly_x , valid))
  result = cbind(result, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, degree = c[i]))
  
}
result

#choisix du modèl définitif  :
svm_model_lineaire = svm(label~.,data = dataset, kernel='linear',cost=1,type='C-classification', scale = F)
svm_model_gauss <- svm(label~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = 0.01)
svm_model_poly <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 7, coef0 = 1)

sum(test$label == predict(svm_model_lineaire, test)) #255 bonnes prédiction sur le jeu de données de test.
sum(test$label == predict(svm_model_gauss, test)) #190 bonnes prédiction sur le jeu de données de test.
sum(test$label == predict(svm_model_poly, test)) #182 bonnes prédiction sur le jeu de données de test.


#prédiction du jeu de donnée compétition avec le meilleur modèle svm
#utilisation du modèle retenu pour prédire les données compétitions
pred_svm = predict(svm_model_lineaire, competition, type = "class")
write.csv(data.frame("Id"=c(1:500), "label"=pred_svm), file ="./mespredictions_SVM_best.csv", row.names = F)

# 3) K-plus-proches voisins (KNN)
library(class)

#séparation Apprentissage/Validation/test
set.seed(20) # (pour reproduire mes résultats d'une session à l'autre)
nall = nrow(dataset) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = dataset[index[1:ntrain],] # création jeu d'apprentissage
valid = dataset[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = dataset[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 1025 colonnes
dim(valid) #255 lignes et 1025 colonnes
dim(test) #255 lignes et 1025 colonnes

# modèle
p_val= knn(train[,2:1025], valid[,2:1025], train$label,k=1)

# affichage prédiction
head(p_val)

# performance 
sum(p_val == valid$label) # 177 bonnes prédictions
(sum(p_val != valid$label)/255)*100 # soit 30.588245% d'erreur.

# amélioration du modèle 
v = seq(from=2, to=10, by=2)
resultat = c()
for (i in 1 : length(v)){
  p_val= knn(train[,2:1025], valid[,2:1025], train$label, k=v[i])
  resultat = cbind(resultat, c(good_pred_valid = sum(p_val == valid$label), k = v[i]))
}
resultat # meilleur perfromance en validation avec  k = 4

#prédiction test 

p_test = knn(train[,2:1025], test[,2:1025], train$label,k=2)
sum(p_test == test$label) # 184 bonnes prédictions
(sum(p_test != test$label)/255)*100 # soit 30.58824% d'erreur.


p_test = knn(train[,2:1025], test[,2:1025], train$label,k=4)
sum(p_test == test$label) # 174 bonnes prédictions
(sum(p_test != test$label)/255)*100 # soit 31.76471% d'erreur.

#prédiction compétition
pred_knn = knn(train[,2:1025], competition[,1:1024], train$label,k=2)
write.csv(data.frame("Id"=c(1:500), "label"=pred_knn), file ="./mespredictions_knn.csv", row.names = F)

# complÃ©ter ici
# 4) ForÃªt alÃ©atoire
library(randomForest)

#séparation Apprentissage/Validation/test
set.seed(20) # (pour reproduire mes résultats d'une session à l'autre)
nall = nrow(dataset) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = dataset[index[1:ntrain],] # création jeu d'apprentissage
valid = dataset[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = dataset[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 1025 colonnes
dim(valid) #255 lignes et 1025 colonnes
dim(test) #255 lignes et 1025 colonnes

# Definition du modèle
# on va avec 2 arbres pour commencer (ntree = 2)
foret = randomForest(label~., data = train, ntree =2)

# performance apprentissage 
sum(train$label == predict(foret, train)) #956 bonnes prédictions en apprentissage
(sum(train$label != predict(foret, train))/1190)*100 # soit 19.66387% d'erreur

#performance en validation
sum(valid$label == predict(foret, valid)) #117 bonnes prédiction en validation
(sum(valid$label != predict(foret, valid))/255)*100 # soit 54.11765% d'erreur



#essaie de ntree entre 100 à 1000
c = seq(from=500, to=1000, by=100)
resultat = c()
for (i in 1:length(c)){
  foret = randomForest(label~., data = train, ntree =c[i])
  god_pred_app = sum(predict(foret, train) == train$label)
  god_pred_valid = sum(predict(foret, valid) == valid$label)
  resultat = cbind(resultat, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, ntree = c[i]))
}
resultat

# modèle définitif
foret_best = randomForest(label~., data = train, ntree =500)

# performance apprentissage 
sum(test$label == predict(foret_best, test)) #201 bonnes prédictions sur les données de test.

#utilisation du modèle retenu pour prédire les données compétitions
pred_foret = predict(foret_best, competition, type = "class")
write.csv(data.frame("Id"=c(1:500), "label"=pred_foret), file ="./mespredictions_foret.csv", row.names = F)


# complÃ©ter ici

# 5) RÃ©gression logistique

# Pour la regression logistique, vous aurez besoin des fonctions de nnets.R ( comme vu dans le TP sur la rÃ©gression logistique
# pour les chiffres ). Attention, je vous conseille d'apprendre un modÃ¨le avec au maximum 150-200 images (sinon ce sera trop long)
# Et pas besoin de puissances ici car les pixels sont 0 ou 1

source("./nnets.R")

# complÃ©ter ici

# modèle 
reg_log = learn_nn(train[1:200, 2:1025], train$label[1:200], 0, 10, c(1024, 7))


# performance apprentissage 
output = calcul_sorties(reg_log, train[1:200, 2:1025])
v = sapply(c(1:200), function (x){
  which.max(output[, x])
} 
)
v

# j'utilise une mÃ©thode autre que celle donnÃ©es par le prof pour gÃ©rer les labels dans cette partie.
# corespondance valeur de v et valeurs de label
x = c()
for (i in 1:length(v)){
  if(v[i]==1) { 
    a = 1
    x = cbind(x, c(a))
  }
  if(v[i]==2) {
     a = 2
     x = cbind(x, c(a))
     }
  if(v[i]==3) {
     a = 6
     x = cbind(x, c(a))
    }
  if(v[i]==4) {
    a = 12
    x = cbind(x, c(a))
    }
  if(v[i]==5) {
    a = 16
    x = cbind(x, c(a))
    }
  if(v[i]==6) {
    a = 16
    x = cbind(x, c(a))
    }
  if(v[i]==7) {
    a = 22
    x = cbind(x, c(a))
    }
}
x

sum(x[1,] == train$label[1:200])# 65 bonnes reponses
(sum(x[1,] != train$label[1:200])/200)*100 # soit 67% d'erreur


# performance en validation  
output = calcul_sorties(reg_log, valid[ ,2:1025])
v = sapply(c(1:255), function (x){
  which.max(output[, x])
} 
)
v

x = c()
for (i in 1:length(v)){
  if(v[i]==1) { 
    a = 1
    x = cbind(x, c(a))
  }
  if(v[i]==2) {
    a = 2
    x = cbind(x, c(a))
  }
  if(v[i]==3) {
    a = 6
    x = cbind(x, c(a))
  }
  if(v[i]==4) {
    a = 12
    x = cbind(x, c(a))
  }
  if(v[i]==5) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==6) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==7) {
    a = 22
    x = cbind(x, c(a))
  }
}
x

sum(x[1,] == valid$label)# 65 bonnes prédictions 
(sum(x[1,] != valid$label)/255)*100 # soit 74.5098% d'erreurs

# essayons d'amélioer notre modèle
reg_log = learn_nn(train[,2:1025], train$label, 0, 100, c(1024, 7))

# performance apprentissage 
output = calcul_sorties(reg_log, train[ ,2:1025])
v = sapply(c(1:1190), function (x){
  which.max(output[, x])
} 
)
v

# corespondance valeur de v et valeurs de label
x = c()
for (i in 1:length(v)){
  if(v[i]==1) { 
    a = 1
    x = cbind(x, c(a))
  }
  if(v[i]==2) {
    a = 2
    x = cbind(x, c(a))
  }
  if(v[i]==3) {
    a = 6
    x = cbind(x, c(a))
  }
  if(v[i]==4) {
    a = 12
    x = cbind(x, c(a))
  }
  if(v[i]==5) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==6) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==7) {
    a = 22
    x = cbind(x, c(a))
  }
}
x

# performance en apprentissage
sum(x[1,] == train$label)# 452 bonnes reponses
(sum(x[1,] != train$labe)/1190)*100 # soit 62.01681% d'erreur


# performance en validation  
output = calcul_sorties(reg_log, valid[ ,2:1025])
v = sapply(c(1:255), function (x){
  which.max(output[, x])
} 
)
v

x = c()
for (i in 1:length(v)){
  if(v[i]==1) { 
    a = 1
    x = cbind(x, c(a))
  }
  if(v[i]==2) {
    a = 2
    x = cbind(x, c(a))
  }
  if(v[i]==3) {
    a = 6
    x = cbind(x, c(a))
  }
  if(v[i]==4) {
    a = 12
    x = cbind(x, c(a))
  }
  if(v[i]==5) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==6) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==7) {
    a = 22
    x = cbind(x, c(a))
  }
}
x

sum(x[1,] == valid$label)# 85 bonnes prédictions 
(sum(x[1,] != valid$label)/255)*100 # soit 66.66667% d'erreurs

# le second modèle est meilleur que le 1er. c'est donc ce modèle qui est choisit pour prédire l'ensemble de test.
# performance sur les données de test 

output = calcul_sorties(reg_log, test[ ,2:1025])
v = sapply(c(1:255), function (x){
  which.max(output[, x])
} 
)
v

x = c()
for (i in 1:length(v)){
  if(v[i]==1) { 
    a = 1
    x = cbind(x, c(a))
  }
  if(v[i]==2) {
    a = 2
    x = cbind(x, c(a))
  }
  if(v[i]==3) {
    a = 6
    x = cbind(x, c(a))
  }
  if(v[i]==4) {
    a = 12
    x = cbind(x, c(a))
  }
  if(v[i]==5) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==6) {
    a = 16
    x = cbind(x, c(a))
  }
  if(v[i]==7) {
    a = 22
    x = cbind(x, c(a))
  }
}
x

# performance en test
sum(x[1,] == test$label)# 90 bonnes prédictions 
(sum(x[1,] != test$label)/255)*100 # soit 64.705887% d'erreurs



# 6) RÃ©seaux de neurones
library(keras)
library(tensorflow)
source("./fonctions_keras.R")

# Chargement du jeu de données
dataset = read.table("./arabic_dataset.txt", header = T)
# Séparation  Apprentissage/Test
set.seed(20)
nall = nrow(dataset) 
ntrain = floor(0.80 * nall)
ntest = floor(0.20* nall)
index = sample(nall)

train_x = dataset[index[1:ntrain],2:1025] # ensemble d'apprentisssage
train_labels = dataset[index[1:ntrain],1] # labels d'apprentissage

head(train_x)
head(train_labels)

test_x = dataset[index[(ntrain+1):nall],2:1025] # ensemble de test
test_labels = dataset[index[(ntrain+1):nall],1] # labels de test

head(test_x)
head(test_labels)

train_x = matrix(unlist(train_x), ncol = 1024)
test_x = matrix(unlist(test_x), ncol = 1024)

head(train_x)
head(test_x)

# On va donc soustraire 1 Ã  train_labels et test_labels, et creer train_y et test_y:
train_y = train_labels-1
test_y = test_labels-1

head(train_y)
head(test_y)

# propositionnalisation des vecteurs
train_y = to_categorical(train_y) 
test_y = to_categorical(test_y)


# test des ensembles
head(train_y) # propositionnalisation => OK
head(test_y)


#modèle 
# 22 sorties car après propositionalisation on obtient des labels qui vont de 1 à 22
# cela ne pose pas vraiment de problème car outre 1, 2, 6, 12, 16, 18, 22, les autres valeurs sont a 0

model <- keras_model_sequential()
model %>%
  layer_dense(units = 100, input_shape =1024, activation = 'relu') %>%
  layer_dense(units = 80, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 22, activation = 'softmax') 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


#modèle
history <- model %>% fit(train_x, train_y, epochs = 600, batch.size = 10, validation_split = 0.2, view_metrics=F)
#callbacks = list(callback_early_stopping("val_loss", patience = 100)

#visualisation
plot_model_loss(history)
plot_model_accuracy(history)

model %>% fit(train_x, train_y, epochs = 600, batch.size = 10, validation_split = 0.2, callbacks = list(callback_early_stopping("val_loss", patience = 100)), view_metrics=F)
#proba de chaque valeur pour tous les indiv
pred = predict(model, test_x)
pred # prédiction
dim(test_x)
#classe choisi pour tous les indiv
v = sapply(c(1:340), function (x){
  which.max(pred[x, ])
} 
)

v_test = sapply(c(1:340), function (x){
  which.max(test_y[x, ])
} 
)
v_test # vraie classes 
v # classe prédictes

# nombre de bonnes prédiction.
sum(v == v_test) # 244
(sum(v != v_test)/340)*100 # soit 28.23529% d'erreur

# accuracy du modèle
model%>%evaluate(test_x, test_y) #taux de bonnes reponse 71,76%

# Nouveau modèle (essayons d'avoir une accuracy plus proche de 1)

# modèle2 ( meilleur modèle après plusieurs essaies)
history <- model %>% fit(train_x, train_y, epochs = 500, batch.size = 10, validation_split = 0.2,  view_metrics=F)

#visualisation
plot_model_loss(history)
plot_model_accuracy(history)

# accuracy du modèle
model%>%evaluate(test_x, test_y) #taux de bonnes reponse 72,35%

#Prediction data compétition
competition = matrix(unlist(competition), ncol = 1024)
pred_rx_neu = predict(model, competition)
pred_final = sapply(c(1:500), function (x){
  which.max(pred_rx_neu[x, ])
} 
)
write.csv(data.frame("Id"=c(1:500), "label"=pred_final), file ="./mespredictions_rx_neu.csv", row.names = F)


#############################################################################
####  DeuxiÃ¨me partie : utilisation de la reprÃ©sentation HOG des images  ####
#############################################################################


## Vous allez maintenant utiliser la reprÃ©sentation HOG des images afin d'essayer d'amÃ©liorer votre score.
library(OpenImageR)
# Rappel : 
# La reprÃ©sentation HOG prend deux paramÃ¨tres cells et orientation (que j'appelle cel et ori)
# Par exemple, on peut commencer avec
cel = 2
ori = 6
# Pour transformer une image (qui correspond Ã  une ligne de votre jeu de donnÃ©es), il faut
idx = 1; # on transforme la premiÃ¨re image
h = HOG(matrix(as.numeric(dataset[idx,2:1025]), nrow = 32, byrow = T), cells = cel, orientations = ori)
length(h) # 24

# Pour transformer toutes les images de dataset:
hog_data = matrix(0,nrow(dataset),cel*cel*ori)
for(i in 1:nrow(dataset)){hog_data[i,] = HOG(matrix(as.numeric(dataset[i,2:1025]), nrow = 32, byrow = T), cells = cel, orientations = ori)}
hog_data  = data.frame("label" = as.factor(dataset$label), hog_data)

# La matrice hog_data contient les images transformÃ©es (en ligne). La premiÃ¨re colonne reprÃ©sente la classe (label) de l'image
# c'est Ã  dire un des caractÃ¨res de l'alphabet arabe que l'on Ã©tudie (1,2,6,12,16,18,22)

# Reprenez les diffÃ©rentes Ã©tapes de la premiÃ¨re partie en utilisant la reprÃ©sentation HOG, avec pour objectif d'amÃ©liorer
# votre score.
# Vous pouvez changer les paramÃ¨tres cel et ori 

# Pour appliquer les diffÃ©rents classifieurs au jeu de compÃ©tition, il faudra Ã©galement transformer en HOG toutes les images 
# du jeu de compÃ©tition

# ForÃªt alÃ©atoire
library(randomForest)

#séparation apprentissage/validation/test
set.seed(20) # (pour reproduire mes résultats d'une session à l'autre)
nall = nrow(hog_data) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = hog_data[index[1:ntrain],] # création jeu d'apprentissage
valid = hog_data[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = hog_data[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 25 colonnes
dim(valid) #255 lignes et 25 colonnes
dim(test) #255 lignes et 25 colonnes

head(train)

#modèle

c = seq(from=100, to=1000, by=100)
resultat = c()
for (i in 1:length(c)){
  foret = randomForest(label~., data = train, ntree =c[i])
  god_pred_app = sum(predict(foret, train) == train$label)
  god_pred_valid = sum(predict(foret, valid) == valid$label)
  resultat = cbind(resultat, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, ntree = c[i]))
}
resultat
# performance test
foret = randomForest(label~., data = train, ntree =600)
sum(test$label == predict(foret, test)) #233 bonnes prédictions sur les données de test.


foret = randomForest(label~., data = train, ntree =1000)
sum(test$label == predict(foret, test)) #232 bonnes prédictions sur les données de test.

# le modèle avec nombre d'arbre = 600 est effectivement le meilleur modèle
foret_best = randomForest(label~., data = train, ntree =600)


# transformation du data set competition au format HOG.
hog_competition = matrix(0,nrow(competition),cel*cel*ori)
for(i in 1:nrow(competition)){hog_competition[i,] = HOG(matrix(as.numeric(competition[i,1:1024]), nrow = 32, byrow = T), cells = cel, orientations = ori)}

hog_competition = data.frame(hog_competition)
head(hog_competition)
# utilisation du modèle retenu pour prédire les données compétitions
pred_foret_hog = predict(foret_best, hog_competition, type = "class")
write.csv(data.frame("Id"=c(1:500), "label"=pred_foret_hog), file ="./mespredictions_foret_hog.csv", row.names = F)

# essayons d'améliorer nombre modèle en augmentant le nombre d'orientations
# objectif : rendre la transformation plus informative
cel = 3
ori = 8

# Tranformation des images du dataset avec les nouvelles valeurs de cell et ori
hog_data = matrix(0,nrow(dataset),cel*cel*ori)
for(i in 1:nrow(dataset)){hog_data[i,] = HOG(matrix(as.numeric(dataset[i,2:1025]), nrow = 32, byrow = T), cells = cel, orientations = ori)}
hog_data  = data.frame("label" = as.factor(dataset$label), hog_data) # Pour rendre la cible catégorielle

#séparation apprentissage/validation/test
set.seed(20) # (pour reproduire mes résultats d'une session à l'autre)
nall = nrow(hog_data) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = hog_data[index[1:ntrain],] # création jeu d'apprentissage
valid = hog_data[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = hog_data[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 73 colonnes
dim(valid) #255 lignes et 73 colonnes
dim(test) #255 lignes et 73 colonnes

#modèle

c = seq(from=100, to=1000, by=100)
resultat = c()
for (i in 1:length(c)){
  foret = randomForest(label~., data = train, ntree =c[i])
  god_pred_app = sum(predict(foret, train) == train$label)
  god_pred_valid = sum(predict(foret, valid) == valid$label)
  resultat = cbind(resultat, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, ntree = c[i]))
}
resultat # meilleur score en validation avec ntree = 100.


# Performance sur le jeu test
foret = randomForest(label~., data = train, ntree =100)
sum(test$label == predict(foret, test)) #238 bonnes prédictions sur les données de test.

foret = randomForest(label~., data = train, ntree =400)
sum(test$label == predict(foret, test)) #238 bonnes prédictions sur les données de test.

foret = randomForest(label~., data = train, ntree =800)
sum(test$label == predict(foret, test)) #241 bonnes prédictions sur les données de test.

# Prédiction compétition avec le modèle ntree = 800

# transformation du data set competition au format HOG.
hog_competition = matrix(0,nrow(competition),cel*cel*ori)
for(i in 1:nrow(competition)){hog_competition[i,] = HOG(matrix(as.numeric(competition[i,1:1024]), nrow = 32, byrow = T), cells = cel, orientations = ori)}

hog_competition = data.frame(hog_competition)# transformation dataframe
# utilisation du modèle (ntree = 800) retenu pour prédire les données compétitions
pred_foret_hog = predict(foret, hog_competition, type = "class")
write.csv(data.frame("Id"=c(1:500), "label"=pred_foret_hog), file ="./mespredictions_foret_hog_cell3_ori8.csv", row.names = F)


# 2) Arbres de dÃ©cision avec transformation HOG

library(rpart)

#affichage des différents ensemble
head(train)
head(valid) 
head(test)

#dimension
dim(train)
dim(valid) 
dim(test)

# modèle sans sans elagage
tr = rpart(label~., data = train, control = list(minbucket = 1,cp = 0, minsplit = 1))

#prediction jeu de d'apprentissage 
sum(predict(tr, train, type = "class") == train$label) # 1190 bonnes prédictions 
(sum(predict(tr, train, type = "class") != train$label)/1190)*100 # soit 0% d'erreur

#prediction jeu de validation
sum(predict(tr, valid, type = "class") == valid$label) # 183 bonnes prédictions
(sum(predict(tr, valid, type = "class") != valid$label)/255)*100 # soit 28.23529% d'erreur

#essayon d'elaguer l'arbre pour améliorer les perf
# Les valeurs de cp possibles  
valeurs_cp = rev(tr$cptable[,1])
#valeur possible de cp 
valeurs_cp


# Methode plus rapide pour choisir la valeur permerttant d'améliorer les performances en validation
v = seq(from=0.0005, to=0.1513353116, by=0.001)
res = c()
for (i in 1:length(v)){
  tr_elague = prune(tr, cp = v[i])
  good_pred_app = sum(predict(tr_elague, train, type = "class") == train$label)
  good_pred_valid = sum(predict(tr_elague, valid, type = "class") == valid$label)
  res = cbind(res, c(good_pred_app= good_pred_app, good_pred_valid = good_pred_valid, cp = v[i]))
  
}
res # 02 valeurs de cp intéressante pour faire un meilleur compromis apprentissage/validation (cp = 0.0005 et cp = 0.0015)

# performance sur les données de test 

# cp = 0.0005
tr_elague = prune(tr, cp = 0.0005)
sum(predict(tr_elague, test, type = "class") == test$label)# 193 bonnes prédiction

# cp = 0.0015
tr_elague = prune(tr, cp = 0.0015)
sum(predict(tr_elague, test, type = "class") == test$label)# 197 bonnes prédiction

# 3) KNN

cel = 3
ori = 8

# modèle
p_val= knn(train[,2:73], valid[,2:73], train$label,k=1)

# affichage prédiction
head(p_val)

# performance 
sum(p_val == valid$label) # 223 bonnes prédictions
(sum(p_val != valid$label)/255)*100 # soit 12.54902% d'erreur.


# amélioration du modèle 
v = seq(from=2, to=10, by=2)
resultat = c()
for (i in 1 : length(v)){
  p_val= knn(train[,2:73], valid[,2:73], train$label,k=v[i])
  resultat = cbind(resultat, c(good_pred_valid = sum(p_val == valid$label), k = v[i]))
}
resultat # meilleur erreu en validation avec k = 4 et/ou 8

# Performance sur test
p_test= knn(train[,2:73], test[,2:73], train$label,k=4)
sum(p_val == test$label) # 220 bonnes prédictions

p_val= knn(train[,2:73], test[,2:73], train$label,k=8)
sum(p_val == test$label) # 222 bonnes prédictions


# 4) SVM avec transformation HOG (cell = 3, ori = 8)
library(e1071)

# on prend c= 50 pour tolérer 50 points mal placés. on reduira au fur et à mesure de l'étude
svm_model = svm(label~.,data = train, kernel='linear',cost=50,type='C-classification', scale = T)

# prédiction apprentissage
sum(train$label == predict(svm_model, train)) # 1181 bonnes prédictions
(sum(train$label != predict(svm_model, train))/1190)*100 # soit 0.7563025% d'erreur

# Prédiction validation
sum(valid$label == predict(svm_model, valid)) # 211 bonnes predictions
(sum(valid$label != predict(svm_model,valid))/255)*100 # soit 17.2549% d'erreur



#faisons varier c pour voir l'évolution 
c= seq(from=1,to=10,by=2)
result = c()
for(i in 1:length(c)){
  svm_model = svm(label~.,data = train, kernel='linear',cost=c[i],type='C-classification', scale = T)
  god_pred_app = sum(train$label == predict(svm_model, train))
  god_pred_valid = sum(valid$label == predict(svm_model, valid))
  result = cbind(result, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, cost = c[i]))
  
}
result # cost = 5 semble données de meilleur prédiction en validation.


#Methode du noyau gaussien

# modèle avec gamma = 1 et cost = 1 pour commencer
svm_model_gaussien <- svm(label~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = 1)

# Performance en apprentissage
sum(train$label == predict(svm_model_gaussien, train)) #1190 bonnes prédiction en apprentissage
(sum(train$label != predict(svm_model_gaussien, train))/1190)*100 # soit 0% d'erreur

# Performance en validation
sum(valid$label == predict(svm_model_gaussien, valid)) #45 bonnes prédictions en validation
(sum(valid$label != predict(svm_model_gaussien, valid))/255)*100 # soit 82.35294% d'erreur


c= seq(from=0.01,to=0.2,by=0.01)
result = c()
for(i in 1:length(c)){
  svm_model_gaussien <- svm(label~.,data = train, kernel='radial',cost=1,type='C-classification', gamma = c[i])
  god_pred_app = sum(train$label == predict(svm_model_gaussien, train))
  god_pred_valid = sum(valid$label == predict(svm_model_gaussien, valid))
  result = cbind(result, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, gamma = c[i]))
  
}
result # meilleur conpromis apprentissage/validation avec le couple (cost = 1, gamma = 0.02)


# Méthode polynomiale

#modèle avec cost = 1 puis dégrés = 2 pour commencer.
svm_model_poly <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 2, coef0 = 1)

# performance apprentissage 
sum(train$label == predict(svm_model_poly, train)) #1139 bonnes prédictions en apprentissage
(sum(train$label != predict(svm_model_poly, train))/1190)*100 # soit 4.285714% d'erreur

#performance en validation
sum(valid$label == predict(svm_model_poly, valid)) #225 bonnes prédiction en validation
(sum(valid$label != predict(svm_model_poly, valid))/255)*100 # soit 11.76471% d'erreur 

#essayons d'améliorer le modèle avec différente valeur de degree

#Essaie de masse 
c = seq(from=2,to=10,by=1)
result = c()
for(i in 1:length(c)){
  svm_model_poly_x <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = c[i], coef0 = 1)
  god_pred_app = sum(train$label == predict(svm_model_poly_x , train))
  god_pred_valid = sum(valid$label == predict(svm_model_poly_x , valid))
  result = cbind(result, c(god_pred_app = god_pred_app, god_pred_valid = god_pred_valid, degree = c[i]))
  
}
result # meilleur compromis apprentissage/validation avec le couple (degree = 4, cost = 1)


#Perfromance test :
svm_model_poly <- svm(label~.,data = train, kernel='polynomial',cost=1,type='C-classification', degree = 4, coef0 = 1)
sum(test$label == predict(svm_model_poly, test)) #230 bonnes prédiction sur le jeu de données de test.
(sum(test$label != predict(svm_model_poly, test))/255)*100 # 9.803922% d'erreur


#prédiction du jeu de donnée compétition avec le meilleur modèle svm

# transformation du data set competition au format HOG.
hog_competition = matrix(0,nrow(competition),cel*cel*ori)
for(i in 1:nrow(competition)){hog_competition[i,] = HOG(matrix(as.numeric(competition[i,1:1024]), nrow = 32, byrow = T), cells = cel, orientations = ori)}
hog_competition = data.frame(hog_competition)# conversion data frame

#utilisation du modèle retenu pour prédire les données compétitions
pred_svm = predict(svm_model_poly, hog_competition, type = "class")
write.csv(data.frame("Id"=c(1:500), "label"=pred_svm), file ="./mespredictions_SVM_HOG_best.csv", row.names = F)


# 5) Regression logistique

#changement des labels 
dataset = change_label_RegLog(dataset)

# Tranformation des images du dataset avec les nouvelles valeurs de cell et ori
hog_data = matrix(0,nrow(dataset),cel*cel*ori)
for(i in 1:nrow(dataset)){hog_data[i,] = HOG(matrix(as.numeric(dataset[i,2:1025]), nrow = 32, byrow = T), cells = cel, orientations = ori)}
hog_data  = data.frame("label" = as.factor(dataset$label), hog_data) # Pour rendre la cible catégorielle


#séparation apprentissage/validation/test
set.seed(20) 
nall = nrow(hog_data) #nombre total de ligne
ntrain = floor(0.7 * nall) #nombre de ligne du train set: 70% 
nvalid = floor(0.15 * nall) #nombre de ligne du val set: 15%
ntest = nall - ntrain - nvalid #nombre de ligne pour test set
index = sample(nall) # permutation aléatoire

train = hog_data[index[1:ntrain],] # création jeu d'apprentissage
valid = hog_data[index[(ntrain+1):(ntrain+nvalid)],] # création jeu de de validation
test = hog_data[index[(ntrain+nvalid+1):(ntrain+nvalid+ntest)],] # création jeu de test

# taille des ensembles
dim(train) #1190 lignes et 73 colonnes
dim(valid) #255 lignes et 73 colonnes
dim(test) #255 lignes et 73 colonnes



# modèle
reg_log_hog = learn_nn(train[,2:73], train$label, 0, 100, c(72, 7))


# prédiction apprentissage 
output = calcul_sorties(reg_log_hog, train[ ,2:73])
v = sapply(c(1:1190), function (x){
  which.max(output[, x])
}
)
v

sum(v == train$label) # 1016 bonnes prédictions
(sum(v != train$label)/1190)*100 #soit  14.62185% d'erreur


#performance validation
output = calcul_sorties(reg_log_hog, valid[ ,2:73])
v = sapply(c(1:255), function (x){
  which.max(output[, x])
}
)
v

sum(v == valid$label) # 206 bonnes prédictions
(sum(v != valid$label)/255)*100 #soit 19.21569% d'erreur

#performance test
output = calcul_sorties(reg_log_hog, test[ ,2:73])
v = sapply(c(1:255), function (x){
  which.max(output[, x])
}
)
v

sum(v == test$label) # 206 bonnes prédictions
(sum(v != test$label)/255)*100 #soit 18.03922% d'erreur


# 6) Reseaux de neurones

dataset = read.table("./arabic_dataset.txt", header = T)
#changement des labels 
dataset = change_label_RN(dataset)
dataset$label = as.factor(dataset$label)

# Tranformation des images du dataset avec les nouvelles valeurs de cell et ori
cel = 3
ori = 8
hog_data = matrix(0,nrow(dataset),cel*cel*ori)
dim(hog_data)
for(i in 1:nrow(dataset)){hog_data[i,] = HOG(matrix(as.numeric(dataset[i,2:1025]), nrow = 32, byrow = T), cells = cel, orientations = ori)}
hog_data  = data.frame("label" = as.factor(dataset$label), hog_data)
head(hog_data)

# Séparation  Apprentissage/Test
dim(hog_data)
set.seed(20)
nall = nrow(hog_data) #total number of rows in data
ntrain = floor(0.80 * nall)
ntest = floor(0.20* nall)
index = sample(nall)

train_x = hog_data[index[1:ntrain],2:73] # ensemble d'apprentisssage
train_labels = hog_data[index[1:ntrain],1] # labels d'apprentissage

head(train_x)
head(train_labels)

test_x = hog_data[index[(ntrain+1):nall],2:73] # ensemble de test
test_labels = hog_data[index[(ntrain+1):nall],1] # labels de test

head(test_x)
head(test_labels)

train_x = matrix(unlist(train_x), ncol = 72)
test_x = matrix(unlist(test_x), ncol = 72)

head(train_x)
head(test_x)

# propositionnalisation des vecteurs
train_y = to_categorical(train_labels) 
test_y = to_categorical(test_labels)


# test des ensembles
train_y # propositionnalisation => OK
head(test_y)


#modèle 

model <- keras_model_sequential()
model %>%
  layer_dense(units = 100, input_shape =72, activation = 'relu') %>%
  layer_dense(units = 80, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 7, activation = 'softmax') 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


#modèle
history <- model %>% fit(train_x, train_y, epochs = 600, batch.size = 10, validation_split = 0.2, view_metrics=F)
#callbacks = list(callback_early_stopping("val_loss", patience = 100)

#visualisation
plot_model_loss(history)
plot_model_accuracy(history)

model %>% fit(train_x, train_y, epochs = 600, batch.size = 10, validation_split = 0.2, callbacks = list(callback_early_stopping("val_loss", patience = 100)), view_metrics=F)
#proba de chaque valeur pour tous les indiv
pred = predict(model, test_x)
head(pred)

# prediction (label avec plus forte proba)
v = sapply(c(1:340), function (x){
  which.max(pred[x, ])
} 
)
v

# depropositionalisation
test_y_de = sapply(c(1:340), function (x){
  which.max(test_y[x, ])
} 
)
test_y_de

#performance
sum(v == test_y_de) # 298 bonnes prédictions
(sum(v != test_y_de)/340)*100 # soit 12.35294% d'erreur

# evaluation accuracy
model%>%evaluate(test_x, test_y) #loss: 1.2782 - accuracy: 0.8765


# essaie d'un second modèle 
history <- model %>% fit(train_x, train_y, epochs = 1000, batch.size = 256, validation_split = 0.2, view_metrics=F)
plot_model_loss(history)
plot_model_accuracy(history)

model %>% fit(train_x, train_y, epochs = 1000, batch.size = 256, validation_split = 0.2, callbacks = list(callback_early_stopping("val_loss", patience = 200)), view_metrics=F)
#proba de chaque valeur pour tous les indiv
pred = predict(model, test_x)
head(pred)

# prediction (label avec plus forte proba)
v = sapply(c(1:340), function (x){
  which.max(pred[x, ])
} 
)
v

# depropositionalisation
test_y_de = sapply(c(1:340), function (x){
  which.max(test_y[x, ])
} 
)
test_y_de

#performance
sum(v == test_y_de) # 295 bonnes prédictions
(sum(v != test_y_de)/340)*100 # soit 13.23529% d'erreur

# evaluation accuracy
model%>%evaluate(test_x, test_y) #loss: 2.2353218 - accuracy: 0.8705


# meilleur
history <- model %>% fit(train_x, train_y, epochs = 600, batch.size = 10, validation_split = 0.2, callbacks = list(callback_early_stopping("val_loss", patience = 100)), view_metrics=F)


#prediction
competition = read.table("./competition.txt", header = T)
hog_competition = matrix(0,nrow(competition),cel*cel*ori)
for(i in 1:nrow(competition)){hog_competition[i,] = HOG(matrix(as.numeric(competition[i,1:1024]), nrow = 32, byrow = T), cells = cel, orientations = ori)}
hog_competition = data.frame(hog_competition)

hog_competition = matrix(unlist(hog_competition), ncol = 72)
pred_rx_neu_hog = predict(model, hog_competition, type = "class")
pred_rx_neu_hog = sapply(c(1:500), function (x){
  which.max(pred_rx_neu_hog[x, ])
} 
)

pred_rx_neu_hog
#Remise des labels à l'etat initial
pred_rx_neu_hog = change_predictions_RN(pred_rx_neu_hog)
write.csv(data.frame("Id"=c(1:500), "label"=pred_rx_neu_hog), file ="./mespredictions_rx_neu_hog_cell3_ori8.csv", row.names = F)
