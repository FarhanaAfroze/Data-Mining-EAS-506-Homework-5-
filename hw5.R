library(DataExplorer)
library(ggcorrplot)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(moments)
library(caTools)
library(MASS)
library(Hmisc)
library(Metrics)
library(leaps)
library(sp)
library(class)
library(caret)
library(tidyverse)
library(rsample)
install.packages("rpart")
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("randomForest")
library(randomForest)

load("vehicle.RData")
View(vehicle)
head(vehicle)

#first explore dataset
dim(vehicle)  #564 rows and 20 columns
sum(is.na(vehicle)) #no missing values found

count_digit = table(vehicle$classdigit)
count_digit

colors=c("blue","red", "yellow", "green")
col=colors
pie(count_digit, labels = count_digit, main = "Class Digits",col=colors)
box()
legend("topright", c("1","2","3","4"), cex = 0.8, fill = col)


count_class = table(vehicle$class)
count_class

colors=c("blue","yellow", "red", "green")
col=colors
pie(count_class, labels = count_class, main = "Class Digits",col=colors)
box()
legend("topright", c("bus","opel","saab","van"), cex = 0.8, fill = col)


#Next we split data into training and testing set
set.seed(1)
splits = createDataPartition(y = vehicle$class,  p = 0.8, list = FALSE)
training <- vehicle[splits, ]
testing <- vehicle[-splits, ]
View(training)
dim(training)
View(testing)
dim(testing)



# next we will create fully grown trees
#next I will drop class digit column
#as.numeric(training$classdigit)
#View(training)

#is.factor(training$classdigit)
#training$classdigit = as.numeric(as.character(training$classdigit))
#is.factor(training$classdigit)

drop = subset(training, select = -c(classdigit))
View(drop)

drop_test = subset(testing, select = -c(classdigit))
View(drop_test)

set.seed(2)
model.control <- rpart.control(minsplit = 50, xval = 5, cp = 0)
tree_model = rpart(class~., data = drop, method = "class", control = model.control)
#tree_model
#tree_model = rpart(class~., data = drop, method = "class")

par(xpd = NA)
plot(tree_model)
text(tree_model, digits = 3, cex = .8)

summary(tree_model)
names(tree_model)
plotcp(tree_model)
printcp(tree_model)

#next we will predict data
tree_predict = predict(tree_model, drop_test, type = "class")
mean(tree_predict == drop_test$class)
mean(tree_predict != drop_test$class)

#we would like to see confusion matrix of classes
table(pred = tree_predict, true = drop_test$class)

# next pruning tree
printcp(tree_model)


cp_ind = which.min(tree_model$cptable[,"xerror"])
cp_ind
cp_val = tree_model$cptable[cp_index,"CP"]
cp_val

tree_model$cptable
plot(tree_model$cptable[,4], main = "cp for model selection", ylab = "cv error" )

cp_min = which.min(tree_model$cptable[,4])
cp_min


prune_model = prune(tree_model, cp = tree_model$cptable[cp_min,1])
prune_model
plot(prune_model, compress = T)
text(prune_model, cex = .8)


tree_prune_predict = predict(prune_model, drop_test, type = "class")
mean(tree_prune_predict == drop_test$class)


# PROBLEM #2
load("prostate.RData")
View(prostate)
dim(prostate)
sum(is.na(prostate))

# data visualization 
plot_correlation(prostate[c(-10)])

# Now getting training and testing data
# In the data set we have train column which includes true and false, and for the true column it would
# used for training and for false column it would be use for testing set.

select1 = c(TRUE)

col_select_train = prostate[prostate$train %in% select1, ]
View(col_select_train)
dim(col_select_train)

select2 = c(FALSE)

col_select_test = prostate[prostate$train %in% select2, ]
View(col_select_test)
dim(col_select_test)

# Next we will drop the "Train" column from both trainging and testing set

trains = col_select_train[c(-10)] #dropping train column
View(trains)
head(trains)

tests = col_select_test[c(-10)] #dropping test column
View(tests)


# normalizing the data

normalize <- function(x) 
{
  return ((x - min(x)) / (max(x) - min(x)))
}

trains = as.data.frame(lapply(trains[1:9], normalize))
View(trains)

tests = as.data.frame(lapply(tests[1:9], normalize))
View(tests)
head(tests)

#compute best subset selection
best_subset_model = regsubsets(lpsa~., data = trains, nvmax = 8)
summary(best_subset_model)
best_subset_model

coef(best_subset_model,8)

# So now We will do model selection
# AIC/CP and BIC

summary_best = summary(best_subset_model)
summary_best

which.min(summary_best$cp)
which.min(summary_best$bic)

plot(best_subset_model, scale = "Cp")
plot(best_subset_model, scale = "bic")

plot(summary_best$cp, xlab = "Number of variables", ylab = "CP", main = "cp model")
points(7, summary_best$cp[7], col= "red", cex = 1.5, pch = 20)

#doing 7 variable model in linear regression
cp_model_linear = lm(lpsa~pgg45+gleason+lcp+svi+lbph+age+lweight+lcavol, data = trains)
summary(cp_model_linear)
pred_cp_model_linear = predict(cp_model_linear, tests)
mse_error_cp = mse(pred_cp_model_linear, tests$lpsa)
mse_error_cp

plot(summary_best$bic, xlab = "Number of variables", ylab = "BIC")
points(2, summary_best$bic[2], col = "blue", cex = 1.5, pch = 19)

bic_model_linear = lm(lpsa~lweight+lcavol, data = trains)
summary(bic_model_linear)
pred_bic_model_linear = predict(bic_model_linear, tests)
mse_error_bic = mse(pred_bic_model_linear, tests$lpsa)
mse_error_bic


# 5 fold cross validation

predict.regsubsets = function(object, newdata, id, ...) {   # source: https://stackoverflow.com/questions/37314192/error-in-r-no-applicable-method-for-predict-applied-to-an-object-of-class-re
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}



set.seed(1)
k = 5;

five_fold = sample(1:k, nrow(trains), replace = TRUE)
five_fold

cross_errors = matrix(NA, k, 8)

for (i in 1:k)
{
  best_subset = regsubsets(lpsa~., data = trains[five_fold!=i, ], nvmax = 8)
  
  for (j in 1:8)
  {
    predictions = predict(best_subset, trains[five_fold==i, ], id=j)
    cross_errors[i,j] = mean((trains$lpsa[five_fold==i]-predictions)^2)
  }
  
}
cross_errors

cross_error_ave = apply(cross_errors, 2, mean)
cross_error_ave

minimum = which.min(cross_error_ave)
minimum

# model 7 is the best


# ten fold cross validation


set.seed(1)
k = 10;

ten_fold = sample(1:k, nrow(trains), replace = TRUE)
ten_fold

cross_errors = matrix(NA, k, 8)

for (i in 1:k)
{
  best_subset = regsubsets(lpsa~., data = trains[ten_fold!=i, ], nvmax = 8)
  
  for (j in 1:8)
  {
    predictions = predict(best_subset, trains[ten_fold==i, ], id=j)
    cross_errors[i,j] = mean((trains$lpsa[ten_fold==i]-predictions)^2)
  }
  
}
cross_errors

  
cross_error_ave = apply(cross_errors, 2, mean)
cross_error_ave

minimum = which.min(cross_error_ave)
minimum


#model 7 is the best

# Bootstrap 
install.packages("bootstrap")
library(boot)
library(bootstrap)

beta.fit <- function(X,Y)
{
  lsfit(X,Y)
}

beta.predict<- function(fit, X)
{
  cbind(1,X)%*%fit$coef
}

sq.error<- function(Y, Yhat)
{
  (Y-Yhat)^2
}
select = summary_best$outmat
error_stores = c()
for (i in 1:8){
  temp <- which(select[i,] == "*")
  res<- bootpred(prostate[,temp], prostate$lpsa, nboot = 100, theta.fit = beta.fit, theta.predict = beta.predict,  err.meas = sq.error)
  error_stores <- c(error_stores, res[[3]])
}
error_stores

# PROBLEM ##3 ########

wine <- read.csv("wine.data.txt", header = FALSE) 
View(wine)  

#we would like to give column names to some names
#data visualization and exploration

colnames(wine) <- c('Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflavanoids','pronthocyanins', 'color', 'Hue', 'Diluation','proline')
wine$Type <- as.factor(wine$Type)
head(wine)
sum(is.na(wine))

levels(wine$Type) <- c("Barolo", "Grigonlino", "Barbera")
View(wine)
head(wine)
dim(wine)
wine_culitvars <- table(wine$Type)
wine_culitvars

par(mfrow=c(1.5,1.5))
colors=c("blue","red", "yellow")
col=colors
pie(wine_culitvars, labels = wine_culitvars, main = "types of culitvars",col=colors)
box()
legend("topright", c("Barolo","Grigonlino","Barbera"), cex = 0.8, fill = col)
  
plot_histogram(wine)

# Next data splitting into training and testing 
set.seed(1)
wine_split = createDataPartition(y = wine$Type, p = 0.8, list = FALSE)
wine_training <- wine[wine_split, ]
wine_testing <- wine[-wine_split, ]
View(wine_training)
View(wine_testing)
dim(wine_training)
dim(wine_testing)

#construct an appropriate classification tree
model.control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
wine_tree = rpart(Type~., data = wine_training, method = "class", control = model.control)
par(xpd = NA)
plot(wine_tree)
text(wine_tree,cex = .8)

summary(wine_tree)
names(wine_tree)
plotcp(wine_tree)
printcp(wine_tree)

#next we will predict data
wine_predict = predict(wine_tree, wine_testing, type = "class")
wine_predict
mean(wine_predict == wine_testing$Type)
mean(wine_predict != wine_testing$Type)

#we would like to see confusion matrix of classes
table(pred = wine_predict, true = wine_testing$Type)

# NEXT we will prune our tree

printcp(wine_tree)
cp_in = which.min(wine_tree$cptable[,"xerror"])
cp_in
cp_value = wine_tree$cptable[cp_in,"CP"]
cp_value

wine_tree$cptable
plot(wine_tree$cptable[,4], main = "cp for model selection", ylab = "cv error" )

cp_mini = which.min(wine_tree$cptable[,4])
cp_mini


prune_model1 = prune(wine_tree, cp = wine_tree$cptable[cp_mini,1])
plot(prune_model1, compress = T, main = "prune tree")
text(prune_model1, cex = .8)


wine_prune_predict = predict(prune_model1, wine_testing, type = "class")
mean(wine_prune_predict == wine_testing$Type)
mean(wine_prune_predict != wine_testing$Type)
#giving same result as without pruning tree

# Next apply ensemble technique- random forest 

set.seed(12)
forest_model <- randomForest(Type~., data = wine_training, importance = TRUE)
forest_model


#next we will tune parameter for random forest
new_forest_model <- randomForest(Type~., data = wine_training, ntree = 500, mtry = 2, importance = TRUE)
new_forest_model

forest_predict = predict(new_forest_model, wine_testing, type = "class")
forest_predict

mean(forest_predict == wine_testing$Type)
mean(forest_predict != wine_testing$Type)

table(forest_predict, wine_testing$Type)

#next we will see variable importance
importance(new_forest_model)

#next we will plot varaible important
varImpPlot(new_forest_model, type = 1)
varImpPlot(new_forest_model, type = 2)

#I would like to try different mtry value so can which one doing best

accuracy_holder = c()
for (i in 1:10)
{
  another_forest_model = randomForest(Type~., data = wine_training, ntree = 500, mtry = i, importance = TRUE)
  pred_for_forest = predict(another_forest_model, wine_testing, type = "class")
  accuracy_holder[i] = mean(pred_for_forest == wine_testing$Type)
}

accuracy_holder
optimal_numer_of_predictor_variables = c(1:10)
plot(optimal_numer_of_predictor_variables, accuracy_holder)

#Next construct LDA model

lda_wine_model = lda(Type~., data = wine_training)
lda_wine_model

#now make prediction on testing set
lda_predict = predict(lda_wine_model, wine_testing)
lda_predict

par(mar=c(0.1,0.1,0.1,0.1))
ldahist(data = lda_predict$x[,1], g = wine_testing$Type)
ldahist(data = lda_predict$x[,2], g = wine_testing$Type)

lda_predict$class

mean(lda_predict$class == wine_testing$Type)
mean(lda_predict$class != wine_testing$Type)
table(lda_predict$class, wine_testing$Type)

#would like to see plot 
lda_predict$x
graph_data = cbind(wine_testing, lda_predict$x)
graph_data
ggplot(graph_data, aes(LD1,LD2)) + geom_point(aes(color = Type))

#normalize <- function(x) 
#{
# return ((x - min(x)) / (max(x) - min(x)))
#}

#wines = as.data.frame(lapply(wine[2:14], normalize))
#View(wines)

#type = wine$Type
#View(type)
#wines = cbind(wines, type)
#View(wines)


#PROBLEM #4 extra credit

load("covertype.RData")
View(covertype)
sum(is.na(covertype))
dim(covertype)

forest_classes <- table(covertype$V55)
forest_classes

# I would like to predict Rawah wilderness area's(1) forest cover type. we will select 1(presesnce) rows of rawas wildnerss area

new_covertype = subset(covertype, V11 == 1)
dim(new_covertype)
View(new_covertype)

#splitting the data set into training and testing
set.seed(1)
forest_split = createDataPartition(y = new_covertype$V55, p = 0.75, list = FALSE)
forest_training <- new_covertype[forest_split, ]

forest_training <- forest_training[-c(11:54)]
forest_training$V55 <- as.factor(forest_training$V55)
View(forest_training)
dim(forest_training)

forest_testing <- new_covertype[-forest_split, ]
forest_testing <- forest_testing[-c(11:54)]
forest_testing$V55 <- as.factor(forest_testing$V55)
View(forest_testing)
dim(forest_testing)

#Implementing random forest
set.seed(12)
forest_model_cover <- randomForest(V55~., data = forest_training, importance = TRUE)
forest_model_cover

#training_error and accuracy

forest_train_predict = predict(forest_model_cover, forest_training, type = "class")
mean(forest_train_predict != forest_training$V55)
mean(forest_train_predict == forest_training$V55)

# test error and accuracy
forest_predict_cover = predict(forest_model_cover, forest_testing, type = "class")
forest_predict_cover

mean(forest_predict_cover == forest_testing$V55)
mean(forest_predict_cover != forest_testing$V55)

# confusion matrix of the test set
table(forest_predict_cover, forest_testing$V55)

set.seed(12)
forest_model_cover_new <- randomForest(V55~., data = forest_training, ntree = 500, mtry = 5,importance = TRUE)
forest_model_cover_new

forest_predict_cover_new = predict(forest_model_cover_new, forest_testing, type = "class")
forest_predict_cover_new

mean(forest_predict_cover_new == forest_testing$V55)
mean(forest_predict_cover_new != forest_testing$V55)

table(forest_predict_cover_new, forest_testing$V55)



