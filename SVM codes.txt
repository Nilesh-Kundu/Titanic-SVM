library(rpart)# tree models 
library(caret) # feature selection
library(rpart.plot) # plot dtree
library(ROCR) # model evaluation
library(e1071) # tuning model
library(RColorBrewer)
library(rattle)# optional, if you can't install it, it's okay
library(tree)
library(ISLR)
library(randomForest)
library(rpart.plot)
library(dplyr)


setwd("C:\\Users\\ADMIN\\Desktop\\R Models\\Decision Tree")
Carseats <- read.csv("Titanic.csv")
head(Carseats)
tail(Carseats)
str(Carseats)
summary(Carseats)

Carseats <- Carseats[ -c(1,4,9,11) ]
## Let's also change the labels under the "status" from (0,1) to (normal, abnormal)   
Carseats$Pclass <- as.factor(Carseats$Pclass) 
Carseats$Survived <- factor(Carseats$Survived, levels = c(0, 1),labels = c('No', 'Yes'))

## Check the missing value (if any)
sapply(Carseats, function(x) sum(is.na(x)))

Carseats <- na.omit(Carseats)

## Now you can randomly split your data in to 70% training set and 30% test set   
set.seed(123)
train <- sample(1:nrow(Carseats), round(0.70*nrow(Carseats),0))
test <- -train
training <- Carseats[train,]
testing <- Carseats[test,]
test_Survived <- testing$Survived


##training the SVM model with linear kernel
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)
 
svm_Linear <- train(Survived ~., data = training, method = "svmLinear",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred	
confusionMatrix(table(test_pred, testing$Survived))

##Building SVM linear classifier with different C values
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(Survived ~., data = training, method = "svmLinear",
trControl=trctrl,
preProcess = c("center", "scale"),
tuneGrid = grid,
tuneLength = 10)
svm_Linear_Grid
plot(svm_Linear_Grid)

svm_Linear_Grid

test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
test_pred_grid
confusionMatrix(table(test_pred_grid, testing$Survived))

##SVM Classifier using Non-Linear Kernel
set.seed(3233)
svm_Radial <- train(Survived ~., data = training, method = "svmRadial",
  trControl=trctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10)
svm_Radial
plot(svm_Radial)

test_pred_Radial <- predict(svm_Radial, newdata = testing)
test_pred_Radial
confusionMatrix(table(test_pred_Radial, testing$Survived))

##testing & tuning our classifier with different values of C & sigma
grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
 0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
 C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
 1, 1.5, 2,5))
 set.seed(3233)
svm_Radial_Grid <- train(Survived ~., data = training, method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneGrid = grid_radial,
                    tuneLength = 10)
 
svm_Radial_Grid
plot(svm_Radial_Grid)

test_pred_Radial_Grid <- predict(svm_Radial_Grid, newdata = testing)
test_pred_Radial_Grid
confusionMatrix(table(test_pred_Radial_Grid, testing$Survived))
