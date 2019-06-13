rm(list=ls())

setwd("C:/Users/madsj/Dropbox/AU/Thesis/Credit risk/Data")
clean_data= read.csv("fulldatafinal.csv",sep=",")
set.seed(1)

clean_data$Status = as.factor(clean_data$Status) # Convert to factor
clean_data$ln.Revenue..t.1 = as.numeric(clean_data$ln.Revenue..t.1)


da=balanced.data[,-1]

###### Remove outliers (Does not work yet) 
library(funModeling)
# data_no_outliers=prep_outliers(data=data, type='set_na', bottom_percent = 0.01, top_percent  = 0.01, method = "bottom_top")

####### Remove NA's 
# clean_data = na.omit(data)
x_corr= balanced.data1[[1]][,-1]
mycorr = cor(x_corr)


#### plots
data(clean_data)
plot(clean_data$Netprofit.Equity ,
     clean_data$LTL..TA, col = clean_data$Status,
     xlim=c(-10,10),
     ylim=c(-2,6))

# https://www.r-bloggers.com/lstm-with-keras-tensorflow/

#####   Train and Test data split 
library(caTools)

split <- sample.split(clean_data$Status, SplitRatio = 0.5)

data_train <- subset(clean_data, split == TRUE)
data_test <- subset(clean_data, split == FALSE)

summary(data_train)

data_scaled_train <- subset(clean_data, split == TRUE)
data_scaled_test <- subset(clean_data, split == FALSE)

#########             SMOTE data generating process 
library(DMwR)

balanced.data <- SMOTE(Status ~., data_train, perc.over = 1400, perc.under=110,  k = 3)
summary(balanced.data)

library(UBL)
# SMOTE + TOMEK
  
  # Tomek-link removes only examples from class summer in every link they appear
  balanced.data1 <- TomekClassif(Status~., balanced.data, dist = "Euclidean",
                          Cl = "Active", rem = "both")
summary(balanced.data1[[1]])

# SMOTE + EEN
library(unbalanced)

x1= balanced.data[,-1]

balanced2= ubENN(X=x1, Y=Status, k = 3, verbose = TRUE)

#### plots
data(balanced.data1[[1]])
plot(balanced.data1[[1]]$Netprofit.Equity ,
     balanced.data1[[1]]$LTL..TA, col = balanced.data1[[1]]$Status,
     xlim=c(-10,10),
     ylim=c(-2,6))

# balanced.dataM=as.matrix(balanced.data)

balanced.dataM = as.matrix(balanced.data)


############------------------Logistic regression--------------------############# 

library(caret)  

model <- glm(Status~.,data=balanced.data1[[1]], family = binomial)
summary(model)
varImp(model)

#Prediction 
predict_log <- predict(model, data_test, type = 'prob')

# Create Confusion Matrix
table(true=data_test$Status,predictions=predict_log >0.5)

#ROC curve 
library(ROCR)
ROCRpred <- prediction(predict, data_test$Status)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))


##############---------------- Classification tree ---------------###########

library(tree)
# Full clean.data
tree.full=tree(Status~.,balanced.data1[[1]])
summary(tree.full)
plot(tree.full)
text(tree.full,pretty=0)
tree.full

# train 
tree.model=tree(Status~., balanced.data1[[1]])
# predict
tree.pred=predict(tree.model,data_test,type="class")
#table
table(data_test$Status, tree.pred)
# CV
cv.tree1=cv.tree(tree.model,FUN=prune.misclass)
names(cv.tree1)
cv.tree1
par(mfrow=c(1,2))
plot(cv.tree1$size,cv.tree1$dev,type="b")
plot(cv.tree1$k,cv.tree1$dev,type="b")
#Prune 
prune.tree=prune.misclass(tree.model,best=5)
plot(prune.tree)
text(prune.tree,pretty=0)
tree.pred=as.data.frame(predict(prune.tree,newdata=data_test,type="vector"))
table(data_test$Status, tree.pred$default >0.5)
tree.pred1=tree.pred$default

hej= (unclass(balanced.data1[[1]]$Status)-1)

############## --------------- Bagging & Random Forest ----------- ################

library(randomForest)

######## Bagging 

bag.SME <- randomForest(Status~.,data=balanced.data1[[1]],mtry=87,importance=TRUE)
# List the content of the container
bag.SME
# Make predictions and plot comparison
yhat.bag <- as.data.frame(predict(bag.SME,newdata=data_test, type="prob"))
yhat.bag1 = yhat.bag$default
#table 
table(data_test$Status, yhat.bag1 > 0.5)

importance(bag.SME)
# Plot
varImpPlot(bag.SME)
######### RandomForest

rf.SME <- randomForest(Status~.,data=balanced.data1[[1]],mtry=10,importance=TRUE)
# List the content of the container
rf.SME
# Make predictions and plot comparison
yhat.rf <- as.data.frame(predict(rf.SME,newdata=data_test, type="prob"))
#table 
yhat.rf1 = yhat.rf$default
table(data_test$Status, yhat.rf1 > 0.5)

importance(rf.SME)
# Plot
varImpPlot(rf.SME)


############## -------------- GBM  BOOSTING-------------------###############
library(gbm)

# Status <- ifelse(Status == "default", 1, 0)
# Status_gbm = as.factor(Status)

# Running the algorithm with 15.000 trees, max tree is 6 and learning rate equal to 0.1
boost.SME <- gbm((unclass(Status)-1) ~.,data=balanced.data1[[1]], distribution= "bernoulli", n.trees=8000,interaction.depth=6, shrinkage =0.1, cv.folds=5)

#boost.SME.ada <- gbm((unclass(Status)-1) ~.,data=balanced.data1[[1]], distribution= "adaboost", n.trees=8000,interaction.depth=6, shrinkage =0.1, cv.folds=5)

best.iter <- gbm.perf(boost.SME,method="cv")

#best.iter.ada <- gbm.perf(boost.SME.ada,method="cv")

legend("topright",c("CV error", "Train Error", "Optimal # of Trees"),col=c("green","black","blue"), lty=c(1,1,2))

# The summary reports the relative influence (plot as well as the numbers)
summary(boost.SME,best.iter)
#summary(boost.SME.ada,best.iter)

# Predictions with the optimal nr. of trees 
yhat.boost <- predict(boost.SME,newdata=data_test,n.trees=best.iter, type="response")

table(true=data_test$Status, predicted=yhat.boost >0.5)

#yhat.boost.ada <- predict(boost.SME.ada,newdata=data_test,n.trees=best.iter.ada)

#table(true=data_test$Status, predicted=yhat.boost.ada >0.5)

###########-------------------------Support Vector Machines------------------------############# 
library(e1071)
####### DATA scaling for SVM 

# Create seperate x matrix for training data
x = model.matrix(Status~.,balanced.data1[[1]])[,-1]

#Scale only x variables between 0 and 1
maxs <- apply(x, 2, max)
mins <- apply(x, 2, min)
scaled_x <- as.data.frame(scale(x, center = mins, scale = maxs - mins))

# Data frame to contain all x variables in scaled form along with the y variable
dat=data.frame(x=scaled_x,y=balanced.data1[[1]]$Status)

# Create seperate x matrix for test data
x_test= model.matrix(data_test$Status~.,data_test)[,-1]

#Scale only x variables between 0 and 1
maxs <- apply(x_test, 2, max)
mins <- apply(x_test, 2, min)
scaled_x_test <- as.data.frame(scale(x_test, center = mins, scale = maxs - mins))

# Data frame to contain all x variables in scaled form along with the y variable
dat_test=data.frame(x=scaled_x_test,y=data_test$Status)

# Tuning polynomial
svmfit_pol_tune=tune.svm(y~., data=dat, degree=3, gamma=2^(-7:-1), cost=2^(1:12))
summary(svmfit_pol_tune)

# Tuning Radial Kernel 
svmfit_rad_tune=tune.svm(y~., data=dat, kernel="radial", gamma=(0.25), cost=c(100, 1000), proabbilities=TRUE) # Best 0.05, 10
summary(svmfit_rad_tune)

# put in the optimal values from trhe tuning model. 0.1, 100
svmfit_pol=svm(y~., data=dat, kernel="polynomial", degree=3,  gamma=0.25, cost=16, probability=TRUE)
# g=0.2, c=20 giver alle default, g=0,15 c=10 giver alle avtive, g=0.05, c=5 giver bedste resultat.    
svmfit_rad=svm(y~., data=dat, kernel="radial",  gamma=0.25, cost=100, probability=TRUE)

summary(svmfit_pol)
#summary(svmfit_rad)
# Predict test set with the fitted model.
dat_test_x=dat_test[,-88] # Remove the classification feature from the test set. 

pred_pol=predict(svmfit_pol,newdata=dat_test_x, probability=TRUE)
pred_pol_prob=as.data.frame(attr(pred_pol, "probabilities"))
pred_pol_prob1 = pred_pol_prob$default

pred_rad=predict(svmfit_rad,newdata=dat_test_x, probability=TRUE)
pred_rad_prob=as.data.frame(attr(pred_rad, "probabilities"))
pred_rad_prob1 =pred_rad_prob$default

#table(true=data_test$Status, pred_rad)
table(true=data_test$Status, pred_pol_prob1 > 0.5)


plot(svmfit_pol, dat, dat$x.EBITDA.Financial.expenses ~ dat$x.Netprofit.Employees,
     slice = list(dat$x.EBITDA.Total.Assets = 0.5, dat$x.Equity.Total.Assets = 0.5))
plot.svm(svmfit_pol, data=dat)

######## ------------ LSTM ---------#############
library(devtools)
library(keras)
library(kerasR)
library(reticulate)
library(tidyverse)
library(caret)
library(magrittr)
library(fastDummies)

## mix the data - find out what is 1 and what i 0 before. Skal stå i rigtig rækkefølge (behøves måske ikke) (0,1 er default)
df2 <- balanced.data1[[1]][sample(nrow(balanced.data1[[1]])),]

# x+y scaling for training data (Skal bruge kerasR)
dummy_data <- fastDummies::dummy_cols(df2,remove_first_dummy = TRUE)
finaldf <- dummy_data[,-1]

X_train <- finaldf %>% 
  select(-Status_default) %>% 
  scale()

y_train <- to_categorical(finaldf$Status_default)

# test

dummy_data_test <- fastDummies::dummy_cols(data_test,remove_first_dummy = TRUE)
finaldf_test <- dummy_data_test[,-1]


X_test <- finaldf_test %>% 
  select(-Status_default) %>% 
  scale()

y_test <- to_categorical(finaldf_test$Status_default)

# create 3d tensor for training

data_yo = as.vector(t(X_train))
dim(data_yo)
data_yo1 = split(data_yo, ceiling(seq_along(data_yo)/29))
data_yo11=as.data.frame(do.call(rbind, data_yo1))

dim(data_yo11)

Sliced <- aperm(`dim<-`(t(data_yo11), list(29, 3, 18405)), c(3, 2, 1))
dim(Sliced)

# The  Multi layer perceptron model 
library(keras)  
model_mlp %>% plot_model()

model_mlp <- keras_model_sequential()

model_mlp %>% 
  layer_dense(units = 88, activation = 'relu', input_shape = ncol(X_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 22, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 4, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'sigmoid')

history <- model_mlp %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('binary_accuracy')
)

model_mlp %>% fit(
  X_train, y_train, 
  epochs = 30, 
  batch_size = 5,
  validation_split = 0.2
)

model_mlp %>% evaluate(X_test, y_test)  
 
predictions_mlp <- model %>% predict_classes(X_test)

Y_pred_mlp = round(predictions_mlp)
# Confusion matrix
CM_mlp = table(true=finaldf_test$Status_default, predtictions=Y_pred_mlp)
print(CM_mlp)

pred_mlp1= as.data.frame(predictions_mlp)
pred_mlp2= pred_mlp1$V2
table(true=data_test$Status, pred_mlp2 >0.5)


# The LSTM model 

data_dim = 29
timesteps = 3
num_classes = 2

# expected input data shape: (batch_size, timesteps, data_dim)

### LSTM Model 
library(keras)
# best is 0.4 0.4 dropout, 50 epochs, recurrent = 0.4 = AUC 0.78, recurent= 0.3 = 0.7864
library(deepviz)
library(magrittr)

model_lstm %>% plot_model()


model_lstm <- keras_model_sequential()

model_lstm %>%

  layer_lstm(units=29, return_sequences="True",input_shape=c(timesteps,data_dim), recurrent_dropout = 0.3) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_lstm(units=29, return_sequences="True", recurrent_dropout = 0.3) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_lstm(units=29) %>%
  layer_dense(units=2, activation='sigmoid')
  
history_lstm <- model_lstm %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model_lstm %>% fit(
  Sliced, y_train, 
  epochs = 60, 
  batch_size = 5,
  validation_split = 0.2
)

# x test data shape 
data_yo_test = as.vector(t(X_test))
dim(data_yo_test)
data_yo1_test = split(data_yo_test, ceiling(seq_along(data_yo_test)/29))
data_yo11_test=as.data.frame(do.call(rbind, data_yo1_test))

dim(data_yo11_test)
# create 3d tensor for training

Sliced_test <- aperm(`dim<-`(t(data_yo11_test), list(29, 3, 21082)), c(3, 2, 1))
dim(Sliced_test)


model_lstm %>% evaluate(Sliced_test, y_test)  

predictions_lstm <- model_lstm %>% predict_classes(Sliced_test)

predictions_lstm1 <- model_lstm %>% predict(Sliced_test)
pred_lstm1= as.data.frame(predictions_lstm1)
pred_lstm2= pred_lstm1$V2
# Confusion matrix
CM = table(true=finaldf_test$Status_default, predtictions=Y_pred)
print(CM)
table()

table(true=data_test$Status, pred_lstm2 >0.5)
  
#### ROC, AUC, P-R
library(pROC)
library(AUC)

dude= data_test$Status   
roc1 <- roc(dude,predict_log)
roc2 <- roc(dude,yhat.bag1)
roc3 = roc(dude, yhat.rf1)
roc4 = roc(dude, yhat.boost)
roc5 <- roc(dude,tree.pred1)
roc6 = roc(dude, pred_pol_prob1)
roc7=roc(dude,pred_mlp2)
roc8=roc(dude,pred_lstm2)


rocplot <- ggroc(list("Logistic" = roc1, "Bagging"=roc2, "RandomForest"=roc3, 
                     "Gradient Boosting"=roc4, "Class tree"=roc5 ,"SVM"=roc6, "MLP"=roc7, "LSTM"=roc8 ), legacy.axes = TRUE)
rocplot + geom_segment(aes(x = 0, xend = 1, y = 0 , yend = 1), color = 24, linetype = 2)
# add confidence intervals til auc table. 
ci1 = ci(roc1,method=c("bootstrap"),  boot.n = 1000)
ci2 = ci(roc2,method=c("bootstrap"),  boot.n = 1000)
ci3 = ci(roc3,method=c("bootstrap"),  boot.n = 1000)
ci4 = ci(roc4,method=c("bootstrap"),  boot.n = 1000)
ci5 = ci(roc5,method=c("bootstrap"),  boot.n = 1000)
ci6 = ci(roc6,method=c("bootstrap"),  boot.n = 1000)
ci7 = ci(roc7,method=c("bootstrap"),  boot.n = 1000)
ci8 = ci(roc8,method=c("bootstrap"),  boot.n = 1000)

  
library(ggplot2)
library(precrec)
l1=c(pred_lstm2)
l2=c(predict_log)
l3=c(yhat.boost)
l5=c(yhat.rf1)
l4=c(dude)

scores1 <- join_scores(l1, l2, l3, l5)
msmdat2 <- mmdata(scores1, l4, modnames = c("LSTM", "Logistic", "Boosting", "Random Forest"))
mscurves <- evalmod(msmdat2)
autoplot(mscurves)

# density plot lav samme rækkefølge hvis inkluder PR curve. 
kk <- data.frame(dens = c(predict_log, pred_lstm2, yhat.boost), lines = rep(c("Logistic", "LSTM", "Boosting"), each=21082))
ggplot(kk, aes(x = dens, fill=lines)) + geom_density(alpha = 0.5)



############# Profit 
#merge 
logframe = cbind.data.frame(dude,predict_log)
lstmframe =cbind.data.frame(dude,pred_lstm2)

###-----logistic----###
#AA+ 0-0.001
logAAA = sum( logframe$predict_log > 0 & logframe$predict_log < 0.0001) #=609
logAAA.Active =sum( predict_log > 0 & predict_log < 0.0001 & logframe$dude =="Active") #=605
#A
logA = sum( logframe$predict_log > 0.0001 & logframe$predict_log < 0.005) #=588
logA.Active =sum( predict_log > 0.0001 & predict_log < 0.005 & logframe$dude =="Active") #=585
#BBB 
logBBB = sum( logframe$predict_log > 0.005 & logframe$predict_log < 0.01) #=245
logBBB.Active =sum( predict_log > 0.005 & predict_log < 0.01 & logframe$dude =="Active") #=243
#BB
logBB = sum( logframe$predict_log > 0.01 & logframe$predict_log < 0.03) #=620
logBB.Active =sum( predict_log > 0.01 & predict_log < 0.03 & logframe$dude =="Active") #615
#B
logB = sum( logframe$predict_log > 0.03 & logframe$predict_log < 0.07) #=910
logB.Active =sum( predict_log > 0.03 & predict_log < 0.07 & logframe$dude =="Active")  # 903
#CCC or lower
logCCC = sum( logframe$predict_log > 0.07 & logframe$predict_log < 0.14) #=1617
logCCC.Active =sum( predict_log > 0.07 & predict_log < 0.14 & logframe$dude =="Active")  # 1599

###-----LSTM----###
#AA+ 0-0.001
lstmAAA = sum(pred_lstm2 > 0 & pred_lstm2 < 0.0001) #=4031
lstmAAA.Active =sum( pred_lstm2> 0 & pred_lstm2 < 0.0001 & dude =="Active") #=4017
#A
lstmA = sum(pred_lstm2 > 0.0001  & pred_lstm2 < 0.005) #=5672
lstmA.Active =sum( pred_lstm2> 0.0001 & pred_lstm2 < 0.005 & dude =="Active") #=5618
#BBB
lstmBBB = sum(pred_lstm2 > 0.005  & pred_lstm2 < 0.01) #=819
lstmBBB.Active =sum( pred_lstm2> 0.005 & pred_lstm2 < 0.01 & dude =="Active") #=801
#BB
lstmBB = sum(pred_lstm2 > 0.01  & pred_lstm2 < 0.03) #=1327
lstmBB.Active =sum( pred_lstm2> 0.01 & pred_lstm2 < 0.03 & dude =="Active") #=1307
#B
lstmB = sum(pred_lstm2 > 0.03  & pred_lstm2 < 0.07) #=889
lstmB.Active =sum( pred_lstm2> 0.03 & pred_lstm2 < 0.07 & dude =="Active") #=868
#CCC or lower
lstmCCC = sum(pred_lstm2 > 0.07  & pred_lstm2 < 0.14) #=700
lstmCCC.Active =sum( pred_lstm2> 0.07 & pred_lstm2 < 0.14 & dude =="Active") #=682

###-----Boosting----###
#AA+ 
boostAAA = sum(yhat.boost > 0 & yhat.boost < 0.0001) #=5914
boostAAA.Active =sum( yhat.boost> 0 & yhat.boost < 0.0001 & dude =="Active") #=5891
#A
boostA = sum(yhat.boost > 0.0001 & yhat.boost < 0.005) #=7422
boostA.Active =sum( yhat.boost> 0.0001 & yhat.boost < 0.005 & dude =="Active") #=7330
#BBB
boostBBB = sum(yhat.boost > 0.005 & yhat.boost < 0.01) #=1186
boostBBB.Active =sum( yhat.boost> 0.005 & yhat.boost < 0.01 & dude =="Active") #=1157
#BB
boostBB = sum(yhat.boost > 0.01 & yhat.boost < 0.03) #=1711
boostBB.Active =sum( yhat.boost> 0.01 & yhat.boost < 0.03 & dude =="Active") #=1660
# B 
boostB = sum(yhat.boost > 0.03 & yhat.boost < 0.07) #=1094
boostB.Active =sum( yhat.boost> 0.03 & yhat.boost < 0.07 & dude =="Active") #=1062
# CCC
boostCCC = sum(yhat.boost > 0.07 & yhat.boost < 0.14) #=813
boostCCC.Active =sum( yhat.boost> 0.07 & yhat.boost < 0.14 & dude =="Active") #=770


                     