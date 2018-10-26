library(hydroGOF)
library(chron)
library(geosphere)
library(plyr)
library(tidyverse)
library(ggplot2)
library(lubridate)
library(e1071) 
library(randomForest)
library(tree)
library(gbm)
library(ROCR)
library(imager)
library(ISLR)
library(rpart)
library(cluster)

#Read the data
data <- read.csv(file="train.csv", header=TRUE, sep=",")
as.POSIXct(min(data$start_timestamp),,origin="1970-01-01")
as.POSIXct(max(data$start_timestamp),,origin="1970-01-01")
head(data)

#Randomly sampled 50,000 obs from the original dataset
set.seed(1)
sample_data = data[sample(nrow(data),50000),]

#split 40,000 training and 10,000 testing
set.seed(1)
train.indices = sample(1:nrow(sample_data), 40000)
Train=sample_data[train.indices,] #40000training
Test=sample_data[-train.indices,] #10000testing

#construct features: distance, weekday, hour, busy
earth.dist <- function (long1, lat1, long2, lat2)
{
  rad <- pi/180
  a1 <- lat1 * rad
  a2 <- long1 * rad
  b1 <- lat2 * rad
  b2 <- long2 * rad
  dlon <- b2 - a2
  dlat <- b1 - a1
  a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R <- 6378.145
  d <- R * c
  return(d)
}
Train<-Train%>%mutate(distance = earth.dist(start_lng, start_lat,end_lng, end_lat))
Train<-Train%>%mutate(weekday = weekdays(as.POSIXct(start_timestamp,origin="1970-01-01")))
Train<-Train%>%mutate(day = as.factor(ifelse(weekday %in% c("Saturday", "Sunday"),"weekend","weekday")))
Train<-Train%>%mutate(hour = as.factor(hour(as.POSIXct(start_timestamp,origin="1970-01-01"))))
Train<-Train%>%mutate(busy = as.factor(ifelse(hour %in% c(7,8,9,16,17,18,19), "busy", "free")))


Test<-Test%>%mutate(distance = earth.dist(start_lng, start_lat,end_lng, end_lat))
Test<-Test%>%mutate(weekday = weekdays(as.POSIXct(start_timestamp,origin="1970-01-01")))
Test<-Test%>%mutate(day = as.factor(ifelse(weekday %in% c("Saturday", "Sunday"),"weekend","weekday")))
Test<-Test%>%mutate(hour = as.factor(hour(as.POSIXct(start_timestamp,origin="1970-01-01"))))
Test<-Test%>%mutate(busy = as.factor(ifelse(hour %in% c(7,8,9,16,17,18,19), "busy", "free")))
Train<-Train%>%select(-weekday)%>%select(-row_id);
Test<-Test%>%select(-weekday)%>%select(-row_id);

#plot the data 
plot(Train$hour, main = "travel start time distribution", xlab="travel start time", ylab="number of rides")
plot(Train$hour, Train$distance,  xlab="travel start time", ylab="distance of rides")
par(mfrow=c(1,2))
plot(Train$start_lat,Train$start_lng, xlab="start latitude", ylab="start longitude",col=4)
plot(Train$end_lat,Train$end_lng,col=2, xlab="end latitude", ylab="end longitude", xlim=c(40.6,41.0),ylim=c(-74.2,-73.6))

#Remove the outlier
#Q1_dis = quantile(Train$distance, 0.25)
#Q3_dis = quantile(Train$distance, 0.75)
#iqr_dis = IQR(Train$distance)
#Q1_dur = quantile(Train$duration, 0.25)
#Q3_dur = quantile(Train$duration, 0.75)
#iqr_dur = IQR(Train$duration)
#Train<-Train%>%filter(Train$distance <= Q3_dis + 3*iqr_dis) 
#Train<-Train%>%filter(Train$duration <= Q3_dur + 3*iqr_dur) 

ggplot(data = Train, aes(duration))+geom_histogram(aes(y =..density..),binwidth=1)+geom_density(na.rm=TRUE)
ggplot(data=Train,aes(distance))+
  geom_histogram(aes(y =..density..),binwidth=0.1,na.rm=TRUE)+
  geom_density(na.rm=TRUE)


#Model1: Linear Regression with Lasso, Cross Validation
do.chunk <- function(chunkid, chunkdef, dat){ # function argument
  train = (chunkdef != chunkid)
  Xtr = dat[train,c(8:12)] # get training set
  Ytr = dat[train,]$duration # get true response values in trainig set
  Xvl = dat[!train,c(8:12)] # get validation set
  Yvl = dat[!train,]$duration # get true response values in validation set
  lm.duration <- lm(duration~distance + busy + day + hour + weekday, data = dat[train,])
  
  predYtr_lm = predict(lm.duration) # predict training values
  predYvl_lm = predict(lm.duration, Xvl) # predict validation values
  data.frame(fold = chunkid,
             train_mse_lm = mean((predYtr_lm - Ytr)^2), # compute and store training error
             val_mse_lm = mean((predYvl_lm - Yvl)^2)) # compute and store test error
}

#cu = cut(1:nrow(Train), 5, label=FALSE)
#s = sample(cu)
#res = ldply(1:5, do.chunk, chunkdef = s, dat = Train)
#error = data.frame(train.average.lm = mean(res$train_mse_lm),
#                   test.average.lm = mean(res$val_mse_lm))

glm.fit <-glm(duration~., data = Train)
sm = summary(glm.fit)
pred_test_glm = predict(glm.fit, Test)
mse(pred_test_glm, Test$duration)

#Model2: Decision Tree
tree.duration<-tree(duration~., data= Train)
draw.tree(tree.duration, cex=0.6)
prune<-prune.tree(tree.duration)
plot(prune)
best.prune = prune$size[which.min(prune$dev)]
pred_test_tree<-predict(tree.duration, newdata=Test)
mse_RT<-mse(pred_test_tree, Test$duration)
sqrt(mse_RT)
pred_train_tree<-predict(tree.duration)
mse_RT_train<-mse(pred_train_tree, Train$duration)
sqrt(mse_RT_train)

mae(pred_test_tree, Test$duration)
mae(pred_train_tree, Train$duration)
mae(pred_test_tree, Test$duration)*10000/sum(Test$duration)
mae(pred_train_tree, Train$duration)*40000/sum(Train$duration)


plot(pred_test_tree,Test$duration, main="Regression Tree",xlab="predict duration", ylab="actual duration")
abline(0, 1, col="red")

#Model3: Support Vector Regression
svm.fit<-svm(duration~., data = Train)
pred_test_svm <- predict(svm.fit, newdata = Test%>%select(-duration))
dim(pred_test_svm)
mse_svm = mse(pred_test_svm, Test$duration)
sqrt(mse_svm)
pred_train_svm<-predict(svm.fit)
mse_svm_train<-mse(pred_train_svm, Train$duration)
sqrt(mse_svm_train)
mae(pred_test_svm, Test$duration)
mae(pred_train_svm, Train$duration)
mae(pred_test_svm, Test$duration)*10000/sum(Test$duration)
mae(pred_train_svm, Train$duration)*40000/sum(Train$duration)


plot(pred_test_svm,Test$duration, main="Support Vector Regression",xlab="predict duration", ylab="actual duration")
abline(0, 1, col="red")

tune_svm = tune(svm, duration~., data = Train, ranges = list(elsilon=seq(0,1,0.1), cost=c(0.001, 0.01, 0.1,1,10,100)))
tune_svm$best.parameters
tune_svm$best.performance
best_model = tune_svm$best.model


#Model4: Random Forest
set.seed(1)
rf<-randomForest(duration~., data = Train, importance = TRUE)
plot(rf,main="Random Forest")
summary(rf)
importance(rf)
varImpPlot(rf, n.var = 9)
pred_test_rf = predict(rf, newdata = Test,n.trees=500)
mse_rf = mse(pred_test_rf, Test$duration)
sqrt(mse_rf)
par(mfrow=c(1,1))
plot(pred_test_rf, Test$duration, main="Random Forest", xlab="predict duration", ylab="actual duration")
abline(0, 1, col="red")

pred_train_rf<-predict(rf)
mse_rf_train<-mse(pred_train_rf, Train$duration)
sqrt(mse_rf_train)

mae(pred_test_rf, Test$duration)
mae(pred_train_rf, Train$duration)
mae(pred_test_rf, Test$duration)*10000/sum(Test$duration)
mae(pred_train_rf, Train$duration)*40000/sum(Train$duration)


#Model 5: Gradient Boosting
set.seed(1)
gbm.duration<-gbm(duration~.,data = Train, distribution = "gaussian", interaction.depth=8,n.trees=10000, shrinkage = 0.01)
par(mfrow=c(1,1))
summary(gbm.duration, plotit=TRUE, order=TRUE)
pred_test_gbm = predict(gbm.duration, newdata = Test, type="response", n.trees=10000)
mse_gb=mse(pred_test_gbm, Test$duration)
sqrt(mse_gb)

pred_train_gb<-predict(gbm.duration, type="response",n.trees = 10000)
mse_gb_train<-mse(pred_train_gb, Train$duration)
sqrt(mse_gb_train)

mae(pred_test_gbm, Test$duration)
mae(pred_train_gb, Train$duration)
mae(pred_test_gbm, Test$duration)*10000/sum(Test$duration)
mae(pred_train_gb, Train$duration)*40000/sum(Train$duration)


par(mfrow=c(1,2))
plot(gbm.duration, i ="distance")
plot(gbm.duration,i="hour")
plot(pred_test_gbm, Test$duration, main="Gradient Boosting Tree", xlab="predict duration", ylab="actual duration")
abline(0, 1, col="red")
