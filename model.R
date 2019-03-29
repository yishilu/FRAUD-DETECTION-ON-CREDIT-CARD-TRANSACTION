setwd("~/Desktop/Spring2019/DSO562/Project 2")
library(dplyr)
library(data.table)
library(faraway)
library(mlbench)
library(ranger)
library(car)
library(xgboost)
library(caret)
library(ggplot2)
library(neuralnet)

modelling=fread('scaled_modeling_data.csv')
score=fread('feature_selection_univariate.csv')
feature=score$Variable
new_modelling=modelling %>% select(feature,Fraud)
oot=fread('scaled_OOT_data.csv')
oot=oot %>% select(Fraud,tot_cs_3,tot_card_0,tot_merch_1,max_merch_1,tot_cm_30,
                   vcv_ac1_nc14,qat_cs_3,qamed_card_30,max_card_30,max_merch_14, 
                   max_card_0,tot_card_14,tot_cs_30,tot_card_1,qat_cm_3)


## forward selection to reduce predictors to 20 
feature_mod_start = glm(Fraud ~ 1, data = new_modelling,family='binomial')
feature_mod_full= glm(Fraud ~ ., data = new_modelling,family='binomial')
feature_mod_forw_aic = step(
  feature_mod_full, 
  scope=list(lower=formula(feature_mod_start),upper=formula(feature_mod_full)), 
  direction = "backward",steps=30)

## create new dataset based on 20 predictors 
sig_data=new_modelling %>% select(Fraud,tot_cs_3,tot_card_0,tot_merch_1,max_merch_1,tot_cm_30,
                                  vcv_ac1_nc14,qat_cs_3,qamed_card_30,max_card_30,max_merch_14, 
                                  max_card_0,avg_cs_0,tot_card_30,tot_card_14,tot_cs_30,
                                  med_cs_14,tot_card_1,qat_cm_3,max_merch_7,med_cm_7)

## use vif to remove variables with collinearity and make sure vif smaller than 10 (max_merch_7,tot_card_30,avg_cs_0)
## remove variables with p-values not significant in the model (med_cm_7,med_cs_14)
feature_model_glm=glm(Fraud~.-max_merch_14-tot_card_14-med_cm_7,sig_data,family='binomial')

vif(feature_model_glm)
summary(feature_model_glm)


## update dataset with 17 predicators
sig_data_17=sig_data %>% select(Fraud,tot_cs_3,tot_card_0,tot_merch_1,max_merch_1,tot_cm_30,
                                vcv_ac1_nc14,qat_cs_3,qamed_card_30,max_card_30,max_merch_14, 
                                max_card_0,tot_card_14,tot_cs_30,tot_card_1,qat_cm_3,med_cm_7,med_cs_14)

## update dataset with 15 predicators
sig_data_new=sig_data %>% select(Fraud,tot_cs_3,tot_card_0,tot_merch_1,max_merch_1,tot_cm_30,
                                 vcv_ac1_nc14,qat_cs_3,qamed_card_30,max_card_30,max_merch_14, 
                                 max_card_0,tot_card_14,tot_cs_30,tot_card_1,qat_cm_3)


## random forest   (parameters: num.trees, mtry, max.depth=3,num.random.splits = 3,min.node.size=2)
fdr.train = NULL
fdr.test = NULL
fdr.oot = NULL

for (j in 1:10){
  rows=sample(nrow(sig_data_new))
  dframe=sig_data_new[rows,]
  split=round(nrow(dframe)*0.7)
  train=dframe[1:split,]
  test=dframe[(split+1):nrow(dframe),]
  
  model.select=ranger(Fraud~.,train, num.trees = 300, mtry = 6)
  model.trainprob = model.select$predictions
  train$trainprob = model.trainprob
  t = train%>%
    arrange(-trainprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.train = append(fdr.train,sum(t$Fraud)/sum(train$Fraud))
  
  model.testprob = predict(model.select, test, predict.all = FALSE, num.trees=model.select$num.trees, type = "response")$predictions
  test$testprob = model.testprob
  t = test%>%
    arrange(-testprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.test = append(fdr.test,sum(t$Fraud)/sum(test$Fraud))
  
  model.ootprob =predict(model.select, oot, predict.all = FALSE, num.trees=model.select$num.trees, type = "response")$predictions
  oot$ootprob = model.ootprob
  t = oot%>%
    arrange(-ootprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.oot = append(fdr.oot,sum(t$Fraud)/sum(oot$Fraud))
  
}

sum(fdr.train)/10
sum(fdr.test)/10
sum(fdr.oot)/10

## Making table


### XGBOOST 

fdr.train = NULL
fdr.test = NULL
fdr.oot = NULL

oot=fread('scaled_OOT_data.csv')
oot=oot %>% select(Fraud,tot_cs_3,tot_card_0,tot_merch_1,max_merch_1,tot_cm_30,
                   vcv_ac1_nc14,qat_cs_3,qamed_card_30,max_card_30,max_merch_14, 
                   max_card_0,tot_card_14,tot_cs_30,tot_card_1,qat_cm_3)

oot_data=as.matrix(oot %>% select(-Fraud))
oot_label=as.matrix(oot %>% select(Fraud))
doot <- xgb.DMatrix(data = oot_data, label= oot_label)


for (j in 1:10){
  rows=sample(nrow(sig_data_new))
  dframe=sig_data_new[rows,]
  split=round(nrow(dframe)*0.7)
  train=dframe[1:split,]
  test=dframe[(split+1):nrow(dframe),]
  train_data=as.matrix(train %>% select(-Fraud))
  train_label=as.matrix(train %>% select(Fraud))
  test_data=as.matrix(test %>% select(-Fraud))
  test_label=as.matrix(test %>% select(Fraud))
  
  dtrain <- xgb.DMatrix(data = train_data, label= train_label)
  dtest <- xgb.DMatrix(data = test_data, label= test_label)
  
  model.select=xgboost(data=dtrain,nround=10,n_estimators=400, objective = "binary:logistic")
  model.trainprob = predict(model.select, dtrain)
  train$trainprob = model.trainprob
  t = train%>%
    arrange(-trainprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.train = append(fdr.train,sum(t$Fraud)/sum(train$Fraud))
  
  
  model.testprob = predict(model.select, dtest)
  test$testprob = model.testprob
  t = test%>%
    arrange(-testprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.test = append(fdr.test,sum(t$Fraud)/sum(test$Fraud))
  
  model.ootprob =predict(model.select, doot)
  oot$ootprob = model.ootprob
  t = oot%>%
    arrange(-ootprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.oot = append(fdr.oot,sum(t$Fraud)/sum(oot$Fraud))
  
}

sum(fdr.train)/10
sum(fdr.test)/10
sum(fdr.oot)/10

### logistic
fdr.train = NULL
fdr.test = NULL
fdr.oot = NULL

for (j in 1:10){
  rows=sample(nrow(sig_data_new))
  dframe=sig_data_new[rows,]
  split=round(nrow(dframe)*0.7)
  train=dframe[1:split,]
  test=dframe[(split+1):nrow(dframe),]
  
  model.select=glm(Fraud~.,data=train,family='binomial')
  model.trainprob = predict(model.select, train,type='response')
  model.trainpred = rep(0,length(model.trainprob))
  model.trainpred[model.trainprob>0.5] = 1
  train$trainprob = model.trainprob
  train$trainpred = model.trainpred
  t = train%>%
    arrange(-trainprob)%>%
    mutate(index = Fraud+trainpred)
  t = t[1:round(nrow(t)*0.03),]
  fdr.train = append(fdr.train,nrow(t[t$index == 2,])/sum(train$Fraud))
  
  
  model.testprob = predict(model.select, test,type='response')
  model.testpred = rep(0,length(model.testprob))
  model.testpred[model.testprob>0.5] = 1
  test$testprob = model.testprob
  test$testpred = model.testpred
  t = test%>%
    arrange(-testprob)%>%
    mutate(index = Fraud+testpred)
  t = t[1:round(nrow(t)*0.03),]
  fdr.test = append(fdr.test,nrow(t[t$index == 2,])/sum(test$Fraud))
  
  model.ootprob =predict(model.select, oot,type='response')
  model.ootpred = rep(0,length(model.ootprob))
  model.ootpred[model.ootprob>0.5] = 1
  oot$ootprob = model.ootprob
  oot$ootpred = model.ootpred
  t = oot%>%
    arrange(-ootprob)%>%
    mutate(index = Fraud+ootpred)
  t = t[1:round(nrow(t)*0.03),]
  fdr.oot = append(fdr.oot,nrow(t[t$index == 2,])/sum(oot$Fraud))
  
}

sum(fdr.train)/10
sum(fdr.test)/10
sum(fdr.oot)/10

### XGBOOST 

fdr.train = NULL
fdr.test = NULL
fdr.oot = NULL

oot=fread('scaled_OOT_data.csv')
oot=oot %>% select(Fraud,tot_cs_3,tot_card_0,tot_merch_1,max_merch_1,tot_cm_30,
                   vcv_ac1_nc14,qat_cs_3,qamed_card_30,max_card_30,max_merch_14, 
                   max_card_0,tot_card_14,tot_cs_30,tot_card_1,qat_cm_3,med_cm_7,med_cs_14)

oot_data=as.matrix(oot %>% select(-Fraud))
oot_label=as.matrix(oot %>% select(Fraud))
doot <- xgb.DMatrix(data = oot_data, label= oot_label)


for (j in 1:10){
  rows=sample(nrow(sig_data_17))
  dframe=sig_data_17[rows,]
  split=round(nrow(dframe)*0.7)
  train=dframe[1:split,]
  test=dframe[(split+1):nrow(dframe),]
  train_data=as.matrix(train %>% select(-Fraud))
  train_label=as.matrix(train %>% select(Fraud))
  test_data=as.matrix(test %>% select(-Fraud))
  test_label=as.matrix(test %>% select(Fraud))
  
  dtrain <- xgb.DMatrix(data = train_data, label= train_label)
  dtest <- xgb.DMatrix(data = test_data, label= test_label)
  
  model.select=xgboost(data=dtrain,nround=10,n_estimators=400, objective = "binary:logistic")
  model.trainprob = predict(model.select, dtrain)
  train$trainprob = model.trainprob
  t = train%>%
    arrange(-trainprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.train = append(fdr.train,sum(t$Fraud)/sum(train$Fraud))
  
  
  model.testprob = predict(model.select, dtest)
  test$testprob = model.testprob
  t = test%>%
    arrange(-testprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.test = append(fdr.test,sum(t$Fraud)/sum(test$Fraud))
  
  model.ootprob =predict(model.select, doot)
  oot$ootprob = model.ootprob
  t = oot%>%
    arrange(-ootprob)
  t = t[1:round(nrow(t)*0.03),]
  fdr.oot = append(fdr.oot,sum(t$Fraud)/sum(oot$Fraud))
  
}

sum(fdr.train)/10
sum(fdr.test)/10
sum(fdr.oot)/10





### neuralnet
fdr.train = NULL
fdr.test = NULL
fdr.oot = NULL

for (j in 1:10){
  rows=sample(nrow(sig_data_new))
  dframe=sig_data_new[rows,]
  split=round(nrow(dframe)*0.7)
  train=dframe[1:split,]
  test=dframe[(split+1):nrow(dframe),]
  
  model.select=neuralnet(Fraud~.,data=train, hidden=500,act.fct = "logistic",
                         linear.output = FALSE)
  model.trainprob = predict(model.select, train)
  model.trainpred = rep(0,length(model.trainprob))
  model.trainpred[model.trainprob>0.5] = 1
  train$trainprob = model.trainprob
  train$trainpred = model.trainpred
  t = train%>%
    arrange(-trainprob)%>%
    mutate(index = Fraud+trainpred)
  t = t[1:round(nrow(t)*0.03),]
  fdr.train = append(fdr.train,nrow(t[t$index == 2,])/sum(train$Fraud))
  
  
  model.testprob = predict(model.select, test)
  model.testpred = rep(0,length(model.testprob))
  model.testpred[model.testprob>0.5] = 1
  test$testprob = model.testprob
  test$testpred = model.testpred
  t = test%>%
    arrange(-testprob)%>%
    mutate(index = Fraud+testpred)
  t = t[1:round(nrow(t)*0.03),]
  fdr.test = append(fdr.test,nrow(t[t$index == 2,])/sum(test$Fraud))
  
  model.ootprob =predict(model.select, oot)
  model.ootpred = rep(0,length(model.ootprob))
  model.ootpred[model.ootprob>0.5] = 1
  oot$ootprob = model.ootprob
  oot$ootpred = model.ootpred
  t = oot%>%
    arrange(-ootprob)%>%
    mutate(index = Fraud+ootpred)
  t = t[1:round(nrow(t)*0.03),]
  fdr.oot = append(fdr.oot,nrow(t[t$index == 2,])/sum(oot$Fraud))
  
}

sum(fdr.train)/10
sum(fdr.test)/10
sum(fdr.oot)/10