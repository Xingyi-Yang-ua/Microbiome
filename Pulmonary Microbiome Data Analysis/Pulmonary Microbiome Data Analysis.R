library(tidyverse)
library(caret)
library(randomForest)
library(reticulate)
np <- import("numpy")

###################################Configuration############################333
data_path = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/lung"
path_genus = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/lung/genus48"
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/lung/count/'

y_path = sprintf('%s/%s', data_path, 'y_all.csv')
tree_info_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/lung/genus48_dic.csv'
#count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count'
count_list_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/lung/gcount_list.csv'
idx_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/lung/idxall.csv'

num_classes = 0 # regression
tree_level_list = c('Genus', 'Family', 'Order', 'Class', 'Phylum')

################################Read phylogenetic tree information#########################

phylogenetic_tree_info = read.csv(tree_info_path)
phylogenetic_tree_info = phylogenetic_tree_info %>% select(tree_level_list)

print(sprintf('Phylogenetic tree level list: %s', str_c(phylogenetic_tree_info %>% colnames, collapse = ', ')))


#################################Read Dataset##########################################
read_dataset <- function(x_path, y_path, sim){
  print(str_c('Load data for repetition ', sim))
  x = read.csv(x_path)
  y = read.csv(y_path)[,sim]
  x = (x - max(x)) / (max(x) - min(x))
  
  idxs = idxs_total[, sim]
  remain_idxs = setdiff(seq(1, dim(x)[1]), idxs)
  
  x_train = x[idxs,]
  x_test = x[remain_idxs,]
  y_train = y[idxs]
  y_test = y[remain_idxs]
  
  return (list(x_train, x_test, y_train, y_test))
}
idxs_total = read.csv(idx_path)
number_of_fold = dim(idxs_total)[2]; number_of_fold

x_list = read.csv(count_list_path, header = FALSE)
x_path = x_list$V1 %>% sprintf('%s/%s', count_path, .)

#####################################Read true weight##################################
tw_1 = np$load(sprintf('%s/tw_1.npy', data_path))
tw_2 = np$load(sprintf('%s/tw_2.npy', data_path))
tw_3 = np$load(sprintf('%s/tw_3.npy', data_path))
tw_4 = np$load(sprintf('%s/tw_4.npy', data_path))

#####################################Simulation########################################
result<-matrix(rep(0, len=13000), nrow = 1000, ncol = 13)
for (i in 1:1000){ 
  random_forest_res <- function(fold){
    print(sprintf('-----------------------------------------------------------------'))
    print(sprintf('Random Forest computation for %dth repetition', fold))
    
    dataset = read_dataset(x_path[fold], y_path, fold)
    x_train = dataset[[1]]
    x_test = dataset[[2]]
    y_train = dataset[[3]]
    y_test = dataset[[4]]
    write.table(matrix(y_train, nrow=1), file="y_train.csv",append=T,sep = ",", row.names = FALSE,col.names = FALSE)
    
    
    fit.rf <- randomForest(y_train~.,data=x_train, ntree=500, importance=TRUE)
    train.pred <- fit.rf$predicted
    test.pred <- predict(fit.rf,x_test)
    write.table(matrix(train.pred, nrow=1), file="train_pred.csv",append=T,sep = ",", row.names = FALSE,col.names = FALSE)
    
    train.mse <- mean((y_train - train.pred)^2)
    train.cor <- cor(y_train, train.pred)
    
    test.mse <- mean((y_test - test.pred)^2)
    test.cor <- cor(y_test, test.pred)
    
    # Feature selection
    ## variable importance
    incmse<-fit.rf$importance[,1]
    selction<-incmse[incmse>0]
    selected1<-c(names(selction))
    x_train_selected<-x_train[,selected1]
    x_test_selected<-x_test[,selected1]
    fit.rf_selected <- randomForest(y_train~.,data=x_train_selected, ntree=500,importance=T)
    train.pred_selected <- fit.rf_selected$predicted
    test.pred_selected <- predict(fit.rf_selected,x_test_selected)
    
    train.mse_selected <- mean((y_train - train.pred_selected)^2)
    train.cor_selected <- cor(y_train, train.pred_selected)
    
    test.mse_selected <- mean((y_test - test.pred_selected)^2)
    test.cor_selected <- cor(y_test, test.pred_selected)
    
    write.csv(matrix(selected1, ncol = 1), file ="lung_genus_rf.csv", row.names=FALSE)
    
    print(sprintf('Train mse: %s, Train Correlation: %s', train.mse, train.cor))
    print(sprintf('Test mse: %s, Test Correlation: %s', test.mse, test.cor))
    print(sprintf('Train mse_selected: %s, Train Correlation_selected: %s', train.mse_selected, train.cor_selected))
    print(sprintf('Test mse_selected: %s, Test Correlation_selected: %s', test.mse_selected, test.cor_selected))
    
    return( c(train.mse, train.cor, test.mse, test.cor,
              train.mse_selected, train.cor_selected, test.mse_selected, test.cor_selected, y_test, test.pred, test.pred_selected ))
  }
  res <- sapply(seq(1,30), random_forest_res)
  results_table = res %>% t %>% data.frame
  mean <- apply(results_table, 2, mean)
  cor_test<- cor(results_table[,9],results_table[,10])
  cor_test_selected<- cor(results_table[,9],results_table[,11])
  print(mean)
  print(cor_test)
  print (cor_test_selected)
  result[i,]<- c(mean, cor_test, cor_test_selected)
}

colnames(result) = c('Training MSE', 'Training Correlation', 'Test MSE', 'Test Correlation', 
                     'Training MSE_selected','Training Correlation_selected', 'Test MSE_selected', 'Test Correlation_selected',
                     'y_test','test.pred','test.pred_selected', 'test_cor','test_cor_selected')
result
