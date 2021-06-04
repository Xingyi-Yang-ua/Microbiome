R.version
library(tidyverse)
library(caret)
library(randomForest)
library(reticulate)
np <- import("numpy")

##################################Configuration############################################
data_path = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/s0"
path_genus = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/genus48"
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count/'

y_path = sprintf('%s/%s', data_path, 'y.csv')
tree_info_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/genus48/genus48_dic.csv'
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count'
count_list_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/gcount_list.csv'
idx_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/s0/idx.csv'

num_classes = 0 # regression
tree_level_list = c('Genus', 'Family', 'Order', 'Class', 'Phylum')

#################################Read phylogenetic tree information#############################

phylogenetic_tree_info = read.csv(tree_info_path)
phylogenetic_tree_info = phylogenetic_tree_info %>% select(all_of(tree_level_list))

print(sprintf('Phylogenetic tree level list: %s', str_c(phylogenetic_tree_info %>% colnames, collapse = ', ')))

###################################read dataset################################################
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

######################################read true tree weight################################
tw_1 = np$load(sprintf('%s/tw_1.npy', data_path))
tw_2 = np$load(sprintf('%s/tw_2.npy', data_path))
tw_3 = np$load(sprintf('%s/tw_3.npy', data_path))
tw_4 = np$load(sprintf('%s/tw_4.npy', data_path))

###################################simulation for all n################################
random_forest_res <- function(fold, importance_type=2, fs_thrd = 0.1){
  print(sprintf('-----------------------------------------------------------------'))
  print(sprintf('Random Forest computation for %dth repetition', fold))
  
  dataset = read_dataset(x_path[fold], y_path, fold)
  x_train = dataset[[1]]
  x_test = dataset[[2]]
  y_train = dataset[[3]]
  y_test = dataset[[4]]
  
  fit.rf <- randomForest(y_train~.,data=x_train, ntree=1000,  mtry=19, importance=TRUE)
  train.pred <- fit.rf$predicted
  test.pred <- predict(fit.rf,x_test)
  
  train.mse <- mean((y_train - train.pred)^2)
  train.cor <- cor(y_train, train.pred)
  
  test.mse <- mean((y_test - test.pred)^2)
  test.cor <- cor(y_test, test.pred)
  
  # Feature selection
  ## variable importance
  incmse<-fit.rf$importance[,1]
  incmse
  selection<-incmse[incmse>0] # select those who has positive change of misclassification rate
  selected1<-c(names(selection))
  x_train_selected<-x_train[,selected1]
  x_test_selected<-x_test[,selected1]
  
  fit.rf_selected <- randomForest(y_train~.,data=x_train_selected, ntree=1000,  mtry=8, importance=T)
  train.pred_selected <- fit.rf_selected$predicted
  test.pred_selected <- predict(fit.rf_selected,x_test_selected)
  
  train.mse_selected <- mean((y_train - train.pred_selected)^2)
  train.cor_selected <- cor(y_train, train.pred_selected)
  
  test.mse_selected <- mean((y_test - test.pred_selected)^2)
  test.cor_selected <- cor(y_test, test.pred_selected)

  selected_genus<-ifelse(incmse >0, 1, 0)
  #selected_genus<-as.data.frame(selected_genus)
  #print(selected_genus)
  
  fold_genus = apply(tw_1[fold,,], 1, sum)
  names(fold_genus) <- phylogenetic_tree_info[1] %>% unique %>% .[,1]
  fold_family = apply(tw_2[fold,,], 1, sum)
  names(fold_family) <- phylogenetic_tree_info[2] %>% unique %>% .[,1]
  fold_order = apply(tw_3[fold,,], 1, sum)
  names(fold_order) <- phylogenetic_tree_info[3] %>% unique %>% .[,1]
  fold_class = apply(tw_4[fold,,], 1, sum)
  names(fold_class) <- phylogenetic_tree_info[4] %>% unique %>% .[,1]
  
  selected_family <- rep(0, fold_family %>% length)
  names(selected_family) <- names(fold_family)
  selected_names <- phylogenetic_tree_info[selected_genus == 1,2] %>% as.character
  selected_family[names(selected_family) %in% selected_names] <- 1
  selected_order <- rep(0, fold_order %>% length)
  names(selected_order) <- names(fold_order)
  selected_names <- phylogenetic_tree_info[selected_genus == 1,3] %>% as.character
  selected_order[names(selected_order) %in% selected_names] <- 1
  selected_class <- rep(0, fold_class %>% length)
  names(selected_class) <- names(fold_class)
  selected_names <- phylogenetic_tree_info[selected_genus == 1,4] %>% as.character
  selected_class[names(selected_class) %in% selected_names] <- 1
  
  #####################################performance#############################
  fs_genus_conf_table <- table(selected_genus, fold_genus)
  fs_genus_sensitivity <- sensitivity(fs_genus_conf_table) 
  fs_genus_specificity <- specificity(fs_genus_conf_table)
  fs_genus_gmeasure <- sqrt(fs_genus_sensitivity*fs_genus_specificity)
  fs_genus_accuracy <- sum(diag(fs_genus_conf_table))/sum(fs_genus_conf_table)
  
  fs_family_conf_table <- table(selected_family, fold_family)
  fs_family_sensitivity <- sensitivity(fs_family_conf_table) 
  fs_family_specificity <- specificity(fs_family_conf_table)
  fs_family_gmeasure <- sqrt(fs_family_sensitivity*fs_family_specificity)
  fs_family_accuracy <- sum(diag(fs_family_conf_table))/sum(fs_family_conf_table)
  
  fs_order_conf_table <- table(selected_order, fold_order)
  fs_order_sensitivity <- sensitivity(fs_order_conf_table) 
  fs_order_specificity <- specificity(fs_order_conf_table)
  fs_order_gmeasure <- sqrt(fs_order_sensitivity*fs_order_specificity)
  fs_order_accuracy <- sum(diag(fs_order_conf_table))/sum(fs_order_conf_table)
  
  fs_class_conf_table <- table(selected_class, fold_class)
  fs_class_sensitivity <- sensitivity(fs_class_conf_table) 
  fs_class_specificity <- specificity(fs_class_conf_table)
  fs_class_gmeasure <- sqrt(fs_class_sensitivity*fs_class_specificity)
  fs_class_accuracy <- sum(diag(fs_class_conf_table))/sum(fs_class_conf_table)
  
  

  print(sprintf('Train mse: %s, Train Correlation: %s', train.mse, train.cor))
  print(sprintf('Test mse: %s, Test Correlation: %s', test.mse, test.cor))
  print(sprintf('Train mse_selected: %s, Train Correlation_selected: %s', train.mse_selected, train.cor_selected))
  print(sprintf('Test mse_selected: %s, Test Correlation_selected: %s', test.mse_selected, test.cor_selected))
  print(sprintf('Taxa selection performance: genus level---------------'))
  print(sprintf('FS sensitivity: %s, FS sensitivity: %s, FS gmeasure: %s, FS accuracy: %s',
                fs_genus_sensitivity, fs_genus_specificity, fs_genus_gmeasure, fs_genus_accuracy))
  print(sprintf('Taxa selection performance: family level---------------'))
  print(sprintf('FS sensitivity: %s, FS sensitivity: %s, FS gmeasure: %s, FS accuracy: %s',
                fs_family_sensitivity, fs_family_specificity, fs_family_gmeasure, fs_family_accuracy))
  print(sprintf('Taxa selection performance: order level---------------'))
  print(sprintf('FS sensitivity: %s, FS sensitivity: %s, FS gmeasure: %s, FS accuracy: %s',
                fs_order_sensitivity, fs_order_specificity, fs_order_gmeasure, fs_order_accuracy))
  print(sprintf('Taxa selection performance: class level---------------'))
  print(sprintf('FS sensitivity: %s, FS sensitivity: %s, FS gmeasure: %s, FS accuracy: %s',
                fs_class_sensitivity, fs_class_specificity, fs_class_gmeasure, fs_class_accuracy))
  
  p<-c(train.mse, train.cor, test.mse, test.cor, 
       train.mse_selected, train.cor_selected,test.mse_selected, test.cor_selected, 
       fs_genus_sensitivity, fs_genus_specificity, fs_genus_gmeasure, fs_genus_accuracy,
       fs_family_sensitivity, fs_family_specificity, fs_family_gmeasure, fs_family_accuracy,
       fs_order_sensitivity, fs_order_specificity, fs_order_gmeasure, fs_order_accuracy,
       fs_class_sensitivity, fs_class_specificity, fs_class_gmeasure, fs_class_accuracy)
  
  write.table(matrix(p, nrow=1), file="s1_1_output1.csv",append=T,sep = ",", row.names = FALSE,col.names = FALSE)
  return (c(train.mse, train.cor, test.mse, test.cor, 
            train.mse_selected, train.cor_selected,test.mse_selected, test.cor_selected, 
            fs_genus_sensitivity, fs_genus_specificity, fs_genus_gmeasure, fs_genus_accuracy,
            fs_family_sensitivity, fs_family_specificity, fs_family_gmeasure, fs_family_accuracy,
            fs_order_sensitivity, fs_order_specificity, fs_order_gmeasure, fs_order_accuracy,
            fs_class_sensitivity, fs_class_specificity, fs_class_gmeasure, fs_class_accuracy))
}

set.seed(100)
res <- sapply(seq(1,1000), random_forest_res)

results_table = res %>% t %>% data.frame
colnames(results_table) = c('Training MSE', 'Training Correlation', 'Test MSE', 'Test Correlation',
                            'Training MSE_selected','Training Correlation_selected', 'Test MSE_selected', 'Test Correlation_selected',
                            'Taxa selection (genus) sensitivity',
                            'Taxa selection (genus) specificity',
                            'Taxa selection (genus) gmeasure', 
                            'Taxa selection (genus) accuracy',
                            'Taxa selection (family) sensitivity',
                            'Taxa selection (family) specificity',
                            'Taxa selection (family) gmeasure', 
                            'Taxa selection (family) accuracy',
                            'Taxa selection (order) sensitivity',
                            'Taxa selection (order) specificity',
                            'Taxa selection (order) gmeasure', 
                            'Taxa selection (order) accuracy',
                            'Taxa selection (class) sensitivity',
                            'Taxa selection (class) specificity',
                            'Taxa selection (class) gmeasure', 
                            'Taxa selection (class) accuracy')
results_table
print('Mean')
print(apply(results_table, 2, mean))

print('SD')
print(apply(results_table, 2, sd))
