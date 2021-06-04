R.version
library(tidyverse)
library(caret)
library(randomForest)
library(reticulate)
library(mltools)
np <- import("numpy")

########################################Configuration###################################

data_path = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/s2"
path_genus = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/genus48"
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count/'

y_path = sprintf('%s/%s', data_path, 'y.csv')
tree_info_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/genus48/genus48_dic.csv'
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count'
count_list_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/gcount_list.csv'
idx_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/s1/idx.csv'

num_classes = 0 # regression
tree_level_list = c('Genus', 'Family', 'Order', 'Class', 'Phylum')

###################################Read phylogenetic tree information########################

phylogenetic_tree_info = read.csv(tree_info_path)
phylogenetic_tree_info = phylogenetic_tree_info %>% select(all_of(tree_level_list))

print(sprintf('Phylogenetic tree level list: %s', str_c(phylogenetic_tree_info %>% colnames, collapse = ', ')))

#########################################Read dataset#######################################

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

###################################read true tree weight##############################
tw_1 = np$load(sprintf('%s/tw_1.npy', data_path))
tw_2 = np$load(sprintf('%s/tw_2.npy', data_path))
tw_3 = np$load(sprintf('%s/tw_3.npy', data_path))
tw_4 = np$load(sprintf('%s/tw_4.npy', data_path))

######################################Simulation for all n##############################
random_forest_res <- function(fold, importance_type=2, fs_thrd = 0.1){
  #     print(sprintf('-----------------------------------------------------------------'))
  #     print(sprintf('Random Forest computation for %dth repetition', fold))
  
  dataset = read_dataset(x_path[fold], y_path, fold)
  x_train = dataset[[1]]
  x_test = dataset[[2]]
  y_train = dataset[[3]]
  y_test = dataset[[4]]
  
  # Binary classification
  y_train = factor(y_train, levels = c(0,1), ordered=TRUE)
  y_test = factor(y_test, levels = c(0,1), ordered=TRUE)
  
  fit.rf <- randomForest(y_train~.,data=x_train, ntree=1000, importance=TRUE)
  train.pred <- fit.rf$predicted
  test.pred <- predict(fit.rf,x_test)
  
  train_conf <- table(train.pred, y_train)
  train_sensitivity <- sensitivity(train_conf)
  train_specificity <- specificity(train_conf)
  train_gmeasure <- sqrt(train_sensitivity*train_specificity)
  train_accuracy <- sum(diag(train_conf))/sum(train_conf)
  train_auc <- auc_roc(train.pred, y_train)
  
  test_conf <- table(test.pred, y_test)
  test_sensitivity <- sensitivity(test_conf)
  test_specificity <- specificity(test_conf)
  test_gmeasure <- sqrt(test_sensitivity*test_specificity)
  test_accuracy <- sum(diag(test_conf))/sum(test_conf)
  test_auc <- auc_roc(test.pred, y_test)
  
  ######################################Feature selection###############################
  ## variable importance
  incmse<-fit.rf$importance[,1]
  selection<-incmse[incmse>0]
  selected1<-c(names(selection))
  x_train_selected<-x_train[,selected1]
  x_test_selected<-x_test[,selected1]
  fit.rf_selected <- randomForest(y_train~.,data=x_train_selected, ntree=1000,importance=T)
  train.pred_selected <- fit.rf_selected$predicted
  test.pred_selected <- predict(fit.rf_selected,x_test_selected)
  
  train_conf_selected <- table(train.pred_selected, y_train)
  train_sensitivity_selected <- sensitivity(train_conf_selected)
  train_specificity_selected <- specificity(train_conf_selected)
  train_gmeasure_selected <- sqrt(train_sensitivity_selected*train_specificity_selected)
  train_accuracy_selected <- sum(diag(train_conf_selected))/sum(train_conf_selected)
  train_auc_selected <- auc_roc(train.pred_selected, y_train)
  
  test_conf_selected <- table(test.pred_selected, y_test)
  test_sensitivity_selected <- sensitivity(test_conf_selected)
  test_specificity_selected <- specificity(test_conf_selected)
  test_gmeasure_selected <- sqrt(test_sensitivity_selected*test_specificity_selected)
  test_accuracy_selected <- sum(diag(test_conf_selected))/sum(test_conf_selected)
  test_auc_selected <- auc_roc(test.pred_selected, y_test)
  
  selected_genus<-ifelse(incmse >0, 1, 0)

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
  
  ###################################performance######################################
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
  
  print(sprintf('Train sensitivity: %s, Train specificity: %s, Train gmeasure: %s, Train accuracy: %s, Train AUC: %s',
                train_sensitivity, train_specificity, train_gmeasure, train_accuracy, train_auc))
  print(sprintf('Test sensitivity: %s, Test specificity: %s, Test gmeasure: %s, Test accuracy: %s, Test AUC: %s',
                test_sensitivity, test_specificity, test_gmeasure, test_accuracy, test_auc))
  print(sprintf('Train sensitivity_selected: %s, Train specificity_selected: %s, Train gmeasure_selected: %s, Train accuracy_selected: %s, Train AUC_selected: %s',
                train_sensitivity_selected, train_specificity_selected, train_gmeasure_selected, train_accuracy_selected, train_auc_selected))
  print(sprintf('Test sensitivity_selected: %s, Test specificity_selected: %s, Test gmeasure_selected: %s, Test accuracy_selected: %s, Test AUC_selected: %s',
                test_sensitivity_selected, test_specificity_selected, test_gmeasure_selected, test_accuracy_selected, test_auc_selected))
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
  p<-c(train_sensitivity, train_specificity, train_gmeasure, train_accuracy, train_auc, 
       test_sensitivity, test_specificity, test_gmeasure, test_accuracy, test_auc,
       train_sensitivity_selected, train_specificity_selected, train_gmeasure_selected, train_accuracy_selected, train_auc_selected, 
       test_sensitivity_selected, test_specificity_selected, test_gmeasure_selected, test_accuracy_selected, test_auc_selected,
       fs_genus_sensitivity, fs_genus_specificity, fs_genus_gmeasure, fs_genus_accuracy,
       fs_family_sensitivity, fs_family_specificity, fs_family_gmeasure, fs_family_accuracy,
       fs_order_sensitivity, fs_order_specificity, fs_order_gmeasure, fs_order_accuracy,
       fs_class_sensitivity, fs_class_specificity, fs_class_gmeasure, fs_class_accuracy)
  
  write.table(matrix(p, nrow=1), file="s2_output1.csv",append=T,sep = ",", row.names = FALSE,col.names = FALSE)
  return (c(train_sensitivity, train_specificity, train_gmeasure, train_accuracy, train_auc, 
            test_sensitivity, test_specificity, test_gmeasure, test_accuracy, test_auc,
            train_sensitivity_selected, train_specificity_selected, train_gmeasure_selected, train_accuracy_selected, train_auc_selected, 
            test_sensitivity_selected, test_specificity_selected, test_gmeasure_selected, test_accuracy_selected, test_auc_selected,
            fs_genus_sensitivity, fs_genus_specificity, fs_genus_gmeasure, fs_genus_accuracy,
            fs_family_sensitivity, fs_family_specificity, fs_family_gmeasure, fs_family_accuracy,
            fs_order_sensitivity, fs_order_specificity, fs_order_gmeasure, fs_order_accuracy,
            fs_class_sensitivity, fs_class_specificity, fs_class_gmeasure, fs_class_accuracy))
}

set.seed(100)
res <- sapply(seq(1,1000), random_forest_res)

results_table = res %>% t %>% data.frame
colnames(results_table) = c('Train sensitivity', 'Train specificity', 'Train gmeasure', 'Train accuracy', 'Train AUC',
                            'Test sensitivity', 'Test specificity', 'Test gmeasure', 'Test accuracy', 'Test AUC',
                            'train_sensitivity_selected', 'train_specificity_selected', 'train_gmeasure_selected', 'train_accuracy_selected', 'train_auc_selected',
                            'test_sensitivity_selected', 'test_specificity_selected', 'test_gmeasure_selected', 'test_accuracy_selected', 'test_auc_selected',
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
apply(results_table, 2, mean)

print('SD')
apply(results_table, 2, sd)