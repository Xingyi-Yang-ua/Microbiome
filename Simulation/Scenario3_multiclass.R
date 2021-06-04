R.version
library(tidyverse)
library(caret)
library(randomForest)
library(reticulate)
library(mltools)
library(mltest)
library(pROC)
np <- import("numpy")

###############################Configuration#######################################

data_path = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/s3"
path_genus = "/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/genus48"
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count/'

y_path = sprintf('%s/%s', data_path, 'y.csv')
tree_info_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/genus48/genus48_dic.csv'
count_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/count'
count_list_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/gcount_list.csv'
idx_path = '/xdisk/jzhou/mig2020/rsgrps/jzhou/xingyiyang/simulation_updated/data/simulation/s1/idx.csv'

num_classes = 0 # regression
tree_level_list = c('Genus', 'Family', 'Order', 'Class', 'Phylum')

############################Read phylogenetic tree information###########################

phylogenetic_tree_info = read.csv(tree_info_path)
phylogenetic_tree_info = phylogenetic_tree_info %>% select(tree_level_list)

print(sprintf('Phylogenetic tree level list: %s', str_c(phylogenetic_tree_info %>% colnames, collapse = ', ')))

#############################Read dataset##########################################

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

#read true tree weight
tw_1 = np$load(sprintf('%s/tw_1.npy', data_path))
tw_2 = np$load(sprintf('%s/tw_2.npy', data_path))
tw_3 = np$load(sprintf('%s/tw_3.npy', data_path))
tw_4 = np$load(sprintf('%s/tw_4.npy', data_path))

###################################Simulation for all n###################################

random_forest_res <- function(fold, importance_type=2, fs_thrd = 0.1){
  print(sprintf('-----------------------------------------------------------------'))
  print(sprintf('Random Forest computation for %dth repetition', fold))

  dataset = read_dataset(x_path[fold], y_path, fold)
  x_train = dataset[[1]]
  x_test = dataset[[2]]
  y_train = dataset[[3]]
  y_test = dataset[[4]]
  
  # Multicategory classification
  y_train = factor(y_train, levels = c(0,1,2), ordered=TRUE)
  y_test = factor(y_test, levels = c(0,1,2), ordered=TRUE)
  
  fit.rf <- randomForest(y_train~.,data=x_train, ntree=1000,  mtry=14, importance=TRUE)
  train.pred <- fit.rf$predicted
  test.pred <- predict(fit.rf,x_test)
  
  ml_res_train <- ml_test(train.pred, y_train)
  ml_roc_train <- multiclass.roc(y_train, factor(train.pred, levels=c(0,1,2), ordered=TRUE))
  train_recall <- mean(ml_res_train$recall)
  train_specificity <- mean(ml_res_train$specificity)
  train_precision <- mean(ml_res_train$precision)
  train_f1 <-mean(ml_res_train$F1)
  train_gmeasure <- sqrt(train_recall*train_specificity)
  train_accuracy <- ml_res_train$accuracy
  train_auc <- ml_roc_train$auc
  
  ml_res_test <- ml_test(test.pred, y_test)
  ml_roc_test <- multiclass.roc(y_test, factor(test.pred, levels=c(0,1,2), ordered=TRUE))
  test_recall <- mean(ml_res_test$recall)
  test_specificity <- mean(ml_res_test$specificity)
  test_precision <- mean(ml_res_test$precision)
  test_f1 <- mean(ml_res_test$F1)
  test_gmeasure <- sqrt(test_recall*test_specificity)
  test_accuracy <- ml_res_test$accuracy
  test_auc <- ml_roc_test$auc
  
  # Feature selection
  ## variable importance
  #     vi_f = importance(fit.rf, type=importance_type)
  #     relative_vi_f <- vi_f / sum(vi_f)
  #     selected_genus <- ifelse(relative_vi_f >= fs_thrd, 1, 0)
  
  #     order <- order(relative_vi_f, decreasing = TRUE)
  #     sorted_relative_vi_f <- relative_vi_f[order]
  #     names(sorted_relative_vi_f) <- colnames(x_train)[order]
  #     print(sorted_relative_vi_f)
  
  #     fold_genus = apply(tw_1[fold,,], 1, sum)
  #     names(fold_genus) <- x_train %>% colnames
  
  #     fs_conf_table <- table(selected_genus, fold_genus)
  incmse<-fit.rf$importance[,1]
  selection<-incmse[incmse>0]
  selected1<-c(names(selection))
  x_train_selected<- x_train[,selected1]
  x_test_selected<- x_test[,selected1]
  fit.rf_selected <- randomForest(y_train~.,data=x_train_selected, ntree=1000,importance=T)
  train.pred_selected <- fit.rf_selected$predicted
  test.pred_selected <- predict(fit.rf_selected,x_test_selected)
  
  ml_res_train_selected <- ml_test(train.pred_selected, y_train)
  ml_roc_train_selected <- multiclass.roc(y_train, factor(train.pred_selected, levels=c(0,1,2), ordered=TRUE))
  train_recall_selected <- mean(ml_res_train_selected$recall)
  train_specificity_selected <- mean(ml_res_train_selected$specificity)
  train_precision_selected <- mean(ml_res_train_selected$precision)
  train_f1_selected <- mean(ml_res_train_selected$F1)
  train_gmeasure_selected <- sqrt(train_recall*train_specificity)
  train_accuracy_selected <- ml_res_train_selected$accuracy
  train_auc_selected <- ml_roc_train_selected$auc
  
  ml_res_test_selected <- ml_test(test.pred_selected, y_test)
  ml_roc_test_selected <- multiclass.roc(y_test, factor(test.pred_selected, levels=c(0,1,2), ordered=TRUE))
  test_recall_selected <- mean(ml_res_test_selected$recall)
  test_specificity_selected <- mean(ml_res_test_selected$specificity)
  test_precision_selected <- mean(ml_res_test_selected$precision)
  test_f1_selected <- mean(ml_res_test_selected$F1)
  test_gmeasure_selected <- sqrt(test_recall*test_specificity)
  test_accuracy_selected <- ml_res_test_selected$accuracy
  test_auc_selected <- ml_roc_test_selected$auc
  
  selected_genus<- ifelse(incmse >0, 1, 0)

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
  
  #######################################performance####################################
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
  
  
  print(sprintf('Train recall: %s, Train precision: %s, Train gmeasure: %s, Train accuracy: %s, Train AUC: %s, Train F1: %s',
                train_recall, train_precision, train_gmeasure, train_accuracy, train_auc, train_f1))
  print(sprintf('Test recall: %s, Test precision: %s, Test gmeasure: %s, Test accuracy: %s, Test AUC: %s, Test F1: %s',
                test_recall, test_precision, test_gmeasure, test_accuracy, test_auc, test_f1))
  print(sprintf('Train recall_selected: %s, Train precision_selected: %s, Train gmeasure_selected: %s, Train accuracy_selected: %s, Train AUC_selected: %s, Train F1_selected: %s',
                train_recall_selected, train_precision_selected, train_gmeasure_selected, train_accuracy_selected, train_auc_selected, train_f1_selected))
  print(sprintf('Test recall_selected: %s, Test precision_selected: %s, Test gmeasure_selected: %s, Test accuracy_selected: %s, Test AUC_selected: %s, Test F1_selected: %s',
                test_recall_selected, test_precision_selected, test_gmeasure_selected, test_accuracy_selected, test_auc_selected, test_f1_selected))
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
  
  return (c(train_recall, train_precision, train_gmeasure, train_accuracy, train_auc, train_f1,
            test_recall, test_precision, test_gmeasure, test_accuracy, test_auc, test_f1,
            train_recall_selected, train_precision_selected, train_gmeasure_selected, train_accuracy_selected, train_auc_selected, train_f1_selected,
            test_recall_selected, test_precision_selected, test_gmeasure_selected, test_accuracy_selected, test_auc_selected, test_f1_selected,
            fs_genus_sensitivity, fs_genus_specificity, fs_genus_gmeasure, fs_genus_accuracy,
            fs_family_sensitivity, fs_family_specificity, fs_family_gmeasure, fs_family_accuracy,
            fs_order_sensitivity, fs_order_specificity, fs_order_gmeasure, fs_order_accuracy,
            fs_class_sensitivity, fs_class_specificity, fs_class_gmeasure, fs_class_accuracy))
}

set.seed(100)
res <- sapply(seq(1,1000), random_forest_res)

results_table = res %>% t %>% data.frame
colnames(results_table) = c('Train recall', 'Train precision', 'Train gmeasure', 'Train accuracy', 'Train AUC','Train F1',
                            'Test recall', 'Test precision', 'Test gmeasure', 'Test accuracy', 'Test AUC','Test F1',
                            'Train recall_selected', 'Train precision_selected','Train gmeasure_selected','Train accuracy_selected','Train AUC_selected', 'Train F1_selected',
                            'Test recall_selected', 'Test precision_selected','Test gmeasure_selected','Test accuracy_selected','Test AUC_selected', 'Test F1_selected',
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