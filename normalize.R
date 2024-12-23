#!/usr/bin/env Rscript
source('r_utils.R')
args = commandArgs(trailingOnly=TRUE)
filename = args[1]
print(filename)
filepath = paste("Data_NCol_Hindlll_same_train_test_mESC/Data_GNN_mESC_Hindlll/", filename, ".txt", sep = "")

adj = as.matrix(read.table(filepath, sep="\t"))
normed = KRnorm(adj)
new_name = paste("Data_NCol_Hindlll_same_train_test_mESC/Data_GNN_mESC_Hindlll/", filename, "_KR_normed.txt", sep = "")
write.table(normed, file= new_name, row.names=FALSE, col.names=FALSE)
