#!/usr/bin/env Rscript
source('r_utils.R')
args = commandArgs(trailingOnly=TRUE)
filename = args[1]
print(filename)
filepath = paste("Data/GATNetSelectiveResidualsUpdated_embedding_512_batch_128_lr_0.001_threshold_1e-8_p_1.75_q_0.4_walk_length_50_num_walks_150_GM12878_generalization_pearson_combined_loss_dynamic_alpha_500kb/", filename, ".txt", sep = "")

adj = as.matrix(read.table(filepath, sep="\t"))
normed = KRnorm(adj)
new_name = paste("Data/GATNetSelectiveResidualsUpdated_embedding_512_batch_128_lr_0.001_threshold_1e-8_p_1.75_q_0.4_walk_length_50_num_walks_150_GM12878_generalization_pearson_combined_loss_dynamic_alpha_500kb/", filename, "_KR_normed.txt", sep = "")
write.table(normed, file= new_name, row.names=FALSE, col.names=FALSE)
