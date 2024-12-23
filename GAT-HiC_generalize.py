import numpy as np
from node2vec import Node2Vec
import sys
import utils
import networkx as nx
import os
from models import GATNetSelectiveResidualsUpdated
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr,pearsonr
import argparse
import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer

if __name__ == "__main__":
    """
    Those Data and Output folders used to store for preprocessed input values and plottings/weights/pdb files.
    """
    
    base_data_dir = 'Data/GATNetSelectiveResidualsUpdated_embedding_512_batch_128_lr_0.001_threshold_1e-8_p_1.75_q_0.4_walk_length_50_num_walks_150_GM12878_generalization_pearson_combined_loss_dynamic_alpha_500kb'
    base_output_dir = 'Generalization_500kb_resolution_GM12878/GATNetSelectiveResidualsUpdated_embedding_512_batch_128_lr_0.001_threshold_1e-8_p_1.75_q_0.4_walk_length_50_num_walks_150_GM12878_generalization_pearson_combined_loss_dynamic_alpha_500kb'

    if not(os.path.exists(base_data_dir)):
        os.makedirs(base_data_dir)

    if not(os.path.exists(base_output_dir)):
        os.makedirs(base_output_dir)

    parser = argparse.ArgumentParser(description='Generalize a trained model to new data using combined loss.')
    parser.add_argument('list_trained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_trained.')
    parser.add_argument('list_untrained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_untrained.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=1000, help='Number of epochs used for model training.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='Learning rate for training GCNN.') 
    parser.add_argument('-thresh', '--loss_diff_threshold', type=float, default=1e-8, help='Loss difference threshold for early stopping.')
    parser.add_argument('-print_interval', type=int, default=10, help='Interval for printing MSE and dSCC values.')
    #parser.add_argument('-alpha', '--alpha', type=float, default=0.1, help='Weight for dSCC loss component in combined loss.')

    args = parser.parse_args()

    filepath_trained = args.list_trained
    filepath_untrained = args.list_untrained
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    #alpha = args.alpha
    epochs = args.epochs
    loss_diff_threshold = args.loss_diff_threshold
    print_interval = args.print_interval
    conversion = 1

    name_trained = os.path.splitext(os.path.basename(filepath_trained))[0]
    name_untrained = os.path.splitext(os.path.basename(filepath_untrained))[0]

    list_trained = np.loadtxt(filepath_trained)
    list_untrained = np.loadtxt(filepath_untrained)
    print(f"Loaded {len(list_trained)} entries for trained and {len(list_untrained)} for untrained data.")

    # load matrix for trained
    if not(os.path.isfile(f'{base_data_dir}/{name_trained}_matrix.txt')):
        print(f'Failed to find matrix form of {filepath_trained} from {base_data_dir}/{name_trained}_matrix.txt.')
        adj_trained = utils.convert_to_matrix(list_trained)
        np.fill_diagonal(adj_trained, 0) 
        np.savetxt(f'{base_data_dir}/{name_trained}_matrix.txt', adj_trained, delimiter='\t')
        print(f'Created matrix form of {filepath_trained} as {base_data_dir}/{name_trained}_matrix.txt.')
    matrix_trained = np.loadtxt(f'{base_data_dir}/{name_trained}_matrix.txt')
    print(f"Matrix trained shape: {matrix_trained.shape}")

    # load matrix for untrained
    if not(os.path.isfile(f'{base_data_dir}/{name_untrained}_matrix.txt')):
        print(f'Failed to find matrix form of {filepath_untrained} from {base_data_dir}/{name_untrained}_matrix.txt.')
        adj_untrained = utils.convert_to_matrix(list_untrained)
        np.fill_diagonal(adj_untrained, 0) # self loops handled with diagonal elements
        np.savetxt(f'{base_data_dir}/{name_untrained}_matrix.txt', adj_untrained, delimiter='\t')
        print(f'Created matrix form of {filepath_untrained} as {base_data_dir}/{name_untrained}_matrix.txt.')
    matrix_untrained = np.loadtxt(f'{base_data_dir}/{name_untrained}_matrix.txt')
    print(f"Matrix untrained shape: {matrix_untrained.shape}")

    # apply KR normalization to matrix trained
    if not(os.path.isfile(f'{base_data_dir}/{name_trained}_matrix_KR_normed.txt')):
        print(f'Failed to find normalized matrix form of {filepath_trained} from {base_data_dir}/{name_trained}_matrix_KR_normed.txt')
        os.system(f'Rscript normalize.R {name_trained}_matrix')
        print(f'Created normalized matrix form of {filepath_trained} as {base_data_dir}/{name_trained}_matrix_KR_normed.txt')
    normed_trained = np.loadtxt(f'{base_data_dir}/{name_trained}_matrix_KR_normed.txt')
    print(f"Normalized trained matrix stats: mean={np.mean(normed_trained)}, std={np.std(normed_trained)}, min={np.min(normed_trained)}, max={np.max(normed_trained)}")

    # apply KR normalization to matrix untrained
    if not(os.path.isfile(f'{base_data_dir}/{name_untrained}_matrix_KR_normed.txt')):
        print(f'Failed to find normalized matrix form of {filepath_untrained} from {base_data_dir}/{name_untrained}_matrix_KR_normed.txt')
        os.system(f'Rscript normalize.R {name_untrained}_matrix')
        print(f'Created normalized matrix form of {filepath_untrained} as {base_data_dir}/{name_untrained}_matrix_KR_normed.txt')
    normed_untrained = np.loadtxt(f'{base_data_dir}/{name_untrained}_matrix_KR_normed.txt')
    print(f"Normalized untrained matrix stats: mean={np.mean(normed_untrained)}, std={np.std(normed_untrained)}, min={np.min(normed_untrained)}, max={np.max(normed_untrained)}")


    # Create node2vec embeddings for the trained data
    if not(os.path.isfile(f'{base_data_dir}/{name_trained}_embeddings.txt')):
        print(f'Failed to find embeddings corresponding to {filepath_trained} from {base_data_dir}/{name_trained}_embeddings.txt')
        G = nx.from_numpy_matrix(matrix_trained)

        # embedding creation for trained
        node2vec_trained = Node2Vec(G, dimensions=512, walk_length=150, num_walks=50, p=1.75, q=0.4, workers=1, seed=42)
        embeddings_trained = node2vec_trained.fit(window=25, min_count=1, batch_words=4)
        embeddings_trained = np.array([embeddings_trained.wv[str(node)] for node in G.nodes()])
        np.savetxt(f'{base_data_dir}/{name_trained}_embeddings.txt', embeddings_trained)
        print(f'Created embeddings corresponding to {filepath_trained} as {base_data_dir}/{name_trained}_embeddings.txt.')
    embeddings_trained = np.loadtxt(f'{base_data_dir}/{name_trained}_embeddings.txt')
    print(f"Trained embeddings stats: shape={embeddings_trained.shape}, mean={np.mean(embeddings_trained)}, std={np.std(embeddings_trained)}, min={np.min(embeddings_trained)}, max={np.max(embeddings_trained)}")

    # Create node2vec embeddings for the untrained data
    if not(os.path.isfile(f'{base_data_dir}/{name_untrained}_embeddings.txt')):
        print(f'Failed to find embeddings corresponding to {filepath_untrained} from {base_data_dir}/{name_untrained}_embeddings.txt')
        G = nx.from_numpy_matrix(matrix_untrained)

        # embedding creation for untrained
        node2vec_untrained = Node2Vec(G, dimensions=512, walk_length=150, num_walks=50, p=1.75, q=0.4, workers=1, seed=42)
        embeddings_untrained = node2vec_untrained.fit(window=25, min_count=1, batch_words=4)
        embeddings_untrained = np.array([embeddings_untrained.wv[str(node)] for node in G.nodes()])
        np.savetxt(f'{base_data_dir}/{name_untrained}_embeddings.txt', embeddings_untrained)
        print(f'Created embeddings corresponding to {filepath_untrained} as {base_data_dir}/{name_untrained}_embeddings.txt.')
    embeddings_untrained = np.loadtxt(f'{base_data_dir}/{name_untrained}_embeddings.txt')
    print(f"Untrained embeddings stats: shape={embeddings_untrained.shape}, mean={np.mean(embeddings_untrained)}, std={np.std(embeddings_untrained)}, min={np.min(embeddings_untrained)}, max={np.max(embeddings_untrained)}")

    data_trained = utils.load_input(normed_trained, embeddings_trained)
    data_untrained = utils.load_input(normed_untrained, embeddings_untrained)
    print(f"Data shapes: Trained (x)={data_trained.x.shape}, Trained (y)={data_trained.y.shape}")
    #print(f"Data shapes: Untrained (x)={data_untrained.x.shape}, Trained (y)={data_untrained.y.shape}")

    
    # Train the model using a fixed number of epochs and combined loss
    if not(os.path.isfile(f'{base_output_dir}/{name_trained}_weights.pt')):
        print(f'Failed to find model weights corresponding to {filepath_trained} from {base_output_dir}/{name_trained}_weights.pt')
        model = GATNetSelectiveResidualsUpdated()

        criterion_mse = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        val_loss_history = []
        loss_history = []  
        dSCC_history = []
        mse_history = []  
        iteration = 0
        #gradient_history = {}

        loss_diff = 1
        old_loss = 1
        truth = utils.cont2dist(data_trained.y, conversion)
        
        
        #for epoch in range(epochs):
        while loss_diff > loss_diff_threshold:
            model.train()
            optimizer.zero_grad()

            out = model(data_trained.x.float(), data_trained.edge_index)
            #print(f"Output shape: {out.shape}, Expected shape: {data_trained.y.shape}")
           

            idx = torch.triu_indices(data_trained.y.shape[0], data_trained.y.shape[1], offset=1)
            dist_truth = truth[idx[0, :], idx[1, :]]
            dist_truth_np = dist_truth.detach().numpy()
            coords = model.get_model(data_trained.x.float(), data_trained.edge_index)
            dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]  # Pairwise Euclidean distances
            #print(f"Computed pairwise distances, shape: {dist_out.shape}")

            dist_out_np = dist_out.detach().numpy() 
            
            mse_loss = criterion_mse(out.float(), truth.float())
            PearsonR, _ = pearsonr(dist_truth.detach().numpy(), dist_out_np)
            
            PearsonR_loss =  (1 - PearsonR)
            alpha_initial = 0.1
            alpha = min(1.0, alpha_initial + (1.0 / (mse_loss.item() + 1e-6)))
            total_loss = mse_loss + alpha * PearsonR_loss
            #total_loss = mse_loss - 0.2 * PearsonR_loss # different options for loss function to get sensitivity of the contact maps
    
            loss_diff = abs(old_loss - total_loss.item())
            total_loss.backward()

            optimizer.step()
            old_loss = total_loss.item()
            #loss_history.append(combined_loss.item())
            loss_history.append(total_loss.item())
            SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

            #print(f"Iteration [{iteration}]:")
            #print(f"  True Distances - Mean: {dist_truth.mean().item():.4f}, Std: {dist_truth.std().item():.4f}, Min: {dist_truth.min().item():.4f}, Max: {dist_truth.max().item():.4f}")
            #print(f"  Predicted Distances - Mean: {dist_out.mean().item():.4f}, Std: {dist_out.std().item():.4f}, Min: {dist_out.min().item():.4f}, Max: {dist_out.max().item():.4f}")
            dSCC_history.append(SpRho)

            if iteration % print_interval == 0:
                print(f"Iteration [{iteration}], Total Loss: {mse_loss.item()}, dSCC: {SpRho}, Loss Diff: {loss_diff}")

            final_mse_loss = mse_loss.item()
            final_combined_loss = total_loss.item()
            iteration += 1


        print(f'\nOptimal dSCC after training: {SpRho}')
        print(f"Pearson Correlation: {PearsonR}")

        #mse_mean, mse_max = np.mean(smoothed_mse_loss), np.max(smoothed_mse_loss)
        dSCC_mean, dSCC_max = np.mean(dSCC_history), np.max(dSCC_history)

        #print(f"\nMSE Loss - Mean: {smoothed_mse_loss}, Max: {smoothed_mse_loss}")
        print(f"dSCC Loss - Mean: {dSCC_mean}, Max: {dSCC_max}")

        torch.save(model.state_dict(), f'{base_output_dir}/{name_trained}_weights.pt')
        utils.WritePDB(coords * 100, f'{base_output_dir}/{name_trained}_structure.pdb')
        print(f'Saved trained model to {base_output_dir}/{name_trained}_weights.pt')
        print(f'Saved optimal structure to {base_output_dir}/{name_trained}_structure.pdb')

        with open(f'{base_output_dir}/{name_trained}_results.txt', 'w') as f:
            f.writelines([f'Optimal dSCC: {SpRho}\n', f'Final loss: {final_mse_loss}\n', f'Final Combined loss: {final_combined_loss}\n' ])


        plt.figure(figsize=(10, 6))
        plt.plot(mse_history, label='MSE Loss', color='red')
        plt.plot(dSCC_history, label='dSCC (Spearman Correlation)', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        plt.title(f'MSE and dSCC Curves for {name_trained}')
        plt.legend()
        plt.savefig(f'{base_output_dir}/{name_trained}_mse_dSCC_plot.png')

        # Plot Training and Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for {name_trained}')
        plt.legend()
        plt.savefig(f'{base_output_dir}/{name_trained}_training_validation_loss_plot.png')

    # Generalize to untrained data
    model = GATNetSelectiveResidualsUpdated()
    model.load_state_dict(torch.load(f'{base_output_dir}/{name_trained}_weights.pt'))
    model.eval()

    fitembed = utils.domain_alignment(list_trained, list_untrained, embeddings_trained, embeddings_untrained)
    data_untrained_fit = utils.load_input(normed_untrained, fitembed)

    truth = utils.cont2dist(data_untrained_fit.y, conversion).float()
    idx = torch.triu_indices(data_untrained_fit.y.shape[0], data_untrained_fit.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]].detach().numpy()
    coords = model.get_model(data_untrained_fit.x.float(), data_untrained_fit.edge_index)
    out = torch.cdist(coords, coords)
    dist_out = out[idx[0, :], idx[1, :]].detach().numpy()

    SpRho_generalization = spearmanr(dist_truth, dist_out)[0]
    print(f'Optimal dSCC for generalized data: {SpRho_generalization}')


    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dist_truth)), dist_truth, label='True Distances', color='blue')
    plt.plot(range(len(dist_out)), dist_out, label='Predicted Distances', color='orange')
    plt.xlabel('Pairwise Distances Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name_untrained}_distance_comparison_plot.png')

    # Scatter Plot (True vs Predicted Distances)
    plt.figure(figsize=(10, 6))
    plt.scatter(dist_truth, dist_out, alpha=0.5)
    plt.plot([dist_truth.min(), dist_truth.max()], [dist_truth.min(), dist_truth.max()], 'r--', label='Ideal (y = x)')
    plt.xlabel('True Pairwise Distance')
    plt.ylabel('Predicted Pairwise Distance')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name_untrained}_scatter_distance_comparison_plot.png')

    # Histogram of True and Predicted Distances
    plt.figure(figsize=(10, 6))
    plt.hist(dist_truth, bins=50, alpha=0.5, label='True Distances', color='blue')
    plt.hist(dist_out, bins=50, alpha=0.5, label='Predicted Distances', color='orange')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{base_output_dir}/{name_untrained}_histogram_distance_comparison_plot.png')

    utils.WritePDB(coords * 100, f'{base_output_dir}/{name_untrained}_generalized_structure.pdb')

    with open(f'{base_output_dir}/{name_untrained}_generalized_log.txt', 'w') as f:
        f.writelines([f'Optimal dSCC: {SpRho_generalization}\n'])
