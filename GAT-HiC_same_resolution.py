import numpy as np
from node2vec import Node2Vec
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
"""
In same resolution evaluation, node2vec embeddings used for GATNetSelectiveResidualsUpdated model.
Model trained on list_trained if there is no weight file appear then used that weight to evaluate on list_untrained
"""

if __name__ == "__main__":
    """
    Those two paths created for Data and Output files storage. Data folder include preprocess version of HiC contact map while Output folder include plottings and weights. 
    """
    if not(os.path.exists('250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878')):
        os.makedirs('250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878')

    if not(os.path.exists('Data_same_resolution/Data_250kb_GAT_GM12878')):
        os.makedirs('Data_same_resolution/Data_250kb_GAT_GM12878')

    # Argument parsing
    parser = argparse.ArgumentParser(description='Generalize a trained model to new data using the same resolution.')
    parser.add_argument('list_trained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_trained.')
    parser.add_argument('list_untrained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_untrained.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation')
    parser.add_argument('-lr', '--learningrate', type=float, default=.0001, help='Learning rate for training GCNN.')
    parser.add_argument('-th', '--threshold', type=float, default=1e-8, help='Loss threshold for training termination.')

    args = parser.parse_args()

    filepath_trained = args.list_trained
    filepath_untrained = args.list_untrained
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold

    conversion = 1

    name_trained = os.path.splitext(os.path.basename(filepath_trained))[0]
    name_untrained = os.path.splitext(os.path.basename(filepath_untrained))[0]

    list_trained = np.loadtxt(filepath_trained)
    list_untrained = np.loadtxt(filepath_untrained)

    for name, list_data, is_trained in [(name_trained, list_trained, True), (name_untrained, list_untrained, False)]:
        data_dir = 'Data_same_resolution/Data_250kb_GAT_GM12878'
        adj_matrix_path = f'{data_dir}/{name}_matrix.txt'
        
        if not(os.path.isfile(adj_matrix_path)):
            print(f'Failed to find matrix form of {name} from {adj_matrix_path}.')
            adj = utils.convert_to_matrix(list_data)
            np.fill_diagonal(adj, 0) 
            np.savetxt(adj_matrix_path, adj, delimiter='\t')
            print(f'Created matrix form of {name} as {adj_matrix_path}.')
        else:
            adj = np.loadtxt(adj_matrix_path)
        
        norm_matrix_path = f'{data_dir}/{name}_matrix_KR_normed.txt'
        if not(os.path.isfile(norm_matrix_path)):
            print(f'Failed to find normalized matrix form of {name} from {norm_matrix_path}')
            os.system(f'Rscript normalize.R {name}_matrix')
            print(f'Created normalized matrix form of {name} as {norm_matrix_path}')
        adj_normalized = np.loadtxt(norm_matrix_path)

        embedding_path = f'{data_dir}/{name}_embeddings.txt'
        if not(os.path.isfile(embedding_path)):
            print(f'Failed to find embeddings corresponding to {name} from {embedding_path}')
            G = nx.from_numpy_matrix(adj)

            node2vec = Node2Vec(G, dimensions=512, walk_length=150, num_walks=50, p=1.75, q=0.4, workers=1, seed=42)
            embeddings = node2vec.fit(window=25, min_count=1, batch_words=4)
            embeddings = np.array([embeddings.wv[str(node)] for node in G.nodes()])
            np.savetxt(embedding_path, embeddings)
            print(f'Created embeddings corresponding to {name} as {embedding_path}.')
        embeddings = np.loadtxt(embedding_path)

        if is_trained:
            normed_trained = adj_normalized
            embeddings_trained = embeddings
        else:
            normed_untrained = adj_normalized
            embeddings_untrained = embeddings

    data_trained = utils.load_input(normed_trained, embeddings_trained)

    model_weight_path = f'250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878/{name_trained}_weights.pt'
    if not(os.path.isfile(model_weight_path)):
        print(f'Failed to find model weights corresponding to {filepath_trained} from {model_weight_path}')

        print(f'Training model using conversion value {conversion}.')
        model = GATNetSelectiveResidualsUpdated()
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        oldloss = 1
        lossdiff = 1
        truth = utils.cont2dist(data_trained.y, conversion)

        loss_history = []
        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()

            out = model(data_trained.x.float(), data_trained.edge_index)

            idx = torch.triu_indices(data_trained.y.shape[0], data_trained.y.shape[1], offset=1)
            dist_truth = truth[idx[0, :], idx[1, :]]
            coords = model.get_model(data_trained.x.float(), data_trained.edge_index)
            dist_out = torch.cdist(coords, coords)[idx[0, :], idx[1, :]]

            mse_loss = criterion(out.float(), truth.float())
            PearsonR, _ = pearsonr(dist_truth.detach().numpy(), dist_out.detach().numpy())

            """
            Contrastive loss penalizes the absolute difference between the predicted and true distances
            focusing on relative differences rather than absolute magnitudes.
            true pairwise distance between nodes i and j - predicted pairwise distance between nodes i and j --> contrastive loss
            More sensitive loss on local structure
            """
            total_loss = 0.0
            contrastive_loss = torch.mean(torch.abs(dist_truth - dist_out))
            #print(f"Contrastive loss: {contrastive_loss}")
            total_loss = total_loss + 0.1 * contrastive_loss
            #print(f"Total loss: {total_loss}")

            lossdiff = abs(oldloss - total_loss.item())
            total_loss.backward()
            optimizer.step()
            oldloss = total_loss.item()

            loss_history.append(total_loss.item())
            print(f'Loss: {total_loss}', end='\r')

        torch.save(model.state_dict(), model_weight_path)
        print(f'Saved trained model corresponding to {filepath_trained} to {model_weight_path}')

        # Plot the training loss history
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loss_history)), loss_history, label='Training Loss', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Iterations')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878/{name_trained}_training_loss_plot.png')
        plt.show()

    # evaluate on same resolution data with trained weight
    model = GATNetSelectiveResidualsUpdated()
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    data_untrained = utils.load_input(normed_untrained, embeddings_untrained)

    temp_spear = []
    temp_models = []

    truth = utils.cont2dist(data_untrained.y, conversion).float()

    idx = torch.triu_indices(data_untrained.y.shape[0], data_untrained.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]].detach().numpy()
    coords = model.get_model(data_untrained.x.float(), data_untrained.edge_index)
    out = torch.cdist(coords, coords)
    dist_out = out[idx[0, :], idx[1, :]].detach().numpy()

    SpRho = spearmanr(dist_truth, dist_out)[0]
    temp_spear.append(SpRho)
    temp_models.append(coords)

    idx = temp_spear.index(max(temp_spear))
    repspear = temp_spear[idx]
    repmod = temp_models[idx]

    print(f'Optimal dSCC for generalized data: {repspear}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dist_truth)), dist_truth, label='True Distances', color='blue')
    plt.plot(range(len(dist_out)), dist_out, label='Predicted Distances', color='orange')
    plt.xlabel('Pairwise Distances Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig(f'250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878/{name_untrained}_distance_comparison_plot.png')

    utils.WritePDB(repmod * 100, f'250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878/{name_untrained}_generalized_structure.pdb')

    with open(f'250kb_train_tested_on_250kb/Outputs_GATNetSelectiveResidualsUpdated_lr_0.0001_threshold_1e-8_GM12878/{name_untrained}_generalized_log.txt', 'w') as f:
        line1 = f'Optimal dSCC: {repspear}\n'
        f.writelines([line1])
