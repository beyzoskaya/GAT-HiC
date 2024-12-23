import numpy as np
from ge import LINE
import sys
import utils
import networkx as nx
import os
from models import Net
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if not(os.path.exists('mESC_Hindll_1mb_train_and_tested/Outputs_GNN')):
        os.makedirs('mESC_Hindll_1mb_train_and_tested/Outputs_GNN')

    if not(os.path.exists('Data_NCol_Hindlll_same_train_test_mESC/Data_GNN_mESC_Hindlll')):
        os.makedirs('Data_NCol_Hindlll_same_train_test_mESC/Data_GNN_mESC_Hindlll')

    # Argument parsing
    parser = argparse.ArgumentParser(description='Generalize a trained model to new data using the same resolution.')
    parser.add_argument('list_trained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_trained.')
    parser.add_argument('list_untrained', type=str, help='File path for list format of raw Hi-C corresponding to embeddings_untrained.')
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs used for embeddings generation')
    parser.add_argument('-lr', '--learningrate', type=float, default=.001, help='Learning rate for training GCNN.')
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
        data_dir = 'Data_NCol_Hindlll_same_train_test_mESC/Data_GNN_mESC_Hindlll'
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

            embed = LINE(G, embedding_size=512, order='second')
            embed.train(batch_size=batch_size, epochs=epochs, verbose=1)
            embeddings = embed.get_embeddings()
            embeddings = list(embeddings.values())
            embeddings = np.asarray(embeddings)

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

    model_weight_path = f'mESC_Hindll_1mb_train_and_tested/Outputs_GNN/{name_trained}_weights.pt'
    if not(os.path.isfile(model_weight_path)):
        print(f'Failed to find model weights corresponding to {filepath_trained} from {model_weight_path}')

        print(f'Training model using conversion value {conversion}.')
        model = Net()
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        oldloss = 1
        lossdiff = 1
        truth = utils.cont2dist(data_trained.y, conversion)

        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()
            out = model(data_trained.x.float(), data_trained.edge_index)
            loss = criterion(out.float(), truth.float())
            lossdiff = abs(oldloss - loss)
            loss.backward()
            optimizer.step()
            oldloss = loss
            print(f'Loss: {loss}', end='\r')

        torch.save(model.state_dict(), model_weight_path)
        print(f'Saved trained model corresponding to {filepath_trained} to {model_weight_path}')

    # Load model for evaluation
    model = Net()
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    # Load untrained data for evaluation
    data_untrained = utils.load_input(normed_untrained, embeddings_untrained)

    # Evaluate the model on untrained data
    temp_spear = []
    temp_models = []

    truth = utils.cont2dist(data_untrained.y, conversion).float()

    idx = torch.triu_indices(data_untrained.y.shape[0], data_untrained.y.shape[1], offset=1)
    dist_truth = truth[idx[0, :], idx[1, :]].detach().numpy()
    coords = model.get_model(data_untrained.x.float(), data_untrained.edge_index)
    out = torch.cdist(coords, coords)
    dist_out = out[idx[0, :], idx[1, :]].detach().numpy()

    # Calculate dSCC
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
    plt.savefig(f'mESC_Hindll_1mb_train_and_tested/Outputs_GNN/{name_untrained}_distance_comparison_plot.png')

    utils.WritePDB(repmod * 100, f'mESC_Hindll_1mb_train_and_tested/Outputs_GNN/{name_untrained}_generalized_structure.pdb')

    with open(f'mESC_Hindll_1mb_train_and_tested/Outputs_GNN/{name_untrained}_generalized_log.txt', 'w') as f:
        line1 = f'Optimal dSCC: {repspear}\n'
        f.writelines([line1])
