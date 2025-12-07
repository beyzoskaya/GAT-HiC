# GAT-HiC Efficient Reconstruction of 3D Chromosome Structure Via Graph Attention Neural Network 
****Check our paper [here](https://doi.org/10.1109/TCBBIO.2025.3633195).****
------------------------------------------------------------------------------------------------------------------------------------
## Overview
GAT-HiC is a method developed to predict three-dimensional chromosome structure from Hi-C interaction data. GAT-HiC can generalize to unseen Hi-C datasets, enabling prediction across various cell populations, restriction enzymes, and Hi-C resolutions. This method combines the unsupervised vertex embedding technique Node2vec with an attention-based graph neural network to predict the 3D coordinates of genomic loci.

## Hi-C Data
------------------------------------------------------------------------------------------------------------------------------------
The Hi-C data used in this project consists of contact maps for each species, where each contact map represents the interaction frequencies between genomic loci across chromosomes. These interaction matrices are crucial for predicting the three-dimensional (3D) structure of chromosomes, providing a foundation for the GATHiC method to infer the 3D coordinates from these interaction patterns.

## Dependencies
The required libraries for running this project are listed in the `requirements.txt` file. To install all the necessary libraries, follow these steps:

### Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/beyzoskaya/GAT-HiC.git
```
### Set Up the Environment
Navigate to the project directory and install the dependencies from the `requirements.txt` file:
```bash
cd GAT-HiC
conda create -n gat_hic python=3.7.12 
pip install -r requirements.txt
```

----------------------------------------------------------------------------------------------------------------------------------	
## Key Scripts 

### 1. `GAT-HiC_generalize.py`

#### Purpose:
This file focuses on the process of training a Graph Attention Network (GAT) model on Hi-C data embeddings, followed by generalizing the trained model to untrained Hi-C data. The model is trained using a combined loss function that includes:

- **MSE Loss**: Minimizes the error between predicted and true distances of genomic loci.
- **Pearson Correlation Loss**: Encourages the model to maintain structural relationships by maximizing the Pearson correlation between predicted and true distances.

The final loss is a weighted sum of these two components, with a dynamic weight (`alpha`) applied to the Pearson correlation loss to balance both terms during training. This helps the model generalize across different Hi-C resolutions.

#### Inputs:
- **`list_trained`**: A file containing the trained Hi-C data in a list format.
- **`list_untrained`**: A file containing the untrained Hi-C data in a list format.
- **`batchsize`** (optional): The batch size used for embeddings generation (default is 128).
- **`epochs`** (optional): The number of epochs for model training (default is 1000).
- **`learningrate`** (optional): The learning rate for training (default is 0.001).
- **`loss_diff_threshold`** (optional): A threshold for early stopping based on loss difference (default is 1e-8).
  
#### Outputs:
- **Trained Model**: The model weights are saved as `weights.pt` in the output directory.
- **Predicted Structure**: The 3D structure of the predicted distances is saved in a `.pdb` file.
- **Training and Validation Plots**: Graphs showing the MSE loss, Spearman correlation, and training/validation loss curves.
- **Results File**: A text file containing the optimal dSCC (Spearman correlation), final loss, and final combined loss.

#### Workflow:
1. **Preprocessing**: The Hi-C matrices are normalized using the KR normalization, and node2vec embeddings are generated for both trained and untrained data.
2. **Model Training**: A GAT model is trained on the preprocessed Hi-C data using a combined loss function that minimizes MSE and a regularization term (Spearman correlation loss).
3. **Generalization**: Once the model is trained, it is tested on untrained Hi-C data, and the performance is evaluated using Spearman's correlation coefficient.
4. **Outputs**: The optimal model and the predicted structure are saved, and performance metrics (e.g., dSCC) are reported.

#### Usage:
To run the script `GAT-HiC_generalize.py`, use the following command in the terminal:

```bash
python GAT-HiC_generalize.py <list_trained> <list_untrained> [-bs BATCHSIZE] [-ep EPOCHS] [-lr LEARNINGRATE] [-loss_diff_threshold LOSS_DIFF_THRESHOLD]
```

Example:
```bash
python GAT-HiC_generalize.py "Hi-C_Data/GM12878/1mb/chr18_1mb_RAWobserved.txt" "Hi-C_dataset/GM12878/500kb/chr18_500kb.RAWobserved.txt"
```

### 2. `GAT-HiC_same_resolution.py`

#### Purpose:
This script trains a Graph Attention Network (GAT) model on Hi-C data embeddings and then generalizes the model to untrained Hi-C data at the same resolution. It focuses on predicting pairwise genomic distances from the embeddings and adjacency matrices and evaluates the model using Spearman's correlation.

The model is trained using a **Contrastive Loss** that minimizes the absolute difference between the true and predicted distances. The loss function is defined as:

- **Contrastive Loss**: Penalizes the absolute difference between the predicted and true pairwise distances, encouraging the model to preserve relative genomic distances.

The final training objective is to minimize this contrastive loss to better capture the underlying genomic structure.

#### Inputs:
- **`list_trained`**: A file containing the trained Hi-C data in list format.
- **`list_untrained`**: A file containing the untrained Hi-C data in list format.
- **`batchsize`** (optional): The batch size used for embeddings generation (default is 128).
- **`epochs`** (optional): The number of epochs for embeddings generation (default is 10).
- **`learningrate`** (optional): The learning rate for training (default is 0.0001).
- **`threshold`** (optional): The loss threshold for early stopping (default is 1e-8).

#### Outputs:
- **Trained Model**: The model weights are saved as `{name_trained}_weights.pt` in the output directory.
- **Training Loss Plot**: A plot of training loss over iterations saved as `{name_trained}_training_loss_plot.png`.
- **Predicted Structure**: The predicted 3D structure of the untrained data is saved in a `.pdb` file (`{name_untrained}_generalized_structure.pdb`).
- **Distance Comparison Plot**: A plot comparing the true vs predicted distances for the untrained data saved as `{name_untrained}_distance_comparison_plot.png`.
- **Results File**: A text file containing the optimal dSCC (Spearman correlation), saved as `{name_untrained}_generalized_log.txt`.

#### Workflow:
1. **Preprocessing**: The Hi-C matrices for both trained and untrained datasets are either loaded from existing files or generated from raw data if not already present. These matrices are normalized using KR normalization, and Node2Vec embeddings are generated for both datasets.
2. **Model Training**: A GAT model is trained on the preprocessed Hi-C data (trained dataset) using a combined loss function that minimizes MSE and a contrastive loss term that penalizes absolute differences in predicted and true distances.
3. **Testing**: After training, the model is tested on the same resolution HiC data, and the performance is evaluated using Spearman's correlation coefficient.
4. **Outputs**: The optimal model weights are saved, and the predicted structure for the untrained Hi-C data is stored in `.pdb` format. Performance metrics (e.g., dSCC) are reported, and training loss plots are saved to visualize the training process.

#### Usage:
To run the script `GAT-HiC_same_resolution.py`, use the following command in the terminal:
```bash
python GAT-HiC_same_resolution.py <list_trained> <list_untrained> [-bs BATCHSIZE] [-ep EPOCHS] [-lr LEARNINGRATE] [-th THRESHOLD]
```
Example:
```bash
python GAT-HiC_same_resolution.py "Hi-C_Data/GM12878/1mb/chr18_1mb_RAWobserved.txt" "Hi-C_dataset/GM12878/1mb/chr18_1mb_RAWobserved.txt"
```
