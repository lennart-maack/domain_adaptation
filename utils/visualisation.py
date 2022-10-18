import torch
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_tsne(feature_embedding, segmentation_mask, pca_n_comp, verbose=0, perplexity=40, n_iter=300):
    """
    Creates a tsne for a given feature embedding and corresponding, downscaled binary segmentation mask.
    This function returns a pandas dataframe with all features (columns) per pixel (row) from a downscaled minibatch feature embedding.
    The three important columns to plot the tsne are column 'tsne-1', 'tsne-2', 'label'.  
    
    Input:
    Feature embedding (torch.tensor): size [Batch_size, D, H*, W*], D being the spatial dimension of the feature embedding, H*,W* height, width of the feature embedding
    Segmentation mask (torch.tensor): size [Batch_size, 1, H*, W*]
    pca_n_comp (int): number of principal components used for principal components analysis
    """
    feature_embedding = feature_embedding.cpu().numpy()
    segmentation_mask = segmentation_mask.cpu().numpy()

    assert (feature_embedding.shape[2], feature_embedding.shape[3]) == (segmentation_mask.shape[2], segmentation_mask.shape[3]), "feature_embedding and segmentation_mask have to have the same shape"

    features_for_single_pixel = np.empty(0)
    list_of_dataframes = list()

    feat_cols = [ 'feature'+str(i) for i in range(feature_embedding.shape[1])]
    feat_cols.append("label")

    # looping through the minibatch/feature maps per image
    for i in range(feature_embedding.shape[0]):

        # create an empty dataframe with the columns feature_0 to feature_'len(feature_embedding.shape[1])' and one last label column
        df = pd.DataFrame(columns=feat_cols)
        
        # looping through the spatial feature dimension
        for x in range(feature_embedding.shape[2]):
            for y in range(feature_embedding.shape[3]):
                for j in range(feature_embedding.shape[1]):
                    # appends the np.array containing the D features for each individual pixel
                    features_for_single_pixel = np.append(features_for_single_pixel, feature_embedding[i][j][x][y])
                if segmentation_mask[i][0][x][y] == 0:
                    # append dataframe with a row and columns containing features in features_for_single_pixel (individually) and one label "background"
                    features_for_single_pixel = np.append(features_for_single_pixel, 0)
                    df = pd.concat([df, pd.DataFrame(features_for_single_pixel.reshape(1,-1), columns=list(df))], ignore_index=True)
                elif segmentation_mask[i][0][x][y] == 1:
                    # append dataframe with a row and columns containing features in features_for_single_pixel (individually) and one label "polyp"
                    features_for_single_pixel = np.append(features_for_single_pixel, 1)
                    df = pd.concat([df, pd.DataFrame(features_for_single_pixel.reshape(1,-1), columns=list(df))], ignore_index=True)
                else:
                    raise ValueError("The value of segmentation_mask is neither 0 nor 1 but has to be one of either")
                
                features_for_single_pixel = np.empty(0)
        list_of_dataframes.append(df)

    # Concat all dataframes (length is equal to batch size) containing a dataframe of features for each pixel and label to one big dataframe
    df_concat = pd.concat([df for df in list_of_dataframes], ignore_index=True)

    feat_cols = [ 'feature'+str(i) for i in range(feature_embedding.shape[1])]
    np_value_subset = df_concat[feat_cols].values

    pca = PCA(n_components=pca_n_comp)
    pca_result = pca.fit_transform(np_value_subset)

    print(f'Cumulative explained variation for {64} principal components: {np.sum(pca.explained_variance_ratio_)}')

    tsne = TSNE(n_components=2, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_pca_results = tsne.fit_transform(pca_result)

    df_concat['tsne-one'] = tsne_pca_results[:,0]
    df_concat['tsne-two'] = tsne_pca_results[:,1]

    return df_concat