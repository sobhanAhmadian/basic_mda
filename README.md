In this project, we implemented 18 different models for the task of the Microbe-Disease Association.
Each model is evaluated using 5-fold cross-validation.

We have two kinds of input for the Microbe-Disease Problem:

+ Interaction Profile

[<img src="/images/ip.png" width="300"/>](/images/ip.png)

Each element of this profile determines whether a microbe is associated with a disease. Our interaction profile is built based on the [HMDAD](https://www.cuilab.cn/hmdad) dataset.

+ A Knowledge Graph

[<img src="/images/kg.png" width="500"/>](/images/kg.png)

This data is based on the [KGNMDA](https://ieeexplore.ieee.org/document/9800198) work.

The output of the problem is whether a disease and a microbe are associated. The following flowchart illustrates this:

[<img src="/images/flowchart.png" width="750"/>](/images/flowchart.png)

# Models Using Interaction Profile for Feature Extraction

In this approach, we just use the interaction profile as the input for our learning algorithm.

## Matrix Completion Approach

This is a flowchart of the matrix completion algorithm:

[<img src="/images/matcomp.png" width="650"/>](/images/matcomp.png)

Based on the matrix factorization algorithm we implemented these two methods:

+ [nmf](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/matrix_completion/nmf.ipynb)
+ [pca](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/matrix_completion/pca.ipynb)

## Feature Extraction + Classifier

In this approach, we first extract features from the interaction profile and then use an MLP classifier to predict association.

+ [dummy_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/dummy_features.ipynb)

For the Dummy feature extraction, we just extract a row and a column from the interaction profile:

[<img src="/images/dummy.png" width="600"/>](/images/dummy.png)

Another approach is to use similarity measures to extract features for both a microbe and a disease. For example, if we want to extract some features of a disease we can measure its similarity to other diseases based on the interaction profile:

[<img src="/images/similarity.png" width="400"/>](/images/similarity.png)

Based on the similarity measure, we implemented two methods:

+ [gaussian_similarity_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/gaussian_similarity_features.ipynb)
+ [jaccard_similarity_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/mlp.ipynb)

At last, we used matrix factorization to extract features:

[<img src="/images/mf.png" width="700"/>](/images/mf.png)

+ [nmf_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/nmf_features.ipynb)
+ [pca_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/pca_features.ipynb)

### Choice of Classifier

In this part, we used the Jaccard similarity features for the feature extraction part but we used different classifiers.

+ [ada_boost](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/ada_boost.ipynb)
+ [decision_tree](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/decision_tree.ipynb)
+ [logistic](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/logistic.ipynb)
+ [mlp](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/mlp.ipynb)
+ [rf](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/rf.ipynb)

# Models Using KG for Feature Extraction

In these approaches, we just used the Knowledge Graph as input for the learning algorithm.

## Deep Graph Neural Network-Based Models
In this approach, we used an Auto-Encoder architecture to extract features:

[<img src="/images/auto.png" width="400"/>](/images/auto.png)

+ [homo_gcn](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/deep_gnn/homo_gcn.ipynb)

## Shallow Graph Neural Network-Based Models

As an alternative approach, we used Node2Vec which is a shallow method for graph feature extraction.

+ [node2vec](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/shallow_gnn/node2vec.ipynb)

## Knowledge-Graph Embedding Models

In this experiment, we used different KG Embedding methods.

+ [complex](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/complex.ipynb)
+ [distmult](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/dismult.ipynb)
+ [rotate](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/rotate.ipynb)
+ [transe](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/transe.ipynb)

## Combination of Models

Here we combined the Node2Vec approach with the Graph Auto-Encoder method. We first extracted features from Node2Vec and used these features as initial features of the Graph Auto-Encoder method.

+ [homo_node2vec_gcn](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/deep_gnn/homo_node2vec_gcn.ipynb)
  
