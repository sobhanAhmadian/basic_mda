In this project, we implemented 18 different models for the task of the Microbe-Disease Association.
Each model is evaluated using 5-fold cross-validation.

We have two kinds of input for the Microbe-Disease Problem:

+ Interaction Profile

![Interaction Profile](/images/ip.png)

Each element of this profile determines whether a microbe is associated with a disease. Our interaction profile is built based on the [HMDAD](https://www.cuilab.cn/hmdad) dataset.

+ A Knowledge Graph

![Knowledge Graph](/images/kg.png)

This data is based on the [KGNMDA](https://ieeexplore.ieee.org/document/9800198) work.

The output of the problem is whether a disease and a microbe are associated. The following flowchart illustrates this:

![Flowchart of Problem](/images/flowchart.png)

# Models Using Interaction Profile for Feature Extraction

In this approach, we just use the interaction profile as the input for our learning algorithm.

## Matrix Completion Approach

![Matrix Completion Algorithms](/images/matcomp.png)

+ [nmf](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/matrix_completion/nmf.ipynb)
+ [pca](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/matrix_completion/pca.ipynb)

## Feature Extraction + Classifier

+ [dummy_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/dummy_features.ipynb)
+ [gaussian_similarity_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/gaussian_similarity_features.ipynb)
+ [jaccard_similarity_features](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/microbe_disease_matrix_feature_based/jaccard_similarity_features/mlp.ipynb)
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

## Deep Graph Neural Network-Based Models
+ [homo_gcn](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/deep_gnn/homo_gcn.ipynb)

## Shallow Graph Neural Network-Based Models
+ [node2vec](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/shallow_gnn/node2vec.ipynb)

## Knowledge-Graph Embedding Models

+ [complex](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/complex.ipynb)
+ [distmult](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/dismult.ipynb)
+ [rotate](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/rotate.ipynb)
+ [transe](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/knowledge_graph_embedding/transe.ipynb)

## Combination of Models
+ [homo_node2vec_gcn](https://github.com/sobhanAhmadian/basic_mda/blob/main/notebooks/report/models/deep_gnn/homo_node2vec_gcn.ipynb)
  
