# DrugXAS
Interpretable and Adaptive Graph Contrastive Learning with Information Sharing for Biomedical Link Prediction

# Requirements:
- python 3.11.9
- cudatoolkit 12.6
- pytorch 2.1.0
- dgl 2.0.0
- Rdkit 2024.3.5 
- numpy 1.26.4
- scikit-learn 1.5.1

# Data:
The data files needed to run the model, including LuoDTI, ZhengDTI, LiangDDA, ZhangDDA, PwuwelsDSE, HuangMDA and DengCDA.
- Adj.csv: The adjacency matrix of the biomedical heterogeneous graph
- DTI/DDA/DSE/MDA/CDA.csv: The known links
- Drug_sim.csv: The similarity measurements of drugs
- Druginformation.csv: The drug SMILES for constructing molecular graphs
- Protein_sim/Disease_sim/SideEffect_sim/miRNA_sim/circRNA_sim.csv: The similarity measurements of different types of entities

# Code:
- data_preprocess.py: Methods of data processing
- metric.py: Metrics calculation
- pos_contrast.py: Get positive and negative samples for contrastive learning
- Contrast.py: Code of graph contrastive learning
- model.py: Model of DRMAHGC
- train_DDA.py: Train the model

# Usage:
Execute ```python train_DDA.py``` 
