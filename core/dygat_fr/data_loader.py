"""
DyGAT-FR Data Utilities

Utilities for converting tabular NIDS data to graph format
and creating temporal increments for incremental learning.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple, Optional, Dict, Any
import warnings


class TabularToGraphConverter:
    """
    Convert tabular NIDS data to graph format.
    
    Creates a k-NN graph where each sample is a node and edges
    connect similar samples based on feature distance.
    
    Args:
        k_neighbors: Number of nearest neighbors for graph construction
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        include_self_loops: Whether to add self-loops
    """
    
    def __init__(
        self,
        k_neighbors: int = 10,
        metric: str = 'euclidean',
        include_self_loops: bool = True
    ):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.include_self_loops = include_self_loops
        self.scaler = StandardScaler()
        self.nn_model = None
    
    def fit(self, X: np.ndarray) -> 'TabularToGraphConverter':
        """
        Fit the converter on training data.
        
        Args:
            X: Feature matrix (N, d)
        
        Returns:
            self
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit nearest neighbors
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, len(X)),
            metric=self.metric,
            algorithm='auto'
        )
        self.nn_model.fit(X_scaled)
        
        return self
    
    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Data:
        """
        Transform tabular data to PyG Data object.
        
        Args:
            X: Feature matrix (N, d)
            y: Labels (N,)
        
        Returns:
            PyG Data object with nodes and edges
        """
        if self.nn_model is None:
            self.fit(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Find k-nearest neighbors
        distances, indices = self.nn_model.kneighbors(X_scaled)
        
        # Build edge index
        n_samples = len(X)
        src_nodes = []
        dst_nodes = []
        
        for i in range(n_samples):
            for j in indices[i]:
                if i != j:  # Skip self-loops here, add separately if needed
                    src_nodes.append(i)
                    dst_nodes.append(j)
        
        # Add self-loops if requested
        if self.include_self_loops:
            for i in range(n_samples):
                src_nodes.append(i)
                dst_nodes.append(i)
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(X_scaled),
            edge_index=edge_index,
            y=torch.LongTensor(y)
        )
        
        data.num_nodes = n_samples
        
        return data
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Data:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X, y)


class TemporalGraphSplitter:
    """
    Create temporal increments from static graph data.
    
    Simulates streaming/incremental scenario by splitting data
    into time-based increments with evolving class distributions.
    
    Args:
        n_increments: Number of temporal splits
        minority_drift: Whether to simulate minority class drift
        drift_intensity: Intensity of class distribution drift
    """
    
    def __init__(
        self,
        n_increments: int = 5,
        minority_drift: bool = True,
        drift_intensity: float = 0.3
    ):
        self.n_increments = n_increments
        self.minority_drift = minority_drift
        self.drift_intensity = drift_intensity
    
    def split(
        self,
        data: Data,
        random_state: int = 42
    ) -> List[Data]:
        """
        Split graph data into temporal increments.
        
        Args:
            data: PyG Data object
            random_state: Random seed
        
        Returns:
            List of Data objects (one per increment)
        """
        np.random.seed(random_state)
        
        n_nodes = data.num_nodes
        indices = np.arange(n_nodes)
        
        # Shuffle indices (simulating temporal ordering)
        np.random.shuffle(indices)
        
        # Split into increments
        splits = np.array_split(indices, self.n_increments)
        
        increments = []
        
        for i, split_indices in enumerate(splits):
            # If minority drift, adjust sampling
            if self.minority_drift:
                # Later increments may have more/different minority samples
                minority_mask = data.y[split_indices] == 1
                minority_idx = split_indices[minority_mask.numpy()]
                majority_idx = split_indices[~minority_mask.numpy()]
                
                # Simulate increasing minority presence over time
                # (optional: can also decrease to simulate forgetting challenge)
                drift_factor = 1.0 + (i / self.n_increments) * self.drift_intensity
                n_minority_keep = min(
                    int(len(minority_idx) * drift_factor),
                    len(minority_idx)
                )
                
                # Combine indices
                final_indices = np.concatenate([
                    minority_idx[:n_minority_keep],
                    majority_idx
                ])
            else:
                final_indices = split_indices
            
            # Create subgraph
            increment_data = self._create_subgraph(data, final_indices)
            increment_data.increment_id = i
            increments.append(increment_data)
        
        return increments
    
    def _create_subgraph(
        self,
        data: Data,
        node_indices: np.ndarray
    ) -> Data:
        """
        Extract subgraph for given node indices.
        
        Args:
            data: Original Data object
            node_indices: Indices of nodes to include
        
        Returns:
            Subgraph Data object
        """
        node_indices = torch.LongTensor(node_indices)
        
        # Create node mapping
        node_map = {int(old): new for new, old in enumerate(node_indices)}
        
        # Filter edges
        edge_index = data.edge_index
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        
        node_set = set(node_indices.tolist())
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_set and dst in node_set:
                mask[i] = True
        
        # Remap edges
        filtered_edges = edge_index[:, mask]
        new_edges = torch.zeros_like(filtered_edges)
        
        for i in range(filtered_edges.size(1)):
            new_edges[0, i] = node_map[filtered_edges[0, i].item()]
            new_edges[1, i] = node_map[filtered_edges[1, i].item()]
        
        # Create new Data object
        subgraph = Data(
            x=data.x[node_indices],
            edge_index=new_edges,
            y=data.y[node_indices]
        )
        subgraph.num_nodes = len(node_indices)
        
        return subgraph


class NIDSDataLoader:
    """
    Load and preprocess NIDS datasets (UNSW-NB15, CIC-IDS, etc.)
    
    Handles:
    - Loading from CSV
    - Feature preprocessing
    - Label encoding
    - Train/test splitting
    - Graph conversion
    
    Args:
        dataset_name: Name of dataset ('unsw-nb15', 'cic-ids2017', etc.)
        data_dir: Directory containing data files
    """
    
    def __init__(
        self,
        dataset_name: str = 'unsw-nb15',
        data_dir: str = '.'
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.feature_columns: Optional[List[str]] = None
        self.graph_converter = TabularToGraphConverter(k_neighbors=10)
    
    def load_unsw_nb15(
        self,
        train_file: str = 'UNSW_NB15_training-set.csv',
        test_file: str = 'UNSW_NB15_testing-set.csv'
    ) -> Tuple[Data, Data]:
        """
        Load UNSW-NB15 dataset and convert to graph format.
        
        Args:
            train_file: Training CSV filename
            test_file: Test CSV filename
        
        Returns:
            Tuple of (train_data, test_data) as PyG Data objects
        """
        import os
        
        # Load CSVs
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Preprocess
        X_train, y_train = self._preprocess_unsw(train_df)
        X_test, y_test = self._preprocess_unsw(test_df)
        
        # Convert to binary (normal vs attack)
        y_train_binary = (y_train > 0).astype(int)  # 0 = normal, 1 = attack
        y_test_binary = (y_test > 0).astype(int)
        
        # Fit converter on training data
        self.graph_converter.fit(X_train)
        
        # Convert to graph
        train_data = self.graph_converter.transform(X_train, y_train_binary)
        test_data = self.graph_converter.transform(X_test, y_test_binary)
        
        # Store attack types for detailed analysis
        train_data.attack_types = y_train
        test_data.attack_types = y_test
        
        return train_data, test_data
    
    def _preprocess_unsw(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess UNSW-NB15 dataframe.
        
        Args:
            df: Raw dataframe
        
        Returns:
            Tuple of (features, labels)
        """
        # Drop ID columns if present
        drop_cols = ['id', 'label']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Separate features and target
        if 'attack_cat' in df.columns:
            y = df['attack_cat'].values
            X = df.drop(columns=['attack_cat'])
        else:
            # Assume last column is target
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1]
        
        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == object:
            # Map 'Normal' to 0, attacks to 1+
            unique_labels = sorted(set(y))
            if 'Normal' in unique_labels:
                unique_labels.remove('Normal')
                label_map = {'Normal': 0}
                for i, label in enumerate(unique_labels, 1):
                    label_map[label] = i
                y = np.array([label_map[l] for l in y])
            else:
                y = self.label_encoder.fit_transform(y)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        return X.values.astype(np.float32), y.astype(np.int64)
    
    def create_increments(
        self,
        data: Data,
        n_increments: int = 5,
        minority_drift: bool = True
    ) -> List[Data]:
        """
        Create temporal increments from data.
        
        Args:
            data: PyG Data object
            n_increments: Number of increments
            minority_drift: Simulate class drift
        
        Returns:
            List of increment Data objects
        """
        splitter = TemporalGraphSplitter(
            n_increments=n_increments,
            minority_drift=minority_drift
        )
        return splitter.split(data)


def create_synthetic_graph(
    n_nodes: int = 1000,
    n_features: int = 33,
    minority_ratio: float = 0.1,
    k_neighbors: int = 10,
    random_state: int = 42
) -> Data:
    """
    Create a synthetic graph for testing.
    
    Args:
        n_nodes: Number of nodes
        n_features: Feature dimension
        minority_ratio: Ratio of minority class
        k_neighbors: k for kNN graph
        random_state: Random seed
    
    Returns:
        Synthetic PyG Data object
    """
    np.random.seed(random_state)
    
    n_minority = int(n_nodes * minority_ratio)
    n_majority = n_nodes - n_minority
    
    # Generate features with class separation
    X_majority = np.random.randn(n_majority, n_features)
    X_minority = np.random.randn(n_minority, n_features) + 1.5  # Shifted
    
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)
    
    # Shuffle
    perm = np.random.permutation(n_nodes)
    X = X[perm]
    y = y[perm]
    
    # Convert to graph
    converter = TabularToGraphConverter(k_neighbors=k_neighbors)
    data = converter.fit_transform(X, y)
    
    return data


def compute_graph_statistics(data: Data) -> Dict[str, Any]:
    """
    Compute statistics for a graph dataset.
    
    Args:
        data: PyG Data object
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1),
        'num_features': data.x.size(1),
        'avg_degree': data.edge_index.size(1) / data.num_nodes,
    }
    
    # Class distribution
    unique, counts = torch.unique(data.y, return_counts=True)
    stats['class_distribution'] = {
        int(u): int(c) for u, c in zip(unique, counts)
    }
    
    # Imbalance ratio
    if len(counts) == 2:
        stats['imbalance_ratio'] = float(counts.min() / counts.max())
    
    return stats
