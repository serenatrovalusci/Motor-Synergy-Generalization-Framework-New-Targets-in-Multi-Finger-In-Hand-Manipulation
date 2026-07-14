import numpy as np
from sklearn.decomposition import PCA, NMF
import torch
import torch.nn as nn


class SpatialSynergy:
    """Spatial synergies.
    """

    def __init__(self, n_synergies, method="nmf"):
        """
        Args:
            n_synergies: Number of synergies
            method: Synergy extraction method PCA or NMF
        """
        self.n_synergies = n_synergies
        self.method = method

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None

    def extract(self, data, max_iter=1000):
        """Extract spatial synergies from given data.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergies have the shape (#synergies, #DoF).
        """
        # Get shape information
        self.dof = data.shape[-1]

         # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)  # transform data from dof to 2*dof dimensions
            self.dof = data.shape[2]  # Update the number of DoF

        # Reshape given data
        data = data.reshape((-1, self.dof))

        if self.method == "nmf" or self.method == "negative-nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)    #used to learn the transformation from data to synergies
            self.synergies = self.model.components_
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies) #reduce the dimensionality of the data from dof to n_synergies
            self.model.fit(data)
            self.synergies = self.model.components_

        return self.synergies

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, length, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Keep the shape temporarily
        data_shape = data.shape

          # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)

        # Reshape given data from a 3D array to a 2D array, the -1 infers the size of that dimension, knowing the size of the other dimensions
        data = data.reshape((-1, self.dof))  # shape: (#trajectories * length, #DoF)

        # Encode the data
        activities = self.model.transform(data)

        # Reshape activities
        activities = activities.reshape((data_shape[0], data_shape[1], self.n_synergies))  # shape: (#trajectories, length, #synergies)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities have the shape (#trajectories, length, #synergies).
        Data is assumed to have the shape (#trajectories, length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Keep the shape temporarily
        act_shape = activities.shape

        # Reshape given activities
        activities = activities.reshape((-1, self.n_synergies))  # shape: (#trajectories * length, #synergies)

        # Decode the synergy activities
        data = self.model.inverse_transform(activities)

        # Reshape reconstruction data
        data = data.reshape((act_shape[0], act_shape[1], self.dof))  # shape: (#trajectories, length, #DoF)

        # Convert non-negative signals backwards
        if self.method == "negative-nmf":
            data = inverse_transform_nonnegative(data)

        return data



class _AutoencoderNet(nn.Module):
    """Shallow MLP autoencoder: dof -> hidden -> K -> hidden -> dof."""

    def __init__(self, dof, n_synergies, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dof, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_synergies),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_synergies, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dof),
            nn.Tanh(),  # joint actions live in [-1, 1]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class AutoencoderSynergy:
    """Nonlinear synergy extraction via a shallow autoencoder.

    Mirrors the ``SpatialSynergy`` interface (``n_synergies``, ``dof``,
    ``extract``/``encode``/``decode``) so it is a drop-in replacement in
    ``synergy_extract_analyze.py`` and ``sac_her_pipeline.py``. Unlike
    PCA/NMF the decoder is a nonlinear MLP, so there is no fixed weight
    matrix -- ``self.synergies`` stays ``None``.
    """

    def __init__(
        self,
        n_synergies,
        method="autoencoder",
        hidden_dim=64,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        weight_decay=0.0,
        val_split=0.1,
        patience=20,
        device=None,
        seed=0,
        verbose=True,
    ):
        """
        Args:
            n_synergies: Bottleneck dimension K.
            hidden_dim: Width of the single hidden layer in encoder/decoder.
            epochs: Max training epochs (may stop early, see `patience`).
            batch_size: Minibatch size for SGD.
            lr: Adam learning rate.
            val_split: Fraction of (flattened) samples held out for early stopping.
            patience: Stop after this many epochs without val-loss improvement.
            device: "cuda"/"cpu"/"mps". Defaults to cuda if available, else cpu.
            verbose: Print periodic training progress.
        """
        self.n_synergies = n_synergies
        self.method = method
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.verbose = verbose

        # Initialize variables
        self.model = None
        self.synergies = None  # no fixed linear basis for a nonlinear decoder
        self.dof = None

    def extract(self, data):
        """Train the autoencoder on given data.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Returns None (there is no fixed weight matrix to hand back, unlike
        the linear methods in `SpatialSynergy`).
        """
        self.dof = data.shape[-1]
        flat = np.asarray(data, dtype=np.float32).reshape(-1, self.dof)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n = flat.shape[0]
        n_val = max(1, int(n * self.val_split))
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        x_train = torch.from_numpy(flat[train_idx]).to(self.device)
        x_val = torch.from_numpy(flat[val_idx]).to(self.device)

        self.model = _AutoencoderNet(self.dof, self.n_synergies, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val, best_state, bad_epochs = float("inf"), None, 0
        n_train = x_train.shape[0]

        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx in torch.randperm(n_train, device=self.device).split(self.batch_size):
                batch = x_train[batch_idx]
                opt.zero_grad()
                loss = torch.mean((self.model(batch) - batch) ** 2)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = torch.mean((self.model(x_val) - x_val) ** 2).item()

            if val_loss < best_val - 1e-6:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"  [autoencoder K={self.n_synergies}] epoch {epoch+1}/{self.epochs}  val_mse={val_loss:.6f}")

            if bad_epochs >= self.patience:
                if self.verbose:
                    print(f"  [autoencoder K={self.n_synergies}] early stop at epoch {epoch+1} "
                          f"(best val_mse={best_val:.6f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        return None

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, length, #synergies).
        """
        if self.model is None:
            return None

        data_shape = data.shape
        flat = np.asarray(data, dtype=np.float32).reshape(-1, self.dof)
        with torch.no_grad():
            z = self.model.encode(torch.from_numpy(flat).to(self.device)).cpu().numpy()

        return z.reshape((data_shape[0], data_shape[1], self.n_synergies))

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities have the shape (#trajectories, length, #synergies).
        Data is assumed to have the shape (#trajectories, length, #DoF).
        """
        if self.model is None:
            return None

        activities = np.asarray(activities)
        act_shape = activities.shape
        flat = activities.reshape(-1, self.n_synergies).astype(np.float32)
        with torch.no_grad():
            x = self.model.decode(torch.from_numpy(flat).to(self.device)).cpu().numpy()

        return x.reshape((act_shape[0], act_shape[1], self.dof))


class SpatioTemporalSynergy:
    """Spatio-temporal synergies.
    """

    def __init__(self, n_synergies, method="nmf"):
        """
        Args:
            n_synergies: Number of synergies
            method: Synergy extraction method pca, nmf, or negative-nmf
        """
        self.n_synergies = n_synergies
        self.method = method

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None
        self.length = None

    def extract(self, data, max_iter=100000):
        """Extract spatio-temporal synergies from given data.

        Data is assumed to have the shape (#data, length, #DoF).
        Synergies have the shape (#synergies, length, #DoF).
        """
        # Get shape information
        self.length = data.shape[1]
        self.dof = data.shape[2]

        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)  # transform data from dof to 2*dof dimensions
            self.dof = data.shape[2]  # Update the number of DoF

        # Reshape given data
        data = data.reshape((data.shape[0], -1))  # shape: (#data, length * #DoF)

        if self.method == "nmf" or self.method == "negative-nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)
            self.synergies = self.model.components_
            self.synergies = self.synergies.reshape((self.n_synergies, self.length, self.dof))  # Reshape synergies
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies)
            self.model.fit(data)
            self.synergies = self.model.components_
            self.synergies = self.synergies.reshape((self.n_synergies, self.length, self.dof))  # Reshape synergies

        return self.synergies

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)

        # Reshape the data from (#trajectories, length, #DoF) to (#trajectories, length * #DoF)
        data = data.reshape((-1, self.length*self.dof))

        # Encode the data
        activities = self.model.transform(data)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities are assumed to have the shape (#trajectories, #activities).
        Data have the shape (#trajectories, length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Decode the synergy activities
        data = self.model.inverse_transform(activities)
        data = data.reshape((-1, self.length, self.dof))  # Reshape the shape from (#trajectories, length * #DoF) to (#trajectories, length, #DoF)

        # Convert non-negative signals backwards
        if self.method == "negative-nmf":
            data = inverse_transform_nonnegative(data)

        return data
    

    

def transform_nonnegative(data):
    """Convert a data that has negative values to non-negative signals with doubled dimensions.

    Data is assumed to have the shape (#trajectories, length, #DoF).
    Converted non-negative data have the shape (#trajectories, length, 2 * #DoF).
    """
    n_dof = data.shape[2]  # Dimensionality of the original data

    # Convert the data to non-negative signals
    data_nn = np.empty((data.shape[0], data.shape[1], n_dof*2))
    data_nn[:, :, :n_dof] = +np.maximum(data, 0.0)
    data_nn[:, :, n_dof:] = -np.minimum(data, 0.0)

    return data_nn

def inverse_transform_nonnegative(data):
    """Inverse conversion of `transform_nonnegative()`; Convert non-negative signals to a data that has negative values.

    Non-negative data is assumed to have the shape (#trajectories, length, 2 * #DoF).
    Reconstructed data have the shape (#trajectories, length, #DoF).
    """
    n_dof = int(data.shape[2] / 2)  # Dimensionality of the original data ([2] refers to the last dimension, which is the third == [-1])

    # Restore the original data
    data_rc = np.empty((data.shape[0], data.shape[1], n_dof))
    data_rc = data[:, :, :n_dof] - data[:, :, n_dof:] #the property written in the paper, a = a+(from 0-dof-1) - a- (from dof to 2*dof-1) 

    return data_rc

def R2(x, y):
    e = x - y
    v = x - np.mean(x)
    fvu = np.sum(e**2) / np.sum(v**2)
    return 1 - fvu
