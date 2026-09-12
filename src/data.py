import random
import numpy as np
import torch
import h5py

class SymbolICLGenerator:
    """
    Generates sequences:
      [x1, y1, x2, y2, ..., xq]
    where the query xq repeats one context class and the target is its
    *within-sequence randomly assigned* label.

    We represent every position as a d_input vector. Symbol positions are
    one-hot over classes; label positions are one-hot over labels appended
    after the symbol coordinates.
    """
    def __init__(self, num_classes=50, num_labels=5, context_len=2, device="cpu"):
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.context_len = context_len
        self.input_dim = num_classes + num_labels
        self.device = device

    def sample(self, batch_size):
        B, K = batch_size, self.context_len
        x = torch.zeros(B, 2*K + 1, self.input_dim, device=self.device)
        targets = torch.empty(B, dtype=torch.long, device=self.device)

        for b in range(B):
            classes = random.sample(range(self.num_classes), K)
            # random labels, consistent within this sequence
            labels = [random.randrange(self.num_labels) for _ in range(K)]
            q_idx = random.randrange(K)
            query_class = classes[q_idx]
            target = labels[q_idx]

            for j, (cls, lab) in enumerate(zip(classes, labels)):
                x[b, 2*j, cls] = 1.0
                x[b, 2*j+1, self.num_classes + lab] = 1.0

            x[b, -1, query_class] = 1.0
            targets[b] = target

        return x, targets


class OmniglotICLGenerator:
    """
    Singh-style generator for the supplied h5 feature file.

    Expected h5 tensor shape:
        [num_classes, num_exemplars, feature_dim]

    Sequence layout:
        exemplar_1, label_1, exemplar_2, label_2, ..., query_exemplar

    Labels are randomly assigned from 0..num_labels-1 independently per
    sequence, but remain consistent for repeated class identities.
    """
    def __init__(self, path, class_start, class_count, exemplar_indices,
                 num_labels=5, context_len=2, device="cpu", dataset_key=None):
        self.device = device
        self.num_labels = num_labels
        self.context_len = context_len

        with h5py.File(path, "r") as f:
            if dataset_key is None:
                keys = []
                f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None)
                if not keys:
                    raise ValueError("No dataset found in h5 file.")
                dataset_key = keys[0]
            arr = np.asarray(f[dataset_key])

        if arr.ndim != 3:
            raise ValueError(f"Expected [classes, exemplars, features], got {arr.shape}")

        self.features = torch.tensor(
            arr[class_start:class_start+class_count], dtype=torch.float32
        )
        self.exemplar_indices = list(exemplar_indices)
        self.feature_dim = self.features.shape[-1]
        self.input_dim = self.feature_dim + num_labels
        self.class_count = self.features.shape[0]

    def _feature(self, cls, ex):
        return self.features[cls, ex]

    def sample(self, batch_size):
        B, K = batch_size, self.context_len
        x = torch.zeros(B, 2*K + 1, self.input_dim, device=self.device)
        targets = torch.empty(B, dtype=torch.long, device=self.device)

        for b in range(B):
            classes = random.sample(range(self.class_count), K)
            labels = [random.randrange(self.num_labels) for _ in range(K)]
            q_idx = random.randrange(K)

            for j, (cls, lab) in enumerate(zip(classes, labels)):
                ex = random.choice(self.exemplar_indices)
                x[b, 2*j, :self.feature_dim] = self._feature(cls, ex).to(self.device)
                x[b, 2*j+1, self.feature_dim + lab] = 1.0

            q_cls = classes[q_idx]
            q_ex = random.choice(self.exemplar_indices)
            x[b, -1, :self.feature_dim] = self._feature(q_cls, q_ex).to(self.device)
            targets[b] = labels[q_idx]

        return x, targets
