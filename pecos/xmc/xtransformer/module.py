#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import glob
import os

import numpy as np
import scipy.sparse as smat
import torch
from torch.utils.data import Dataset, TensorDataset


class MLProblemWithText(object):
    """Object that defines a ML-Problem with text input.
    Containing the input text and X, Y, C, M, M_pred matrices.

        X_text (list of str or dict of tensors): instance text, len(text) = nr_inst
                or dictionary of tokenized text (dict of torch.tensor)
        Y (csr_matrix): training labels, shape = (nr_inst, nr_labels)
        X_feat (csr_matrix or ndarray): instance numerical features, shape = (nr_inst, nr_features)
        C (csc_matrix, optional): clustering matrix, shape = (nr_labels, nr_codes)
        M (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes)
                model will be trained only on its non-zero indices
                its values will not be used.
    """

    def __init__(self, X_text, Y, X_feat=None, C=None, M=None):
        self.X_text = X_text  # List of text / dict of tensors / path to dir storing the tensors

        self.Y_label = Y
        self.X_feat = X_feat
        self.C = C
        self.M = M

        self.type_check()

    @property
    def is_tokenized(self):
        return isinstance(self.X_text, dict) or \
               isinstance(self.X_text, str) and os.path.exists(self.X_text)

    def type_check(self):
        if self.X_feat is not None and not isinstance(self.X_feat, (smat.csr_matrix, np.ndarray)):
            raise TypeError(f"Expect X to be csr_matrix or ndarray, got {type(self.X)}")
        if not isinstance(self.Y_label, smat.csr_matrix):
            raise TypeError(f"Expect Y to be csr_matrix, got {type(self.Y_label)}")
        if self.C is not None and not isinstance(self.C, smat.csc_matrix):
            raise TypeError(f"Expect C to be csc_matrix, got {type(self.C)}")
        if self.M is not None and not isinstance(self.M, smat.csr_matrix):
            raise TypeError(f"Expect M to be csr_matrix, got {type(self.M)}")

    @property
    def nr_labels(self):
        return self.Y_label.shape[1]

    @property
    def nr_features(self):
        return None if self.X_feat is None else self.X_feat.shape[1]

    @property
    def nr_codes(self):
        return None if self.C is None else self.C.shape[1]


class MLMultiTaskProblemWithText(object):
    """Object that defines a ML-MultiTask-Problem with text input.
    Containing the input text and X, Y_label, Y_class, C, M, M_pred matrices.

        X_text (list of str or dict of tensors): instance text, len(text) = nr_inst
                or dictionary of tokenized text (dict of torch.tensor)
        Y_label (csr_matrix): training labels, shape = (nr_inst, nr_labels)
        Y_class (ndarray): training classes, shape = (nr_inst)
        X_feat (csr_matrix or ndarray): instance numerical features, shape = (nr_inst, nr_features)
        C (csc_matrix, optional): clustering matrix, shape = (nr_labels, nr_codes)
        M (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes)
                model will be trained only on its non-zero indices
                its values will not be used.
    """

    def __init__(self, X_text, Y_label, Y_class, X_feat=None, C=None, M=None):
        self.X_text = X_text

        self.Y_label = Y_label
        self.Y_class = Y_class
        self.X_feat = X_feat
        self.C = C
        self.M = M

        self.type_check()

    @property
    def is_tokenized(self):
        return isinstance(self.X_text, dict)

    def type_check(self):
        if self.X_feat is not None and not isinstance(self.X_feat, (smat.csr_matrix, np.ndarray)):
            raise TypeError(f"Expect X to be csr_matrix or ndarray, got {type(self.X)}")
        if not isinstance(self.Y_label, smat.csr_matrix):
            raise TypeError(f"Expect Y_label to be csr_matrix, got {type(self.Y_label)}")
        if self.C is not None and not isinstance(self.C, smat.csc_matrix):
            raise TypeError(f"Expect C to be csc_matrix, got {type(self.C)}")
        if self.M is not None and not isinstance(self.M, smat.csr_matrix):
            raise TypeError(f"Expect M to be csr_matrix, got {type(self.M)}")

    @property
    def nr_labels(self):
        return self.Y_label.shape[1]

    @property
    def nr_features(self):
        return None if self.X_feat is None else self.X_feat.shape[1]

    @property
    def nr_codes(self):
        return None if self.C is None else self.C.shape[1]



class XMCDataset(Dataset):
    """Dataset to hold feature and label tensors for XMC training and prediction.

    Args:
        *features (tensors): feature tensors, required to have same first
            dimension: nr_inst
        label_values (tensor or None): label values with shape = (nr_inst, num_active_labels)
        label_indices (tensor or None): label indices with shape = (nr_inst, num_active_labels)

    Return values depend on the label_values and label_indices:
        if label_values is None and label_indices is not None:
            data[i] = (feature[0][i], feature[1][i], ..., label_values[i], label_indices[i])
        elif label_values is not None:
            data[i] = (feature[0][i], feature[1][i], ..., label_values[i])
        elif label_indices is not None:
            data[i] = (feature[0][i], feature[1][i], ..., label_indices[i])
        else:
            data[i] = (feature[0][i], feature[1][i], ...)
    """

    def __init__(self, *features, label_values=None, label_indices=None, memmap=False):
        if not memmap:
            self.nr_inst = features[0].size(0)
            self.data = TensorDataset(*features)
        else:
            self.data_path = features[0]
            self.nr_inst = len(glob.glob1(self.data_path, "*.pt"))

        if label_values is not None and label_values.size(0) != self.nr_inst:
            raise ValueError(f"First dimension mismatch between features {self.nr_inst} and label_values {label_values.size(0)}")
        if label_indices is not None and label_indices.size(0) != self.nr_inst:
            raise ValueError(f"First dimension mismatch between features {self.nr_inst} and label_indices {label_indices.size(0)}")

        self.label_values = label_values
        self.label_indices = label_indices
        self.memmap = memmap

    def get_data_item_from_dir(self, i):
        assert self.memmap
        data_item = torch.load(f"{self.data_path}/{i}.pt")
        return tuple([data_item["input_ids"][0], data_item["attention_mask"][0], data_item["token_type_ids"][0], torch.tensor(i)])

    def __getitem__(self, index):
        if not self.memmap:
            data_i = self.data[index]
        else:
            data_i = self.get_data_item_from_dir(index)

        if self.label_values is not None and self.label_indices is not None:
            return data_i + (self.label_values[index], self.label_indices[index])
        elif self.label_indices is not None:
            return data_i + (self.label_indices[index],)
        elif self.label_values is not None:
            return data_i + (self.label_values[index],)
        else:
            return data_i

    def __len__(self):
        return self.nr_inst

    def refresh_labels(self, label_values=None, label_indices=None):
        """Refresh label-values and label-indices from given tensors"""
        self.label_values = label_values
        self.label_indices = label_indices


class MultiTaskDataset(Dataset):
    """Dataset to hold feature, label and class tensors for multi-task training and prediction.

    Args:
        *features (tensors): feature tensors, required to have same first
            dimension: nr_inst
        class_value (tensor): class value with shape = (nr_inst) in multi-class classification
        label_values (tensor or None): label values with shape = (nr_inst, num_active_labels) in multi-label classification
        label_indices (tensor or None): label indices with shape = (nr_inst, num_active_labels) in multi-label classification

    Return values depend on the label_values and label_indices:
        if label_values is None and label_indices is not None:
            data[i] = (feature[0][i], feature[1][i], ..., class_value[i], label_values[i], label_indices[i])
        elif label_values is not None:
            data[i] = (feature[0][i], feature[1][i], ..., class_value[i], label_values[i])
        elif label_indices is not None:
            data[i] = (feature[0][i], feature[1][i], ..., class_value[i], label_indices[i])
        else:
            data[i] = (feature[0][i], feature[1][i], class_value[i], ...)
    """

    def __init__(self, *features, class_value, label_values=None, label_indices=None, memmap=False):
        if not memmap:
            self.nr_inst = features[0].size(0)
            self.data = TensorDataset(*features)
        else:
            self.data_path = features[0]
            self.nr_inst = len(glob.glob1(self.data_path, "*.pt"))

        if label_values is not None and label_values.size(0) != self.nr_inst:
            raise ValueError("First dimension mismatch between features and label_values")
        if label_indices is not None and label_indices.size(0) != self.nr_inst:
            raise ValueError("First dimension mismatch between features and label_indices")

        self.class_value = class_value

        self.label_values = label_values
        self.label_indices = label_indices
        self.memmap = memmap

    def classes(self):
        return self.class_value

    def get_data_item_from_dir(self, i):
        assert self.memmap
        data_item = torch.load(f"{self.data_path}/{i}.pt")
        return tuple([data_item["input_ids"][0], data_item["attention_mask"][0], data_item["token_type_ids"][0], torch.tensor(i)])

    def __getitem__(self, index):
        if not self.memmap:
            data_i = self.data[index]
        else:
            data_i = self.get_data_item_from_dir(index)

        if self.label_values is not None and self.label_indices is not None:
            return data_i + (self.class_value[index], self.label_values[index], self.label_indices[index])
        elif self.label_indices is not None:
            return data_i + (self.class_value[index], self.label_indices[index],)
        elif self.label_values is not None:
            return data_i + (self.class_value[index], self.label_values[index],)
        elif self.class_value is not None:
            return data_i + (self.class_value[index],)
        else:
            return data_i

    def __len__(self):
        return self.nr_inst

    def refresh_labels(self, label_values=None, label_indices=None):
        """Refresh label-values and label-indices from given tensors"""
        self.label_values = label_values
        self.label_indices = label_indices
