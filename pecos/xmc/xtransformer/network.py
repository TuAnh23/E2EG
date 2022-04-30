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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pecos.utils.featurization.text.vectorizers import TransformerModelClass
from pecos.xmc import MLModel
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizerFast,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLNetConfig,
    XLNetModel,
    XLNetPreTrainedModel,
    XLNetTokenizerFast,
    DistilBertModel,
    DistilBertConfig,
    DistilBertTokenizerFast,
    DistilBertPreTrainedModel,
)
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import SequenceSummary

from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    ROBERTA_INPUTS_DOCSTRING,
    ROBERTA_START_DOCSTRING,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLM_ROBERTA_START_DOCSTRING
from transformers.models.xlnet.modeling_xlnet import (
    XLNET_INPUTS_DOCSTRING,
    XLNET_START_DOCSTRING,
)
from transformers.models.distilbert.modeling_distilbert import (
    DISTILBERT_INPUTS_DOCSTRING,
    DISTILBERT_START_DOCSTRING,
)


class HingeLoss(nn.Module):
    """Hinge loss function module for multi-label classification"""

    def __init__(self, margin=1.0, power=2, cost_weighted=False):
        """
        Args:
            margin (float, optional): margin for the hinge loss. Default 1.0
            power (int, optional): exponent for the hinge loss. Default to 2 for squared-hinge
            cost_weighted (bool, optional): whether to use label value as weight. Default False
        """
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.power = power
        self.cost_weighted = cost_weighted

    def forward(self, z, y, C_pos=1.0, C_neg=1.0):
        """Compute the hinge loss

        Args:
            z (torch.tensor): predicted matrix of size: (batch_size * output_size)
            y (torch.tensor): 0/1 ground truth of size: (batch_size * output_size)
            C_pos (float, optional): positive penalty for the hinge loss. Default 1.0
            C_neg (float, optional): negative penalty for the hinge loss. Default 1.0

        Returns:
            loss (torch.tensor): the tensor of average loss
        """
        # convert y into {-1,1}
        y_binary = (y > 0).float()
        y_new = 2.0 * y_binary - 1.0

        # Hinge loss
        loss = F.relu(self.margin - y_new * z)
        loss = loss**self.power
        # if y = [1, 4, 0, 0]
        if self.cost_weighted:
            # weight = [1, 4, 1, 1]
            loss = loss * (C_pos * y + C_neg * (1.0 - y_binary))
        else:
            # weight = [1, 1, 1, 1]
            loss = loss * (C_pos * y_binary + C_neg * (1.0 - y_binary))
        return loss.mean(1)


class TransformerLinearXMCHead(nn.Module):
    """XMC head for Transformers

    Containing label weight embeddings and label bias embeddings
    """

    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.label_pad = num_labels
        self.num_labels = num_labels
        self.W = nn.Embedding(num_labels + 1, hidden_size, padding_idx=self.label_pad)
        self.b = nn.Embedding(num_labels + 1, 1, padding_idx=self.label_pad)

        self.random_init()

    @property
    def device(self):
        return self.W.weight.device

    def random_init(self):
        """Initialize the weight and bias embeddings

        Initialize label weight embedding with N(0, 0.02) while keeping PAD
        column to be 0. Initialize label bias embedding with 0.
        """
        mat = 0.02 * np.random.randn(self.label_pad, self.W.weight.shape[1])
        mat = np.hstack([mat, np.zeros([mat.shape[0], 1])])
        self.init_from(mat)

    def inherit(self, prev_head, C):
        prev_W = prev_head.W.weight[:-1, :].detach().numpy()
        prev_b = prev_head.b.weight[:-1, :].detach().numpy()

        cur_W = C * prev_W
        cur_b = C * prev_b

        mat = np.hstack([cur_W, cur_b])

        self.init_from(mat)

    def bootstrap(self, prob, **kwargs):
        """Initialize head with weights learned from linear model using transformer embeddings

        Args:
            prob (MLProblem): the multi-label problem to bootstrap with
            kwargs:
                Cp (float): the weight on positive samples. Default 100.0
                Cn (float): the weight on negative samples. Default 100.0
                threshold (float): the threshold to sparsify the model
        """
        # use large Cp and Cn to reduce regularization
        Cp = kwargs.get("Cp", 100.0)
        Cn = kwargs.get("Cn", 100.0)
        # sparsification is by default turned off since dense features are used
        threshold = kwargs.get("threshold", 0)
        mat = MLModel.train(prob, threshold=threshold, Cp=Cp, Cn=Cn)
        mat = mat.W.toarray().T
        self.init_from(mat)

    def init_from(self, mat):
        """Initialize the weight and bias embeddings with given matrix

        Args:
            mat (ndarray): matrix used for initialize, shape = (nr_labels, hidden_size + 1)
        """
        if not isinstance(mat, np.ndarray):
            raise ValueError("Expect ndarray to initialize label embedding")
        if mat.shape[0] != self.label_pad:
            raise ValueError("nr_labels mismatch!")

        # add padding index by appending an all-zero row
        mat = np.vstack([mat, np.zeros([1, mat.shape[1]])])
        # split weight and bias
        self.W = nn.Embedding.from_pretrained(
            torch.FloatTensor(mat[:, :-1]),
            freeze=False,
            sparse=True,
            padding_idx=self.label_pad,
        )
        self.b = nn.Embedding.from_pretrained(
            torch.FloatTensor(mat[:, -1]).view((self.label_pad + 1, 1)),
            freeze=False,
            sparse=True,
            padding_idx=self.label_pad,
        )

    def forward(self, pooled_output=None, output_indices=None, num_device=1):
        if output_indices is None:
            # for parallel training, need to send a copy to each device
            W_act = self.W.weight[:-1, :].repeat(num_device, 1, 1)
            b_act = self.b.weight[:-1].repeat(num_device, 1, 1)
        else:
            output_indices = output_indices.to(self.device)
            W_act = self.W(output_indices)  # (batch_size, nr_act_labels, dim)
            b_act = self.b(output_indices)
        return W_act, b_act


@add_start_docstrings(
    """Bert Model with mutli-label classification head on top for XMC.\n""",
    BERT_START_DOCSTRING,
)
class BertForXMC(BertPreTrainedModel):
    """
    Examples:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForXMC.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config):
        super(BertForXMC, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def init_from(self, model):
        self.bert = model.bert

    @add_start_docstrings(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each label,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        instance_hidden_states = outputs.last_hidden_state

        logits = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }


@add_start_docstrings(
    """Bert Model with two classification heads on top: one for eXtreme Multi-label Classification (XMC) and one for 
     the common Multi-class Classification.\n""",
    BERT_START_DOCSTRING,
)
class BertForMultiTask(BertPreTrainedModel):
    """
    Examples:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultiTask.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config, num_classes, mclass_pred_hyperparam=None, freeze_mclass_head=False,
                 init_scheme_mclass_head=None):
        super(BertForMultiTask, self).__init__(config)
        self.num_labels = config.num_labels  # Number of labels for multi-label XMC
        self.num_classes = num_classes

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # For multi-class classification
        self.freeze_mclass_head = freeze_mclass_head
        self.mclass_pred_hyperparam = mclass_pred_hyperparam
        numb_layers_mclass_pred = self.mclass_pred_hyperparam['numb_layers_mclass_pred']
        mclass_pred_dropout_prob = self.mclass_pred_hyperparam['mclass_pred_dropout_prob']
        mclass_pred_batchnorm = self.mclass_pred_hyperparam['mclass_pred_batchnorm']
        mclass_pred_hidden_size = self.mclass_pred_hyperparam['mclass_pred_hidden_size']

        self.mclass_seq = nn.Sequential()

        if numb_layers_mclass_pred < 1:
            raise RuntimeError("Number of layers for multi-class prediction should be at least 1.")

        for i in range(0, numb_layers_mclass_pred):
            if i == numb_layers_mclass_pred-1:
                # In the last layer, output number of values correspond to the number of classes
                linear_output_size = self.num_classes
            else:
                linear_output_size = mclass_pred_hidden_size

            if i == 0:
                # In the first layer, input size is the hidden size of BERT
                linear_input_size = config.hidden_size
            else:
                linear_input_size = mclass_pred_hidden_size

            if i >= 1:
                self.mclass_seq.add_module(f"dropout{i-1}", nn.Dropout(mclass_pred_dropout_prob))
            self.mclass_seq.add_module(f"linear{i}", nn.Linear(linear_input_size, linear_output_size))

            if i != numb_layers_mclass_pred - 1:
                # Batchnorm and relu in between layers
                if mclass_pred_batchnorm == "yes":
                    self.mclass_seq.add_module(f"batchnorm{i}", nn.BatchNorm1d(linear_output_size))
                self.mclass_seq.add_module(f"relu{i}", nn.ReLU())

        self.init_weights()

        if init_scheme_mclass_head is not None and init_scheme_mclass_head != 'default':
            # Replace the default N(0, 0.02) initialization
            def manual_weight_init(m):
                if type(m) == nn.Linear:
                    if init_scheme_mclass_head == 'uniform':
                        nn.init.uniform_(m.weight, a=-1.0/m.in_features, b=1.0/m.in_features)
                        if m.bias is not None:
                            m.bias.data.fill_(1.0/m.in_features)
                    elif init_scheme_mclass_head == 'constant':
                        nn.init.constant_(m.weight, val=1.0/m.in_features)
                        if m.bias is not None:
                            m.bias.data.fill_(1.0/m.in_features)

            self.mclass_seq.apply(manual_weight_init)

        for param in self.mclass_seq.parameters():
            if self.freeze_mclass_head:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def init_from(self, model):
        self.bert = model.bert
        self.mclass_seq = model.mclass_seq
        for param in self.mclass_seq.parameters():
            if self.freeze_mclass_head:
                param.requires_grad = False
            else:
                param.requires_grad = True


    @add_start_docstrings(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits_mlabel': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each LABEL
                in the prediction head for multi-label classification XMC,
                 'logits_mclass': (:obj:`torch.FloatTensor` of shape (batch_size, num_classes)) pred logits for each CLASS
                in the prediction head for multi-class classification,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        instance_hidden_states = outputs.last_hidden_state

        # For multi-class classification head
        logits_mclass = self.mclass_seq(pooled_output)

        # For multi-label classification XMC head
        logits_mlabel = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits_mlabel = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits_mlabel": logits_mlabel,
            "logits_mclass": logits_mclass,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }


@add_start_docstrings(
    """Roberta Model with mutli-label classification head on top for XMC.\n""",
    ROBERTA_START_DOCSTRING,
)
class RobertaForXMC(RobertaPreTrainedModel):
    """
    Examples:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForXMC.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config):
        super(RobertaForXMC, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def init_from(self, model):
        self.roberta = model.roberta

    @add_start_docstrings(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each label,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        instance_hidden_states = outputs.last_hidden_state
        logits = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }

@add_start_docstrings(
    """Roberta Model with two classification heads on top: one for eXtreme Multi-label Classification (XMC) and one for 
     the common Multi-class Classification.\n""",
    ROBERTA_START_DOCSTRING,
)
class RobertaForMultiTask(RobertaPreTrainedModel):
    """
    Examples:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultiTask.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config, num_classes, mclass_pred_hyperparam=None, freeze_mclass_head=False,
                 init_scheme_mclass_head=None):
        super(RobertaForMultiTask, self).__init__(config)
        self.num_labels = config.num_labels  # Number of labels for multi-label XMC
        self.num_classes = num_classes

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # For multi-class classification
        self.freeze_mclass_head = freeze_mclass_head
        self.mclass_pred_hyperparam = mclass_pred_hyperparam
        numb_layers_mclass_pred = self.mclass_pred_hyperparam['numb_layers_mclass_pred']
        mclass_pred_dropout_prob = self.mclass_pred_hyperparam['mclass_pred_dropout_prob']
        mclass_pred_batchnorm = self.mclass_pred_hyperparam['mclass_pred_batchnorm']
        mclass_pred_hidden_size = self.mclass_pred_hyperparam['mclass_pred_hidden_size']

        self.mclass_seq = nn.Sequential()

        if numb_layers_mclass_pred < 1:
            raise RuntimeError("Number of layers for multi-class prediction should be at least 1.")

        for i in range(0, numb_layers_mclass_pred):
            if i == numb_layers_mclass_pred-1:
                # In the last layer, output number of values correspond to the number of classes
                linear_output_size = self.num_classes
            else:
                linear_output_size = mclass_pred_hidden_size

            if i == 0:
                # In the first layer, input size is the hidden size of Roberta
                linear_input_size = config.hidden_size
            else:
                linear_input_size = mclass_pred_hidden_size

            if i >= 1:
                self.mclass_seq.add_module(f"dropout{i-1}", nn.Dropout(mclass_pred_dropout_prob))
            self.mclass_seq.add_module(f"linear{i}", nn.Linear(linear_input_size, linear_output_size))

            if i != numb_layers_mclass_pred - 1:
                # Batchnorm and relu in between layers
                if mclass_pred_batchnorm == "yes":
                    self.mclass_seq.add_module(f"batchnorm{i}", nn.BatchNorm1d(linear_output_size))
                self.mclass_seq.add_module(f"relu{i}", nn.ReLU())

        self.init_weights()

        if init_scheme_mclass_head is not None and init_scheme_mclass_head != 'default':
            # Replace the default N(0, 0.02) initialization
            def manual_weight_init(m):
                if type(m) == nn.Linear:
                    if init_scheme_mclass_head == 'uniform':
                        nn.init.uniform_(m.weight, a=-1.0/m.in_features, b=1.0/m.in_features)
                        if m.bias is not None:
                            m.bias.data.fill_(1.0/m.in_features)
                    elif init_scheme_mclass_head == 'constant':
                        nn.init.constant_(m.weight, val=1.0/m.in_features)
                        if m.bias is not None:
                            m.bias.data.fill_(1.0/m.in_features)

            self.mclass_seq.apply(manual_weight_init)

        for param in self.mclass_seq.parameters():
            if self.freeze_mclass_head:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def init_from(self, model):
        self.roberta = model.roberta
        self.mclass_seq = model.mclass_seq
        for param in self.mclass_seq.parameters():
            if self.freeze_mclass_head:
                param.requires_grad = False
            else:
                param.requires_grad = True


    @add_start_docstrings(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits_mlabel': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each LABEL
                in the prediction head for multi-label classification XMC,
                 'logits_mclass': (:obj:`torch.FloatTensor` of shape (batch_size, num_classes)) pred logits for each CLASS
                in the prediction head for multi-class classification,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        instance_hidden_states = outputs.last_hidden_state

        # For multi-class classification head
        logits_mclass = self.mclass_seq(pooled_output)

        # For multi-label classification XMC head
        logits_mlabel = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits_mlabel = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits_mlabel": logits_mlabel,
            "logits_mclass": logits_mclass,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }


@add_start_docstrings(
    """XLM-Roberta Model with mutli-label classification head on top for XMC.\n""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForXMC(RobertaForXMC):
    """
    This class overrides :class:`RobertaForXMC`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig  # type: ignore


@add_start_docstrings(
    """XLNet Model with mutli-label classification head on top for XMC.\n""",
    XLNET_START_DOCSTRING,
)
class XLNetForXMC(XLNetPreTrainedModel):
    """
    Examples:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = XLNetForXMC.from_pretrained('xlnet-large-cased')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config):
        super(XLNetForXMC, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def init_from(self, model):
        self.transformer = model.transformer

    @add_start_docstrings(XLNET_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each label,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        instance_hidden_states = outputs.last_hidden_state
        pooled_output = self.sequence_summary(instance_hidden_states)

        logits = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }


@add_start_docstrings(
    """DistilBert Model with mutli-label classification head on top for XMC.\n""",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForXMC(DistilBertPreTrainedModel):
    """
    Examples:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForXMC.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config):
        super(DistilBertForXMC, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def init_from(self, model):
        self.distilbert = model.distilbert

    @add_start_docstrings(DISTILBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each label,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        instance_hidden_states = outputs.last_hidden_state
        logits = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }


@add_start_docstrings(
    """DistilBert Model with two classification heads on top: one for eXtreme Multi-label Classification (XMC) and one for 
     the common Multi-class Classification.\n""",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForMultiTask(DistilBertPreTrainedModel):
    """
    Examples:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForMultiTask.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("iphone 11 case", add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs["hidden_states"]
    """

    def __init__(self, config, num_classes, mclass_pred_hyperparam=None, freeze_mclass_head=False,
                 init_scheme_mclass_head=None):
        super(DistilBertForMultiTask, self).__init__(config)
        self.num_labels = config.num_labels  # Number of labels for multi-label XMC
        self.num_classes = num_classes

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        # For multi-class classification
        self.freeze_mclass_head = freeze_mclass_head
        self.mclass_pred_hyperparam = mclass_pred_hyperparam
        numb_layers_mclass_pred = self.mclass_pred_hyperparam['numb_layers_mclass_pred']
        mclass_pred_dropout_prob = self.mclass_pred_hyperparam['mclass_pred_dropout_prob']
        mclass_pred_batchnorm = self.mclass_pred_hyperparam['mclass_pred_batchnorm']
        mclass_pred_hidden_size = self.mclass_pred_hyperparam['mclass_pred_hidden_size']

        self.mclass_seq = nn.Sequential()

        if numb_layers_mclass_pred < 1:
            raise RuntimeError("Number of layers for multi-class prediction should be at least 1.")

        for i in range(0, numb_layers_mclass_pred):
            if i == numb_layers_mclass_pred-1:
                # In the last layer, output number of values correspond to the number of classes
                linear_output_size = self.num_classes
            else:
                linear_output_size = mclass_pred_hidden_size

            if i == 0:
                # In the first layer, input size is the hidden size of DistilBert
                linear_input_size = config.hidden_size
            else:
                linear_input_size = mclass_pred_hidden_size

            if i >= 1:
                self.mclass_seq.add_module(f"dropout{i-1}", nn.Dropout(mclass_pred_dropout_prob))
            self.mclass_seq.add_module(f"linear{i}", nn.Linear(linear_input_size, linear_output_size))

            if i != numb_layers_mclass_pred - 1:
                # Batchnorm and relu in between layers
                if mclass_pred_batchnorm == "yes":
                    self.mclass_seq.add_module(f"batchnorm{i}", nn.BatchNorm1d(linear_output_size))
                self.mclass_seq.add_module(f"relu{i}", nn.ReLU())

        self.init_weights()

        if init_scheme_mclass_head is not None and init_scheme_mclass_head != 'default':
            # Replace the default N(0, 0.02) initialization
            def manual_weight_init(m):
                if type(m) == nn.Linear:
                    if init_scheme_mclass_head == 'uniform':
                        nn.init.uniform_(m.weight, a=-1.0/m.in_features, b=1.0/m.in_features)
                        if m.bias is not None:
                            m.bias.data.fill_(1.0/m.in_features)
                    elif init_scheme_mclass_head == 'constant':
                        nn.init.constant_(m.weight, val=1.0/m.in_features)
                        if m.bias is not None:
                            m.bias.data.fill_(1.0/m.in_features)

            self.mclass_seq.apply(manual_weight_init)

        for param in self.mclass_seq.parameters():
            if self.freeze_mclass_head:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def init_from(self, model):
        self.distilbert = model.distilbert
        self.mclass_seq = model.mclass_seq
        for param in self.mclass_seq.parameters():
            if self.freeze_mclass_head:
                param.requires_grad = False
            else:
                param.requires_grad = True


    @add_start_docstrings(DISTILBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_embedding=None,
    ):
        r"""
        Returns:
          :obj:`dict` containing:
                {'logits_mlabel': (:obj:`torch.FloatTensor` of shape (batch_size, num_labels)) pred logits for each LABEL
                in the prediction head for multi-label classification XMC,
                 'logits_mclass': (:obj:`torch.FloatTensor` of shape (batch_size, num_classes)) pred logits for each CLASS
                in the prediction head for multi-class classification,
                 'pooled_output': (:obj:`torch.FloatTensor` of shape (batch_size, hidden_dim)) input sequence embedding vector,
                 'hidden_states': (:obj:`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_dim)) the last layer hidden states,
                }
        """
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        instance_hidden_states = outputs.last_hidden_state

        # For multi-class classification head
        logits_mclass = self.mclass_seq(pooled_output)

        # For multi-label classification XMC head
        logits_mlabel = None
        if label_embedding is not None:
            W_act, b_act = label_embedding
            W_act = W_act.to(pooled_output.device)
            b_act = b_act.to(pooled_output.device)
            logits_mlabel = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return {
            "logits_mlabel": logits_mlabel,
            "logits_mclass": logits_mclass,
            "pooled_output": pooled_output,
            "hidden_states": instance_hidden_states,
        }


ENCODER_CLASSES = {
    "bert": TransformerModelClass(BertConfig, BertForXMC, BertTokenizerFast),
    "bert-multitask": TransformerModelClass(BertConfig, BertForMultiTask, BertTokenizerFast),
    "roberta": TransformerModelClass(RobertaConfig, RobertaForXMC, RobertaTokenizerFast),
    "roberta-multitask": TransformerModelClass(RobertaConfig, RobertaForMultiTask, RobertaTokenizerFast),
    "xlm-roberta": TransformerModelClass(
        XLMRobertaConfig, XLMRobertaForXMC, XLMRobertaTokenizerFast
    ),
    "xlnet": TransformerModelClass(XLNetConfig, XLNetForXMC, XLNetTokenizerFast),
    "distilbert": TransformerModelClass(
        DistilBertConfig, DistilBertForXMC, DistilBertTokenizerFast
    ),
    "distilbert-multitask": TransformerModelClass(
        DistilBertConfig, DistilBertForMultiTask, DistilBertTokenizerFast
    ),
}
