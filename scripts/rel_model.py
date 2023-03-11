from typing import Callable, List, Tuple

import spacy
import torch
from spacy.tokens import Doc, Span
from thinc.api import Model, PyTorchWrapper, chain, with_array
from thinc.types import Floats2d, Ints1d, Ragged, cast
from torch import nn


@spacy.registry.architectures("rel_model.v1")
def create_relation_model(
    create_instance_tensor: Model[List[Doc], Floats2d],
    classification_layer: Model[Floats2d, Floats2d],
) -> Model[List[Doc], Floats2d]:
    model = chain(create_instance_tensor, with_array(classification_layer))
    return model


@spacy.registry.architectures("rel_classification_layer.v1")
def create_classification_layer(
    nO: int = None, nI: int = None, nH: List[int] = None
) -> Model[Floats2d, Floats2d]:
    """Classifies tensor representing a relation.
    """
    torch_model = TorchRelationshipClassifier(in_features=nI, out_features=nO, hidden_dims=nH)
    model = PyTorchWrapper(torch_model)
    model.name = "Classification layer."
    model.init = init_cls
    return model


@spacy.registry.misc("rel_instance_generator.v1")
def create_instances(max_length: int) -> Callable[[Doc], List[Tuple[Span, Span]]]:
    """Responsible for generating pairs of entities (as spans) that can be related.
    """
    def get_candidates(doc: Doc) -> List[Tuple[Span, Span]]:
        candidates = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1 != ent2:
                    if max_length and abs(ent2.start - ent1.start) <= max_length:
                        candidates.append((ent1, ent2))
        return candidates

    return get_candidates


@spacy.registry.architectures("rel_instance_tensor.v1")
def create_tensors(
    tok2vec: Model[List[Doc], List[Floats2d]],
    pooling: Model[Ragged, Floats2d],
    get_instances: Callable[[Doc], List[Tuple[Span, Span]]],
) -> Model[List[Doc], Floats2d]:
    """Creates model that processes tok2vec output to embed pairs of entities into a single tensor.
    """

    return Model(
        "instance_tensors",
        instance_forward,
        layers=[tok2vec, pooling],
        refs={"tok2vec": tok2vec, "pooling": pooling},
        attrs={"get_instances": get_instances},
        init=instance_init,
    )


def instance_forward(model: Model[List[Doc], Floats2d], docs: List[Doc], is_train: bool) -> Tuple[Floats2d, Callable]:
    # pylint: disable=too-many-locals
    pooling = model.get_ref("pooling")
    tok2vec = model.get_ref("tok2vec")
    get_instances = model.attrs["get_instances"]
    all_instances = [get_instances(doc) for doc in docs]
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)

    ents = []
    lengths = []

    for doc_nr, (instances, tokvec) in enumerate(zip(all_instances, tokvecs)):  # pylint: disable=unused-variable
        token_indices = []
        for instance in instances:
            for ent in instance:
                token_indices.extend(list(range(ent.start, ent.end)))
                lengths.append(ent.end - ent.start)
        ents.append(tokvec[token_indices])
    lengths = cast(Ints1d, model.ops.asarray(lengths, dtype="int32"))
    entities = Ragged(model.ops.flatten(ents), lengths)
    pooled, bp_pooled = pooling(entities, is_train)

    # Reshape so that pairs of rows are concatenated
    relations = model.ops.reshape2f(pooled, -1, pooled.shape[1] * 2)

    def backprop(d_relations: Floats2d) -> List[Doc]:
        d_pooled = model.ops.reshape2f(d_relations, d_relations.shape[0] * 2, -1)
        d_ents = bp_pooled(d_pooled).data
        d_tokvecs = []
        ent_index = 0
        for doc_nr, instances in enumerate(all_instances):
            shape = tokvecs[doc_nr].shape
            d_tokvec = model.ops.alloc2f(*shape)
            count_occ = model.ops.alloc2f(*shape)
            for instance in instances:
                for ent in instance:
                    d_tokvec[ent.start: ent.end] += d_ents[ent_index]
                    count_occ[ent.start: ent.end] += 1
                    ent_index += ent.end - ent.start
            d_tokvec /= count_occ + 0.00000000001
            d_tokvecs.append(d_tokvec)

        d_docs = bp_tokvecs(d_tokvecs)
        return d_docs

    return relations, backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:  # pylint: disable=unused-argument
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model


def init_cls(
    model: Model[Floats2d, Floats2d],
    X: Floats2d = None,
    Y: Floats2d = None,
) -> Model[Floats2d, Floats2d]:
    # pylint: disable=protected-access
    if X is not None and model.has_dim("nI") is None:
        model.shims[0]._model.set_input_shape(X.shape[1])
        model.set_dim("nI", X.shape[1])
    if Y is not None and model.has_dim("nO") is None:
        model.shims[0]._model.set_output_shape(Y.shape[1])
        model.set_dim("nO", Y.shape[1])

    return model


def is_dropout_module(
    module: nn.Module,
    dropout_modules: List[nn.Module] = None,
) -> bool:
    """Detect if a PyTorch Module is a Dropout layer
    module (nn.Module): Module to check
    dropout_modules (List[nn.Module], optional): List of Modules that count as Dropout layers.
    RETURNS (bool): True if module is a Dropout layer.
    """
    if dropout_modules is None:
        dropout_modules = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]

    for m in dropout_modules:
        if isinstance(module, m):
            return True
    return False


class TorchRelationshipClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dims: List[int]):
        super().__init__()

        if hidden_dims is None:
            raise ValueError("TorchRelationshipClassifier cannot be initialized without hidden dimensions.")

        self.in_features = in_features or 1
        self.out_features = out_features or 1

        self.input_layer = nn.Linear(self.in_features, self.hidden_dims[0])
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.out_features)

        self.layers = nn.Sequential([
            self.input_layer,
            *[
                nn.Linear(in_feats, out_feats)
                for (in_feats, out_feats) in zip(self.hidden_dims[:-1], self.hidden_dims[1:])
            ],
            self.output_layer,
            nn.Softmax(),
        ])

    def forward(self, inputs: torch.Tensor):
        return self.layers(inputs)

    def _set_layer_shape(self, layer: nn.Module, in_features: int, out_features: int):
        """Dynamically set the shape of a layer
        name (str): Layer name
        in_features (int): New input shape
        out_features (int): New output shape
        """
        with torch.no_grad():
            layer.out_features = out_features
            layer.in_features = in_features
            layer.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.Tensor(out_features))
            layer.reset_parameters()

    def set_input_shape(self, in_features: int):
        """Dynamically set the shape of the input layer
        in_features (int): New input layer shape
        """
        self._set_layer_shape(self.input_layer, in_features, self.hidden_dim[0])

    def set_output_shape(self, out_features: int):
        """Dynamically set the shape of the output layer
        nO (int): New output layer shape
        """
        self._set_layer_shape(self.output_layer, self.hidden_dim[-1], out_features)

    def set_dropout_rate(self, dropout: float):
        """Set the dropout rate of all Dropout layers in the model.
        dropout (float): Dropout rate to set
        """
        dropout_layers = [
            module for module in self.modules() if is_dropout_module(module)
        ]
        for layer in dropout_layers:
            layer.p = dropout
