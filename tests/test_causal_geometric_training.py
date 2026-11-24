import types

import pytest
import torch
from torch import nn

from igbundle.core.config import IGBundleConfig
from igbundle.geometry.riemannian import FiberBundleLambdaCalculus, RiemannianGeometry
from igbundle.training.geometric_training import RiemannianOptimizer
from igbundle.training.geometric_trainer import GeometricTrainer as HFGeometricTrainer


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=7, hidden_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens):
        return types.SimpleNamespace(logits=self.output(self.embedding(tokens)))


def test_fisher_refresh_preserves_task_gradient_and_updates_parameters():
    torch.manual_seed(7)
    model = TinyLanguageModel()
    optimizer = RiemannianOptimizer(model.parameters(), lr=1e-2, momentum=0.0)
    tokens = torch.tensor([[0, 1, 2], [2, 3, 4]])
    logits = model(tokens).logits
    nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), tokens.view(-1)
    ).backward()
    gradients = [p.grad.detach().clone() for p in model.parameters()]
    parameters = [p.detach().clone() for p in model.parameters()]

    optimizer.update_fisher(model, tokens)
    for parameter, expected in zip(model.parameters(), gradients):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected)

    optimizer.step()
    assert all(
        not torch.equal(parameter, before)
        for parameter, before in zip(model.parameters(), parameters)
    )


def test_lambda_abstraction_contract_and_shape():
    config = IGBundleConfig(latent_dim=4, num_categories=3, num_components=2)
    calculus = FiberBundleLambdaCalculus(config)
    variable = torch.randn(2, 3, 2, 7)
    body = torch.randn_like(variable)
    assert calculus.lambda_abstraction(variable, body).shape == variable.shape

    with pytest.raises(ValueError, match="final dimension 7"):
        calculus.lambda_abstraction(variable[..., :-1], body[..., :-1])


def test_conformal_connection_is_metric_derived_and_torsion_free():
    config = IGBundleConfig(latent_dim=2, num_categories=3, num_components=1)
    geometry = RiemannianGeometry(config)
    position = torch.tensor([[[[0.4, -0.2]]]])
    metric = geometry.get_metric(position)
    gamma = geometry.christoffel_symbols(position, metric)

    assert geometry.connection_contract()["model"] == "conformal_levi_civita"
    assert geometry.connection_contract()["flat"] is False
    assert torch.count_nonzero(gamma) > 0
    torch.testing.assert_close(gamma, gamma.transpose(-1, -2))

    first = torch.tensor([[[[0.3, -0.1]]]])
    second = torch.tensor([[[[-0.2, 0.6]]]])
    explicit = torch.einsum('...kij,...i,...j->...k', gamma, first, second)
    torch.testing.assert_close(
        geometry.connection_action(position, first, second), explicit
    )


def test_closed_loop_transport_returns_witnessed_holonomy_defect():
    config = IGBundleConfig(latent_dim=2, num_categories=3, num_components=1)
    geometry = RiemannianGeometry(config)
    vector = torch.tensor([[[[1.0, 0.0]]]])
    loop = torch.tensor(
        [[[[[0.4, 0.2], [0.7, 0.2], [0.7, 0.5], [0.4, 0.5], [0.4, 0.2]]]]]
    )
    witness = geometry.holonomy(vector, loop)

    assert witness["returned"].shape == vector.shape
    assert witness["defect_norm"].shape == vector.shape[:-1]
    assert torch.all(witness["defect_norm"] > 0)


def test_hf_geometric_mode_fails_closed_without_state():
    with pytest.raises(RuntimeError, match="Refusing to silently run ordinary SFT"):
        HFGeometricTrainer._extract_geometric_state(
            {"hidden_states": (torch.randn(1, 2, 3),)}, required=True
        )


def test_hf_geometric_state_collector_is_an_explicit_transport():
    sentinel = object()
    collector = types.SimpleNamespace(last_state=sentinel)
    assert HFGeometricTrainer._extract_geometric_state(
        {}, state_collector=collector, required=True
    ) is sentinel
