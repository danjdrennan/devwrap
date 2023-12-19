import pytest
import torch

from wrapper import ensure_device


class Model(torch.nn.Module):
    """Min example of a torch model with new tensors created in the method."""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """This breaks if we put x and `self.param` on GPU."""
        eps = torch.randn(1)
        return self.param * x + eps


class DeviceSafeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    @ensure_device
    def forward(self, x):
        """The decorator preserves the docstring of the forward method."""
        # Using the decorator ensures `eps` is put on the same device as `x`.
        eps = torch.randn(1)
        return self.param * x + eps


def test_fails():
    device = torch.device("cuda", index=0)

    model = Model().to(device)
    x = torch.tensor(1.0, device=device)

    with pytest.raises(RuntimeError):
        model(x)


def test_passes():
    device = torch.device("cuda", index=0)
    model = DeviceSafeModel().to(device)
    x = torch.tensor(1.0, device=device)

    with torch.no_grad():
        assert model(x).device == device


def test_preserves_docstring():
    assert (
        DeviceSafeModel.forward.__doc__
        == "The decorator preserves the docstring of the forward method."
    )
