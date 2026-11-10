import torch

from turboquant import random_orthogonal_matrix


def test_random_rotation_is_orthogonal() -> None:
    rotation = random_orthogonal_matrix(128, seed=7)
    identity = torch.eye(128, dtype=rotation.dtype)
    torch.testing.assert_close(rotation @ rotation.T, identity, atol=1e-12, rtol=1e-12)


def test_random_rotation_is_reproducible() -> None:
    first = random_orthogonal_matrix(16, seed=3)
    second = random_orthogonal_matrix(16, seed=3)
    torch.testing.assert_close(first, second)
