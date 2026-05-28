import math

import numpy as np

from strategies.shared import mean_normalize_embeddings


def test_mean_normalize_embeddings_single_vector_unit_norm():
    e = np.array([3.0, 4.0])
    out = mean_normalize_embeddings([e])
    assert out.shape == (2,)
    assert math.isclose(float(np.linalg.norm(out)), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_mean_normalize_embeddings_empty_list_raises():
    # current implementation requires at least one embedding
    import pytest

    with pytest.raises(ValueError):
        mean_normalize_embeddings([])
