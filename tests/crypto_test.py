import numpy as np
from models.crypto import pseudo_random_generator, derive_pair_seed


def test_prg_is_deterministic():
    a = pseudo_random_generator(b"seed", 10)
    b = pseudo_random_generator(b"seed", 10)
    assert np.allclose(a, b)


def test_pairwise_seed_is_symmetric():
    s1 = derive_pair_seed(b"k", "A", "B")
    s2 = derive_pair_seed(b"k", "B", "A")
    assert s1 == s2
