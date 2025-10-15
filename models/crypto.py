import numpy
import hashlib
from numpy.typing import NDArray


def pseudo_random_generator(
    seed: bytes,
    length: int
) -> NDArray[numpy.float64]:

    if length < 0:
        raise ValueError("Length invalid. Must be > 0")

    output: NDArray[numpy.float64] = numpy.empty(length, dtype=numpy.float64)
    counter: int = 0
    index: int = 0

    while index < length:
        digest: bytes = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()

        for j in range(0, 32, 8):
            if index >= length:
                break

            chunk: bytes = digest[j: j + 8]
            value_u64: int = int.from_bytes(chunk, "big")
            output[index] = (value_u64 / (2 ** 64)) - 0.5
            index += 1

        counter += 1

    return output


def derive_pair_seed(
    client_secret: bytes,
    identifier_a: str,
    identifier_b: str
) -> bytes:

    low, high = sorted([identifier_a, identifier_b])
    material: bytes = client_secret + b"|pair|" + low.encode() + b"|" + high.encode()
    return hashlib.sha256(material).digest()
