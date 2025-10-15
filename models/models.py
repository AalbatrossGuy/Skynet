import numpy
from typing import Tuple
from numpy.typing import NDArray


def sigmoid(
    x: NDArray[numpy.float64] | float
) -> NDArray[numpy.float64]:
    return 1.0 / (1.0 + numpy.exp(-x))


class Logistic:
    def __init__(self, feature_weight: int) -> None:
        self._dim: int = feature_weight + 1
        self.weight: NDArray[numpy.float64] = numpy.zeroes(
            self._dim,
            dtype=numpy.float64
        )

    def get_model_weight(self) -> NDArray[numpy.float64]:
        return self.weight.copy()

    def set_model_weight(
            self,
            new_weight: NDArray[numpy.float64]
    ) -> None:
        self.weight = new_weight.copy()

    def _add_bias(
            self,
            X: NDArray[numpy.float64]
    ) -> NDArray[numpy.float64]:
        return numpy.hstack(
            [X, numpy.ones((X.shape[0], 1), dtype=numpy.float64)]
        )

    def predict_probability(
            self,
            X: NDArray[numpy.float64]
    ) -> NDArray[numpy.float64]:
        X_b: NDArray[numpy.float64] = self._add_bias(X)
        return sigmoid(X_b @ self.weight)

    def predict(
        self,
        X: NDArray[numpy.float64]
    ) -> NDArray[numpy.int64]:
        return (self.predict_probability(X) >= 0.5).astype(numpy.int64)

    def update_local(
        self,
        feature_matrix: NDArray[numpy.float64],
        binary_targets: NDArray[numpy.float64],
        epochs: int = 1,
        learning_rate: float = 0.3
    ) -> NDArray[numpy.float64]:
        w0: NDArray[numpy.float64] = self.get_model_weight()
        X_b: NDArray[numpy.float64] = self._add_bias(feature_matrix)
        weight: NDArray[numpy.float64] = w0.copy()

        n: int = feature_matrix.shape[0]
        for _ in range(epochs):
            p: NDArray[numpy.float64] = sigmoid(X_b @ weight)
            gradient: NDArray[numpy.float64] = (X_b @ (p - binary_targets)) / n
            weight -= learning_rate * gradient

        delta: NDArray[numpy.float64] = weight - w0
        self.set_model_weight(weight)
        return delta


