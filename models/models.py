# import numpy
# from numpy.typing import NDArray
#
#
# def sigmoid(
#     x: NDArray[numpy.float64] | float
# ) -> NDArray[numpy.float64]:
#     return 1.0 / (1.0 + numpy.exp(-x))
#
#
# class Logistic:
#     def __init__(self, feature_weight: int) -> None:
#         self._dim: int = feature_weight + 1
#         self.weight: NDArray[numpy.float64] = numpy.zeros(
#             self._dim,
#             dtype=numpy.float64
#         )
#
#     def get_model_weight(self) -> NDArray[numpy.float64]:
#         return self.weight.copy()
#
#     def set_model_weight(
#         self,
#         new_weight: NDArray[numpy.float64]
#     ) -> None:
#         self.weight = new_weight.copy()
#
#     def _add_bias(
#         self,
#         X: NDArray[numpy.float64]
#     ) -> NDArray[numpy.float64]:
#         return numpy.hstack(
#             [X, numpy.ones((X.shape[0], 1), dtype=numpy.float64)]
#         )
#
#     def predict_probability(
#         self,
#         X: NDArray[numpy.float64]
#     ) -> NDArray[numpy.float64]:
#         X_b: NDArray[numpy.float64] = self._add_bias(X)
#         return sigmoid(X_b @ self.weight)
#
#     def predict(
#         self,
#         X: NDArray[numpy.float64]
#     ) -> NDArray[numpy.int64]:
#         return (self.predict_probability(X) >= 0.5).astype(numpy.int64)
#
#     def update_local(
#         self,
#         feature_matrix: NDArray[numpy.float64],
#         binary_targets: NDArray[numpy.float64],
#         epochs: int = 1,
#         learning_rate: float = 0.3
#     ) -> NDArray[numpy.float64]:
#         w0: NDArray[numpy.float64] = self.get_model_weight()
#         X_b: NDArray[numpy.float64] = self._add_bias(feature_matrix)
#         weight: NDArray[numpy.float64] = w0.copy()
#
#         n: int = feature_matrix.shape[0]
#         for _ in range(epochs):
#             p: NDArray[numpy.float64] = sigmoid(X_b @ weight)
#             gradient: NDArray[numpy.float64] = (X_b @ (p - binary_targets)) / n
#             weight -= learning_rate * gradient
#
#         delta: NDArray[numpy.float64] = weight - w0
#         self.set_model_weight(weight)
#         return delta

import numpy as np
from numpy.typing import NDArray


def sigmoid(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


class Logistic:
    def __init__(self, feature_weight: int) -> None:
        self._dim: int = feature_weight + 1           # +1 for bias
        self.weight: NDArray[np.float64] = np.zeros(self._dim, dtype=np.float64)

    def get_model_weight(self) -> NDArray[np.float64]:
        return self.weight.copy()

    def set_model_weight(self, new_weight: NDArray[np.float64]) -> None:
        w = np.asarray(new_weight, dtype=np.float64).ravel()   # ensure 1-D
        if w.size != self._dim:
            raise ValueError(f"set_model_weight: expected {self._dim} values, got {w.size}")
        self.weight = w

    def _add_bias(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])

    def predict_probability(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X_b = self._add_bias(X)                                # (n, d+1)
        w = self.weight.ravel()                                # (d+1,)
        return sigmoid(X_b @ w)                                # (n,)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        return (self.predict_probability(X) >= 0.5).astype(np.int64)

    def update_local(
        self,
        feature_matrix: NDArray[np.float64],
        binary_targets: NDArray[np.float64],
        epochs: int = 1,
        learning_rate: float = 0.3,
    ) -> NDArray[np.float64]:
        X = np.asarray(feature_matrix, dtype=np.float64)
        y = np.asarray(binary_targets, dtype=np.float64).ravel()   # (n,)

        X_b = self._add_bias(X)                                    # (n, d+1)
        w0 = self.get_model_weight().ravel()                        # (d+1,)
        if w0.size != X_b.shape[1]:
            raise ValueError(f"weight size {w0.size} != features+1 {X_b.shape[1]}")

        w = w0.copy()
        n = X_b.shape[0]

        for _ in range(epochs):
            p = sigmoid(X_b @ w)                                    # (n,)
            grad = (X_b.T @ (p - y)) / n                            # (d+1,)
            w -= learning_rate * grad

        delta = w - w0
        self.weight = w
        return delta

