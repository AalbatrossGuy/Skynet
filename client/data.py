import numpy


def generate_dataset_local(
        n: int,
        data: int,
        seed: int | None,
        prevalence: float = 0.12
):
    generate_random_seed = numpy.random.RandomState(seed)
    X_matrix = generate_random_seed.normal(0, 1, size=(n, data))
    weight = generate_random_seed.normal(0, 0.7, size=(data, ))
    log_value = X_matrix @ weight
    y: int = (generate_random_seed.rand(n) < prevalence * 1 / (1 + numpy.exp(-log_value))).astype(int)

    for _ in range(int(n * 0.02)):
        index_1, index_2 = generate_random_seed.randint(0, n), generate_random_seed.randint(0, data)
        X_matrix[index_1, index_2] += generate_random_seed.uniform(3, 6)
        y[index_1] = 1

    return X_matrix, y
