"""
problem.py
"""
from deephyper.problem import HpProblem

Problem = HpProblem()
Problem.add_dim('epochs', (5, 500))
Problem.add_dim('units1', (1, 64))
Problem.add_dim('units2', (1, 64))
Problem.add_dim('dropout1', (0.0, 1.0))
Problem.add_dim('dropout2', (0.0, 1.0))
Problem.add_dim('batch_size', (5, 500))
Problem.add_dim('activation', ['relu', 'elu', 'selu', 'tanh'])
Problem.add_dim('optimizer', ['Adam', 'RMSprop', 'SGD', 'Nadam', 'Adagrad'])

Problem.add_starting_point(
    epochs=5,
    units1=1,
    units2=2,
    dropout1=0.0,
    dropout2=0.0,
    batch_size=8,
    activation='relu',
    optimizer='SGD',
)


if __name__ == "__main__":
    print(Problem)
