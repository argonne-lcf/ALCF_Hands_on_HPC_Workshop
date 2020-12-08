"""
problem.py
"""
import ConfigSpace as cs

from deephyper.problem import HpProblem

Problem = HpProblem()

# call signature: Problem.add_dim(name, value)
Problem.add_dim('units1', (1, 64))            # int in range 1-64
Problem.add_dim('units2', (1, 64))            # int in range 1-64
Problem.add_dim('dropout1', (0.0, 1.0))       # float in range 0-1
Problem.add_dim('dropout2', (0.0, 1.0))  	  # float in range 0-1
Problem.add_dim('batch_size', (5, 500)) 	  # int in range 5-500
Problem.add_dim('learning_rate', (0.0, 1.0))  # float in range 0-1

# one of ['relu', ..., ]
Problem.add_dim('activation', ['relu', 'elu', 'selu', 'tanh'])

optimizer = Problem.add_dim('optimizer', [
    'Adam', 'RMSprop', 'SGD', 'Nadam', 'Adagrad'
])

# Only vary momentum if optimizer is SGD
momentum = Problem.add_dim("momentum", (0.5, 0.9))
Problem.add_condition(cs.EqualsCondition(momentum, optimizer, "SGD"))

# Add a starting point to try first
Problem.add_starting_point(
    units1=16,
    units2=32,
    dropout1=0.0,
    dropout2=0.0,
    batch_size=16,
    activation='relu',
    optimizer='SGD',
    learning_rate=0.001,
    momentum=0.5,
)


if __name__ == "__main__":
    print(Problem)
