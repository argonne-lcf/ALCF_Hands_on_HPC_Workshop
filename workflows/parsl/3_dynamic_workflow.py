import parsl
from parsl.app.app import join_app, python_app

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html


@python_app
def add(*args):
    """Add all of the arguments together. If no arguments, then
    zero is returned (the neutral element of +)
    """
    accumulator = 0
    for v in args:
        accumulator += v
    return accumulator


# Here we use a join_app that can launch a sub-workflow
# Join apps return a future object so that the parsl workflow can
# continue to run other tasks while waiting for the sub-workflow
@join_app
def fibonacci(n):
    if n == 0:
        return add()
    elif n == 1:
        return add(1)
    else:
        return add(fibonacci(n - 1), fibonacci(n - 2))


with parsl.load():
    fib_series = fibonacci(10)

    print(fib_series.result())
