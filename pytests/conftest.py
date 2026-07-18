"""Pytest configuration for the astronomix test / benchmark suite."""


def pytest_addoption(parser):
    parser.addoption(
        "--reproduce-paper",
        action="store_true",
        default=False,
        help="regenerate every methods-paper figure from the examples/ "
             "generators (see pytests/test_reproduce_paper.py; slow, needs a GPU).",
    )
