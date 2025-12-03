import pytest


def pytest_addoption(parser):
    """
    Add a command-line option to enable real LLM API tests.

    Usage:
        pytest --run-real-api
    """
    parser.addoption(
        "--run-real-api",
        action="store_true",
        default=False,
        help="Run tests that call real LLM APIs (OpenAI / Google).",
    )


@pytest.fixture
def run_real_api(request):
    """Return True if --run-real-api was passed to pytest."""
    return request.config.getoption("--run-real-api")
