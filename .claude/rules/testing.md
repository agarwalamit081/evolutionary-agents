# Testing Standards and Requirements

## Test Coverage
- ALL new features MUST include unit tests. Target > 80% code coverage.
- NEVER comment out, delete, or bypass failing unit tests. Fix the application code to pass the tests securely.
- ALL tests MUST be deterministic. Use fixed seeds, mocked responses, and controlled test data.
- NEVER skip tests with `@pytest.mark.skip`, `test.skip()`, or `xit`/`xdescribe` without a documented reason and timeline.

## Test Quality
- Tests MUST be independent and idempotent. No test should depend on another test's execution order or side effects.
- ALWAYS use descriptive test names that explain the expected behavior: `test_should_return_404_when_user_not_found` not `test_get_user`.
- NEVER use exact-string matching for AI-generated content in tests. Use semantic assertions or fuzzy matching.
- ALWAYS clean up test fixtures, temporary files, and database state after tests run.

## When Changing Code
- When changing a function signature, you MUST find and update ALL call sites AND all related tests simultaneously.
- When adding new functionality, ALWAYS add corresponding tests.
- When fixing a bug, ALWAYS add a regression test that would have caught the original bug.
- Run the full test suite before committing code. Do not commit with failing tests.

## Test Organization
- Mirror the source directory structure in the test directory.
- Group related tests into test classes or describe blocks.
- Use factories or fixtures for test data creation, not hardcoded values.

## Mocking and External Dependencies
- ALWAYS mock external API calls in unit tests. Never make real HTTP requests in unit tests.
- For AI/LLM-related tests, implement deterministic mocking of LLM responses for CI/CD environments to prevent flaky tests.
- Reserve live LLM calls for dedicated staging evaluation environments only.
- NEVER mock the system under test. Only mock dependencies and external services.

## Async Testing
- Use `pytest-asyncio` with `@pytest.mark.asyncio` for all async function tests.
- Create async fixtures with `@pytest_asyncio.fixture` for database sessions, HTTP clients, and other async resources.
- ALWAYS use `async with` for async context managers in tests — never forget `await`.

## FastAPI Testing
- Use `TestClient` from `fastapi.testclient` for synchronous route testing.
- Use `httpx.AsyncClient` with `ASGITransport` for async route testing.
- ALWAYS override dependency injection in tests — never connect to production databases or external services.

## LLM Evaluation Testing
- Use `ragas` for RAG evaluation metrics: faithfulness, answer relevancy, context precision, context recall.
- Use `deepeval` for unit-test-style LLM evaluations: answer similarity, bias detection, toxicity checks.
- Pin evaluation model versions and use fixed random seeds for reproducible eval runs.
- Define quantitative pass/fail thresholds for every evaluation metric.

## Pre-Commit Test Requirements
- ALL tests must pass before code is committed.
- Run `uv run python -m pytest` (Python) or `npm test` (JS/TS) before every commit.
- NEVER use bare `python -m pytest` or `pytest`. ALWAYS use `uv run python -m pytest` to ensure the correct virtual environment.
- If a test fails after a code change, the code change is wrong, not the test.
