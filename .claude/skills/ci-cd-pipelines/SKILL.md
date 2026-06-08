---
name: ci-cd-pipelines
description: Design and optimize continuous integration and deployment pipelines (e.g., GitHub Actions, GitLab CI) for speed, security, and reliability.
---

**When to Use**
- Creating or modifying `.github/workflows/` or `.gitlab-ci.yml` files.
- Debugging pipeline failures or slow build times.
- Implementing security scanning or automated releases.

**Core Principles**
1. **Fail Fast:** Run linting and fast unit tests before long-running E2E tests or builds.
2. **Caching:** Aggressively cache dependencies (npm, pip, Docker layers) to reduce pipeline time.
3. **Least Privilege:** Use OIDC (OpenID Connect) for cloud authentication instead of long-lived secrets.
4. **Immutability:** Build artifacts once (e.g., Docker image) and promote the *exact same artifact* through staging to production.
5. **Matrix Builds:** Use matrix strategies to test across multiple OS/Node/Python versions efficiently.

**Workflow**
1. Define triggers (push to main, PRs, tags).
2. Set up environment (checkout, cache, install dependencies).
3. Run quality gates (lint, type-check, unit test).
4. Build and push artifacts (with SBOM/security scanning).
5. Deploy to target environment with approval gates if production.