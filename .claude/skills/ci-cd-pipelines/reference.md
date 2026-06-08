---
description: CI/CD Pipelines Reference
---

## GitHub Actions Core Concepts

| Concept | Description |
|---|---|
| Workflow | Automated process in `.github/workflows/` |
| Job | Group of steps executed on a runner |
| Step | Individual action (run command or `uses:` action) |
| Matrix | Test across multiple versions/OS combinations |
| Concurrency | Cancel in-progress runs on new commits |
| Environment | Named deployment target with protection rules |

## Security Best Practices

- **Pin actions to SHA**: `uses: actions/checkout@8ade135a41bc03ea155e62e844d188df1ea18608` instead of `@v4`.
- **OIDC over secrets**: Use `aws-actions/configure-aws-credentials` with `role-to-assume` instead of static AK/SK.
- **GITHUB_TOKEN permissions**: Set `permissions: contents: read` at workflow level. Grant write only where needed.
- **Dependabot**: Enable for automated dependency updates.
- **Secret scanning**: Enable in repo settings. Never commit secrets.

## Docker Optimization

- **Multi-stage builds**: Build stage with full SDK, runtime stage with minimal image.
- **Layer caching**: Order Dockerfile from least to most frequently changed layers.
- **GitHub Actions cache**: Use `docker/build-push-action` with `cache-from: type=gha, cache-to: type=gha,mode=max`.
- **Image sizing**: Use `distroless` or `alpine` base images. Never use `latest` tag in production.

## Deployment Strategies

| Strategy | How | Downtime | Rollback |
|---|---|---|---|
| Rolling update | Replace instances gradually | None | Redeploy previous version |
| Blue-green | Two identical environments, switch traffic | None | Switch back to blue |
| Canary | Route small % traffic to new version | None | Shift traffic back |
| Recreate | Stop old, start new | Yes | Redeploy previous |

## Environment Protection Rules

```yaml
deploy-production:
  environment: production
  needs: [test]
  steps:
    - run: echo "Deploying to production"
```

Configuration: Required reviewers (1+), wait timer (5 min), deployment branches (`main` only), environment secrets.

## GitLab CI Quick Reference

| GitHub Actions | GitLab CI |
|---|---|
| `on: push` | `rules: - if: '$CI_PIPELINE_SOURCE == "push"'` |
| `jobs.build.runs-on: ubuntu-latest` | `image: ubuntu:latest` |
| `steps.uses: actions/checkout@v4` | Built-in (automatic) |
| `matrix: node-version: [18, 20]` | `parallel: matrix: NODE_VERSION: [18, 20]` |
| `cache: path: ~/.npm` | `cache: key: npm paths: - ~/.npm` |
| `concurrency: group: ${{ github.ref }}` | `resource_group: $CI_COMMIT_REF` |

## CI Anti-Patterns

- Running everything in one job (no parallelism, slow feedback).
- No caching (dependencies downloaded every run).
- Committing secrets or using long-lived tokens.
- Mutable build artifacts (rebuilding for each environment).
- No fail-fast (running all jobs even when early ones fail).
