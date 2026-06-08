---
description: CI/CD Pipelines Examples
---

**Example 1: Node.js/TypeScript CI/CD Pipeline**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, 'release/*']
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20', cache: 'npm' }
      - run: npm ci
      - run: npm run lint && npm run type-check
      - run: npm run test:unit
        env: { CI: true }

  build-and-push:
    needs: quality-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions: { contents: read, packages: write }
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: '${{ github.actor }}', password: '${{ secrets.GITHUB_TOKEN }}' }
      - uses: docker/build-push-action@v5
        with: { context: '.', push: true, tags: 'ghcr.io/${{ github.repository }}:sha-${{ github.sha }}', cache-from: 'type=gha', cache-to: 'type=gha,mode=max' }
```

---

**Example 2: Database Migration with Environment Protection**

```yaml
  migrate-db:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - run: npx prisma migrate deploy
        env: { DATABASE_URL: '${{ secrets.STAGING_DATABASE_URL }}' }
```

---

**Example 3: Python CI with Poetry and Matrix Testing**

```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12', '3.13']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '${{ matrix.python-version }}' }
      - run: pip install poetry && poetry install
      - run: poetry run ruff check .
      - run: poetry run mypy src/
      - run: poetry run pytest --cov=src --cov-report=xml -q
      - uses: codecov/codecov-action@v4
        with: { token: '${{ secrets.CODECOV_TOKEN }}' }
```

---

**Example 4: Security Scanning Step**

```yaml
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy container scan
        uses: aquasecurity/trivy-action@master
        with: { image-ref: 'ghcr.io/${{ github.repository }}:latest', severity: 'CRITICAL,HIGH', exit-code: '1' }

      - name: Run npm audit
        run: npm audit --audit-level=high

      - name: Run Snyk test
        uses: snyk/actions/node@master
        env: { SNYK_TOKEN: '${{ secrets.SNYK_TOKEN }}' }
```

---

**Example 5: Blue-Green Deployment with Approval Gate**

```yaml
  deploy-production:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://myapp.example.com
    steps:
      - name: Deploy to green environment
        run: |
          kubectl set image deployment/myapp myapp=ghcr.io/repo:sha-${{ github.sha }} --namespace=green
          kubectl rollout status deployment/myapp --namespace=green --timeout=120s

      - name: Run smoke tests against green
        run: npm run test:smoke -- --base-url=https://green.myapp.example.com

      - name: Switch traffic to green
        run: kubectl patch service/myapp -p '{"spec":{"selector":{"version":"green"}}}'

      - name: Rollback on failure
        if: failure()
        run: kubectl patch service/myapp -p '{"spec":{"selector":{"version":"blue"}}}'
```

---

**Example 6: Monorepo CI with Path-Based Triggers**

```yaml
name: Monorepo CI

on: [push, pull_request]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      api: '${{ steps.filter.outputs.api }}'
      web: '${{ steps.filter.outputs.web }}'
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            api:
              - 'services/api/**'
            web:
              - 'apps/web/**'

  test-api:
    needs: detect-changes
    if: needs.detect-changes.outputs.api == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd services/api && npm ci && npm test

  test-web:
    needs: detect-changes
    if: needs.detect-changes.outputs.web == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd apps/web && npm ci && npm test
```
