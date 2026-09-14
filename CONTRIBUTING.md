# Contributing to nowcast-midas

## 1. Set up your fork

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-github-user>/nowcast-midas.git
cd nowcast-midas
```

Git names your fork `origin`; this command names the main repository `upstream`, so you can fetch updates from it:

```bash
git remote add upstream https://github.com/bank-of-england/nowcast-midas.git
```

Install the package with dev dependencies in editable mode:
```bash
pip install -e ".[dev]"
```

Install the pre-commit hooks:
```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

The hooks make sure everything is OK:

- Conventional Commit message checks for Release Please
- Ruff linting, with automatic fixes where possible
- Ruff formatting
- API documentation generation in `docs/api.md`
- NumPy-style docstring checks with `pydoclint`
- A Zensical documentation build
- The full pytest suite

To run the same checks without making a commit:

```bash
pre-commit run --all-files
```

When the public API changes, update their exports and docstrings too; that part cannot be done automatically for you. What the hooks do is generating the doc and the `docs/api.md` from the export and docstrings.

## 2. Code

The `main` branch holds released code and accepts changes only through pull requests. Start each change from the main repository's `dev` branch, then open a pull request from your fork back to `dev`. Changes can accumulate there until the maintainers are ready to release a new version of the package.

```text
issue -> fork -> branch from upstream/dev -> develop and test -> push to fork -> PR to dev -> automatic checks and review -> merge to dev
```

Branch names such as `feature/<issue>`, `fix/<issue>`, and `docs/<topic>` make the work easy to spot. For commit subjects, the project uses [Conventional Commits](https://www.conventionalcommits.org/): `fix:`, `feat:`,
`deps:`, `docs:`, and `chore:` are the usual choices. Release Please later turns these into the changelog, so it's important to follow this approach.

Here is a typical start:

```bash
git fetch upstream
git switch -c fix/123-short-description upstream/dev
```

When the change feels ready, add or update its tests and run:

```bash
pre-commit run --all-files
```

After committing the changes, push the branch to your fork:

```bash
git push -u origin fix/123-short-description
```

Then open a pull request to the `dev` branch of the main repository.

## 3. Submit a pull request

Opening a PR to `dev` starts two workflows:

* package-quality: essentially the pre-commit hooks plus checking that the package build is clean.
* ecosystem: Checks that the code changes do not break compatibility with the OPERA ecosystem packages.

Package-quality must pass before your branch can be merged to dev but the ecosystem check is optional.

## For maintainers:
### 4. Release a version (for maintainers)

When the changes in `dev` are ready to ship, a maintainer opens a pull request from `dev` to `main`. This triggers the package-quality and ecosystem workflows again.

Once merged, Release Please will wake. It will read the new commits (that's why it's important to start commits with "fix:" "feat:" etc), updates the version and `CHANGELOG.md`, and opens a second pull request. Release Please is set up to increment +0.0.1 to the version number; if you want something else write the following in the PR message:
`Release-As: x.y.z`.

```text
A contrib PR -> package-quality + ecosystem + review -> merge -> Release Please wakes up -> new PR with version + changelog updated -> package-quality -> auto-merge if package-quality is green -> New GitHub Tag Release
```

The publication of the new GitHub Release starts two other workflows:
* (1) publish-pypi: Publish the package version to PyPI.
* (2) Doc deployment: Deploys the updated documentation.

### It's not finished!

If the publication to PyPI is successful, the publish-pypi workflow will trigger yet another workflow:
* update-ecosystem: PR to the main in the opera-eco repo with the new package version, the opera-eco checks are successful the PR merges automatically and Release Please will publish a new version of the opera-eco package.

### The whole workflow

```text
------ Installation
-> fork the repository
-> clone your fork
-> add the main repository as upstream
-> pip install -e ".[dev]"
-> pre-commit install

------ Development
-> branch from upstream/dev
-> develop and test

------ PR to dev
-> push the branch to your fork
-> open a PR from your fork to dev
-> package-quality and ecosystem run
-> review and merge to dev
-> repeat until dev is ready to release

------ PR to main
-> open a PR from dev to main
-> package-quality, ecosystem, and review
-> merge to main

------ Everything is automatic from here
-> Release Please opens a version and changelog PR
-> package-quality passes
-> Release Please PR auto-merges
-> GitHub creates the version tag and release
-> documentation deploys to GitHub Pages
-> publish-pypi publishes the package
-> update-ecosystem starts
-> update-ecosystem opens a PR in opera-eco
-> opera-eco runs its own package-quality workflow
-> auto-merge waits for that workflow
-> opera-eco updates its nowcast-midas pin
```

### Workflow summaries

- **Package quality** checks pull requests to `main` or `dev` by building the package, checking the code and documentation, and running the tests.
- **Ecosystem** runs the OPERA ecosystem contract and pipeline tests for pull requests to `main` or `dev`, except Release Please pull requests.
- **Release Please** runs after changes reach `main`, then opens or updates the release pull request and enables auto-merge when appropriate.
- **Publish to PyPI** builds and publishes the package after a release, then starts the ecosystem pin update.
- **Deploy documentation** builds and deploys the documentation to GitHub Pages after a release.
- **Update opera-eco pin** updates the pinned package version and generated APIs, then opens or updates an auto-merged pull request in `opera-eco`.
