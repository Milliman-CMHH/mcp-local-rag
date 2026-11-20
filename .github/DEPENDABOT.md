# Dependabot updates

Dependabot opens PRs that bump the GitHub Actions (`.github/workflows/`), the
Python dependencies (`pyproject.toml` and `uv.lock`) and the Copilot CLI
(`.github/copilot-cli/`). Three workflows handle them:

| Workflow | What it does |
|---|---|
| `ci.yml` | Runs every check: `uv sync --locked`, ruff, pyright, pytest, and actionlint and zizmor on the workflows. Its `CI` job is the one required status check. |
| `dependabot-auto-merge.yml` | If every commit is Dependabot's own, queues a squash merge for when `CI` passes, then approves the PR. |
| `dependabot-fix.yml` | If `CI` fails on a PR whose commits are all Dependabot's own, Copilot CLI proposes a fix, the merge app pushes it, and the people in `REVIEWERS` are asked to review it. |

A commit is **Dependabot's own** when it is authored by `dependabot[bot]`,
verified, and committed by GitHub (`web-flow`). Any other commit, such as a
Copilot fix, makes the PR wait for a person. A Copilot fix that touches
`.github/workflows/` or `.github/actions/` is never pushed; the patch stays in
the run's artifacts.

This repository is public. Both workflows skip PRs from forks: they act only
on PRs opened by `dependabot[bot]` from a branch in this repository. Neither
gives the PR's code a token that can write to the repository, and the merge
app's key is only available to jobs that run from `main`.

## What happens to a Dependabot PR

- **CI passes:** it merges by itself. The workflow approves it and queues the
  merge; the merge completes when `CI` and the other ruleset checks (CodeQL,
  code quality) pass.
- **CI fails:** Copilot pushes a fix (or explains why it couldn't), adds the
  `dependabot-fix` label and requests review from the people in `REVIEWERS`
  in `dependabot-fix.yml`. Once one approves, the queued merge completes. If
  the fix is wrong, push a better one or close the PR.
- **Dependabot rebases the PR:** the fix commit is dropped (it carries
  `[dependabot skip]`) and everything starts again.

## Settings these workflows need

Repository admins set these. Each is checked by GitHub, not by the workflows.

1. **A GitHub App (the merge app)** in Milliman-CMHH, installed on this
   repository, with these repository permissions:
   - Contents: read and write
   - Pull requests: read and write
   - Workflows: read and write

   Each job asks for a token with only the permissions it needs. The token
   that pushes Copilot fixes has Contents write only, so GitHub rejects a fix
   that changes a workflow.
2. **An environment named `dependabot-automation`** (Settings → Environments),
   with *Deployment branches and tags* set to *Selected branches and tags* and
   a single rule for `main`. Create it before these workflows first run:
   GitHub creates a missing environment with no branch rule. Add the secret
   `MERGE_APP_PRIVATE_KEY` (a private key generated for the app) there, and
   nowhere else: not as a repository or Dependabot secret, and not in the
   `copilot` or `pypi` environments. The app's client ID goes in the
   **repository** variable `MERGE_APP_CLIENT_ID`; it isn't secret.
3. **The `Main Branch Protection` ruleset** requires:
   - the `CI` status check (source: GitHub Actions);
   - one approving review, with *Require approval of the most recent
     reviewable push* turned on. That is what makes a person approve a Copilot
     fix: the merge app pushed it, and the workflow's approval is for an
     earlier commit.

   Squash must stay in its allowed merge methods. Its code quality and CodeQL
   rules also apply to Dependabot PRs.
4. **Settings → Actions → General → Workflow permissions:** turn on *Allow
   GitHub Actions to create and approve pull requests*, so the auto-merge
   workflow can approve.
5. **Settings → General → Pull Requests:** *Allow auto-merge* and *Allow
   squash merging* on.
6. **GitHub Copilot** enabled for the organization, so the fix workflow's
   `copilot-requests` permission works.

Until the app and the environment are set up, the auto-merge and fix
workflows fail at their app-token step, and every Dependabot PR waits for a
person.

## Copilot CLI version

`.github/copilot-cli/package.json` pins `@github/copilot` to an exact version,
and the lockfile pins its dependencies. The fix workflow runs the CLI with a
token that can ask Copilot for changes, so only a version that someone or
Dependabot chose, and CI didn't flag, gets that token. Dependabot proposes a
new version only after it has been out for 7 days (`cooldown` in
`dependabot.yml`), because CI never runs the CLI itself and can't catch a bad
release.
