# Publication checklist

This checklist is a release gate, not a claim that automated scanning proves a
repository free of sensitive information.

## Source and history

Install [Gitleaks 8.30.1](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1)
from its upstream release and verify the published checksum. From the checkout:

```sh
python3 tools/audit_secrets.py --scanner /path/to/gitleaks \
  --report-dir /path/to/private/new-audit-directory
```

The output directory must not already exist and must be outside the checkout.
The script snapshots tracked and non-ignored untracked files without following
symlinks, hashes every input, scans archives to depth three, and scans all local
Git refs with full history and the staged index separately. Tracked deletions are
recorded as omissions from the working snapshot; history is still scanned. It
fails on findings, incomplete scanning, unsupported scanner versions, and
non-regular source entries. It does not fetch or publish.
Reports are private, redacted, and never uploaded by CI. Do not commit them.

Inspect remote branches, tags, pull-request refs, release assets, Git LFS objects,
submodules, wiki content and package artifacts separately. A local scan cannot
cover objects it does not have. Never use a mirror push or make a private
repository public merely because the default branch passes. Removing content in
a new commit does not remove previous versions. Rotate exposed credentials;
history cleanup and remote visibility changes require maintainer approval.

The scanner configuration extends upstream defaults. Any false-positive exception
must identify a reviewed literal and its file/rule; do not exempt entire notebooks,
test directories, commit histories, or all high-entropy values.

## Standalone documentation and research

Review the README, contribution guide, notebooks, diagrams, screenshots, source
comments, fixtures, generated files and compressed evidence manually. Remove
private project plans, unpublished cross-project research, personal data, private
hostnames, workstation paths and internal network topology. Preserve originals in
private storage before replacing project notes with standalone explanations.
Describe historical measurements honestly; a rewritten summary is not a new
benchmark or a requalification of a changed binary.

Use repository-relative links, user-selected paths and documented configuration.
A clean checkout must not require a private parent workspace, credentials,
unpublished sibling repository or a maintainer's model cache.

## Licensing and artifacts

Review first-party licensing and retain all third-party copyright and license
notices. Audit vendored source, fonts, models, training/curriculum data, generated
content and binary dependencies independently; the project's license does not
grant rights to unrelated assets. Keep private corpora, signing keys, pairing
state, sessions, dumps and service configuration out of source and packages.

Run the documented hardware-free tests and inspect the actual source/wheel/npm/
application archives. A source scan alone does not qualify an installer, GPU
backend, dependency advisory status, or model download for distribution.
