# Contributing to Beholdd

Follow [development setup](docs/development.md) and run hardware-free checks for
the crates and tools you change. Keep CPU reference behavior and backend support
claims distinct. Live GPU and real-checkpoint tests require explicit fixture
selection and separate bounded qualification; they are not ordinary unit tests.

## Pull requests

Keep changes focused, explain the intended behavior, and add a regression test
for fixes. Update the public documentation when setup or observable behavior
changes. Report the exact checks run and any skipped or blocked checks; do not
claim hardware or model qualification from compilation alone.

Do not include private configuration, credentials, machine paths, internal
research, chat exports, memory databases or raw diagnostic bundles. Review the
[publication checklist](docs/publication.md), including history and packaged
assets. Report vulnerabilities through [SECURITY.md](SECURITY.md).

First-party contributions use the repository's [Apache-2.0 license](LICENSE).
Preserve third-party notices and identify the source and redistribution terms of
new dependencies, datasets and other assets. Model weights and user data are not
automatically covered by the source-code license.

Execute Software and community contributors are recognized in
[CONTRIBUTORS.md](CONTRIBUTORS.md). You may add your preferred name to the
community section when submitting a contribution.

Release identity, signing, publication, and repository visibility are maintainer
operations. A source cleanup or passing local test does not publish a release.
