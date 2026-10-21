# Release validation

The public repository does not schedule CI jobs or publish packages automatically.
Validation runs on private runners against a pinned public baseline and an exact,
reviewed candidate patch. The private checkout is not exported into the candidate.

A release requires:

1. Source-tree and per-file hashes matching the reviewed candidate.
2. Passing standalone installed-wheel tests on Python 3.11 and 3.12, plus Rust tests.
3. Distribution-content checks against `release-manifest.json`.
4. Private validation for every wheel platform approved for that release.
5. Final review of the public diff and distribution hashes before publication.

Validation artifacts are not automatically uploaded to package indexes. A missing
private platform builder keeps that platform's release validation incomplete.
