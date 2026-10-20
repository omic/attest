# Release notes

## 0.2.1

- Avoid rewriting unchanged embedding indexes; persist caller-supplied embedding model labels.
- Mark canonical `interacts_with` relationships as symmetric, including claims ingested using `interacts`.
- Declare the HTTP and fuzzy-matching dependencies used by the public API.
- Correct package versions and Apache-2.0 metadata; include the license in native distributions.
- Provide a native source distribution for Python environments without a matching wheel.
- Remove calibration hooks that depended on unavailable private modules, their integration tests, and unused evaluation generators.
- Require standalone tests, all platform builds, and reviewed artifact contents before publication.

The release starts from the public 0.2.0 tree. It contains no newly exported private implementation. Optional integrations already present in that public tree may still require separately installed components.
