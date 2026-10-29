# AutoCompany onboarding

AutoCompany helps an existing agent client connect a company's first Fabric workflow,
prepare useful work and report what still needs attention. The shared kit is maintained
in `attestdb/onboarding/skill/`, alongside its version and file-hash manifest. OSS and
Enterprise consume these same resources; private services and tenant data stay separate.

## Export the installed kit

With a package built from this source, create a new ZIP or skill folder:

```sh
python -m attestdb.onboarding --zip autocompany-setup.zip
python -m attestdb.onboarding --directory /path/to/new/autocompany-setup
```

Use either command for the format your current client accepts. Both refuse existing
destinations. The folder includes `SKILL.md`, the Fabric catalog, the workspace creator
and `kit-manifest.json`. The ZIP also records its edition and manifest digest in
`distribution.json`. Installation does not authenticate accounts or enable scheduling.

Give the client this prompt:

> Use the AutoCompany setup skill to help me connect my company and prepare the
> first useful workflow. Reuse my existing accounts, verify what is connected,
> and show the first artifact and remaining gaps.

The workspace creator records the kit version, marks connections `not_checked` and
starts automation paused. It preserves an existing manifest and refuses changed
configuration. Service access and workflow completion require separate observations.

## Enterprise distribution

The Enterprise service's `/connect` page links `/autocompany-skill.zip`. That download
uses the shared installed kit and adds `references/enterprise.md` for organization
identity, approved services, human decisions and operational qualification. Its
`distribution.json` records the same shared-kit version plus the extension hash.
Use matching OSS/core and Enterprise builds; the service does not carry a second copy
of the catalog. The generic extension contains no tenant routes or credentials.

## Maintenance and public release

Edit the shared resources here, bump the kit version and refresh their hashes in
`kit-manifest.json`. Run the workspace/export tests and Enterprise download tests.
The kit version is independent of the Python distribution version.

The public candidate includes the shared module and a portable copy under
`skills/autocompany-setup/`. `scripts/stage_oss_candidate.py` verifies the reviewed
candidate tree, then checks that both copies match the maintained kit exactly.
Regenerate the reviewed candidate patch, manifest and pins when the kit changes;
do not weaken tree verification or copy private Enterprise files into public releases.

Public examples remain synthetic references. Real engineering investigations,
compliance review workbenches and sales workflows have their own integration,
authorization and operating requirements. Their existence does not qualify an
offline reference as a supported hosted service.
