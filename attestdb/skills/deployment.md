# Deployment and Publishing

## PyPI Publishing

**Package name:** `attest_py` (underscore). PyPI normalizes hyphens to underscores. The trusted publisher must be registered as `attest_py`, not `attest-py`. Workflow name, owner, repo, and environment must all match exactly.

**Version files** (all three must match on release):
- `oss/pyproject.toml`
- `rust/attest-py/pyproject.toml`
- `rust/attest-py/Cargo.toml`

**Trigger:** `gh release create v0.X.Y` → GitHub Actions builds 6 wheel targets + sdist.

## AWS / Ubuntu

Ubuntu 24.04 does NOT have `awscli` in apt. Install AWS CLI v2 from the official zip:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip
cd /tmp && unzip awscliv2.zip && sudo ./aws/install
```

## Stripe Integration

Stripe API `2025-03-31.basil` requires **Meter objects** for metered prices:

1. Create a Meter via `stripe.billing.Meter.create(event_name=..., aggregation=...)`
2. Create a Price backed by that Meter via `stripe.Price.create(metered usage_type, meter=meter_id)`
3. Graduated tiers must be configured via the `tiers` parameter or in Stripe Dashboard

Flat-rate tier prices are separate from metered prices — each plan needs both.

## Infrastructure

- **Static site:** `docs/site/` → scp to EC2 `/var/www/attestdb/`
- **API:** Terraform in `enterprise/infra/`, t4g.small ARM, us-west-2
- **Userdata changes:** `terraform taint aws_instance.app` then apply (IP changes!)
- `attestdb-enterprise` needs `uvicorn` + `pydantic[email]` explicitly installed

## Security Groups

Two separate SGs — don't mix them up. The static website and the API instance use different security groups. SSH whitelist must target the correct SG for whichever server you're accessing. Check CLAUDE.md for the actual SG IDs.
