# Security

Use [private vulnerability reporting](https://github.com/exec-software/exec-beholdd/security/advisories/new)
when it is available. If it is unavailable, request a private reporting channel in
an issue without disclosing exploit details, credentials, model inputs or logs.
Include the affected commit, configuration, platform and a minimal safe reproducer.
There is no guaranteed response-time SLA.

## Deployment boundaries

Treat model files, templates and network requests as untrusted. Use trusted model
sources, verify artifact identities, keep limits enabled, and run the service as
an unprivileged account. Do not assume the inference API is an authentication or
tenant-isolation boundary; restrict access and add an authenticated gateway for
network deployments. Do not publish development configurations or private models.

Native GPU execution uses the caller's device permissions and trusted kernels.
Driver faults can affect other processes. Experimental probes must run separately
with bounded memory and execution time; ordinary automated tests must not launch
GPU work or download model weights. Do not attach a working GPU crash trigger to
a public issue.

Keep payload tracing and diagnostic captures private. Review dependency advisories,
model redistribution terms and [the publication checklist](docs/publication.md)
before distributing source or binaries. Rotate any committed credential even if
the repository was private; a source edit does not remove earlier copies.
