# Immutable GHCR builder for the Glyph alpha-0.8 candidate

This repository builds the SN117 candidate on a disposable GitHub-hosted runner
and pushes it to GHCR with the workflow repository's short-lived `GITHUB_TOKEN`.
No GHCR credential is needed on the operator's host.

## Bound inputs

- Base image: `mongo1001/glyph-rwkv-mem-v2@sha256:15518bf00777d22337b3e478407239fa37a0ad459644f455294d00af72d0338d`
- Model repository: `putty77/glyph-rwkv-alpha0p8-build-source`
- Model revision: `b50b12b2fb937387180752797dc41ff0119828c7`
- Model file: `selected-model.pth`
- Model SHA-256: `d59ff9c80e8a2eba56eeac38343bad5d90e9d439a3fc699dbdef53af0516929e`
- Codec SHA-256: `ca9f652520acb2f71bfadf2aa182dd9775eb8c69fe420f93f7ff36d21db4ca64`

The model URL is public and commit-pinned. The workflow downloads it without an
HF token and refuses a revision that is not a 40-character commit or a model
whose size/SHA differs.

The image adds Debian's minimal `openssh-client` package solely so a Vast RTX
4090 validation instance can establish its SSH tunnel. Validators sever image
network access before scored compression/decompression, and the codec/model
files remain independently SHA-bound.

## Publish procedure

1. Put these files in the public GitHub repository
   `luckydevstar/glyph-rwkv-alpha0p8-hier256-b96` and review the commit.
2. From the repository's default branch, manually run **Publish immutable Glyph
   image**. The job is explicitly denied on any other repository or branch.
3. Download the `glyph-image-<source SHA>` artifact. It contains `image.json`
   plus the submission `manifest.json`, both bound to the digest returned by
   `docker/build-push-action`.
4. On the first publication only, open the new package's settings and change its
   visibility to **Public**. New GHCR packages default to private. Making one
   public is irreversible, so check the package name first.
5. Run **Verify anonymous GHCR digest pull** and paste the captured
   `sha256:...` digest. This runs on a fresh VM, grants no `packages` permission,
   creates an empty Docker credential store, and pulls the exact digest.
6. Use only `ghcr.io/luckydevstar/glyph-rwkv-alpha0p8-hier256-b96@sha256:...` in the
   final Glyph manifest. A tag such as `sha-<source commit>` is only a label.

The publish workflow has only `contents: read` and `packages: write`. All
third-party and GitHub actions are pinned to full commit SHAs. The OCI source
label links the package to this repository so its `GITHUB_TOKEN` receives the
correct package relationship.

## Determinism

The base, model, codec and action implementations are immutable. The workflow
builds only `linux/amd64`, disables cache, SBOM and provenance index wrappers,
and passes the source commit time as `SOURCE_DATE_EPOCH`. Re-running the same
source commit should therefore reproduce the image digest. The digest returned
by GHCR remains the authority: compare repeat-run outputs before submitting.

GitHub's standard runner exposes limited free disk while this build temporarily
holds the 3.06 GB model more than once. `free-disk-space.sh` removes unused
Android/.NET/Haskell/Boost/CodeQL SDKs and refuses to build unless 20 GiB is
free. If GitHub changes its runner image and cleanup no longer reaches that
threshold, select a larger Ubuntu runner; do not lower the safety threshold.

## Independent host check

After the package is public, any Docker host can prove that validators need no
credential:

```bash
scripts/verify-anonymous-pull.sh \
  ghcr.io/luckydevstar/glyph-rwkv-alpha0p8-hier256-b96 \
  sha256:REPLACE_WITH_CAPTURED_DIGEST
```

The script points `DOCKER_CONFIG` at a newly-created empty config, pulls by
digest, and verifies the resulting `RepoDigest`. Run the normal Glyph precheck
and RTX 4090 DockerRunner lifecycle against the same digest before committing a
hotkey.

## One-time blocker

`GITHUB_TOKEN` can publish the repository-associated package but this draft does
not attempt to change visibility through an API. The package owner must make the
first GHCR version public in GitHub's package settings before anonymous pull can
pass. No repository has been created, no workflow has been run, and no image has
been published by this draft.
