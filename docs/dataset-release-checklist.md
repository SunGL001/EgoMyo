# EgoMyo dataset release checklist

Use this checklist before uploading any dataset file to the Hugging Face
repository. A valid download link or working training loader does not imply
permission to publish the underlying recordings or annotations.

## Release authorization and privacy

- [ ] Confirm that every participant's consent explicitly permits the planned
  public or gated research release and the selected hosting platform.
- [ ] Obtain the required institutional, data-owner and collaborator approvals;
  record the approval/version and the person responsible for the release.
- [ ] Review the actual release-eligibility fields in every package manifest.
  Do not publish any package whose release/acceptance flag is false, missing or
  unresolved.
- [ ] Check video and derived files for faces, voices, names, screens, badges,
  room/location clues, timestamps and other direct or indirect identifiers.
- [ ] Replace participant/session/trial names with stable, non-identifying IDs;
  keep any re-identification key off the Hub and out of GitHub.
- [ ] Confirm rights and redistribution terms for camera footage, annotations,
  sensor streams, calibration assets and any third-party materials.
- [ ] Select and approve a dataset license. Do not guess a license or copy one
  from the code repository; if terms are not settled, keep the dataset private
  or gated and omit a license claim until approval.
- [ ] Decide whether public, gated-access or private hosting is appropriate and
  document who can request access and how requests are reviewed.

## Package and schema

- [ ] Publish a versioned, immutable release with a manifest, schema version,
  file inventory, sizes and SHA-256 checksums.
- [ ] Document units, coordinate frames, sampling rates, synchronization
  assumptions, missing-value conventions and validity masks.
- [ ] Describe every annotation, including how contact/force values were
  derived, their calibration status and known limitations.
- [ ] Include only de-identified split membership and intended train/validation/
  test protocol definitions; ensure no participant leakage across protocols.
- [ ] Keep licensed MANO/WiLoR/DINO weights, credentials, raw internal notes,
  temporary caches, checkpoints and unrelated project files out of the dataset
  repository unless their redistribution is explicitly permitted.
- [ ] Ensure the released model-ready export follows the layout in the code
  README, or publish a documented conversion step from the released source
  package to that layout.
- [ ] Run `python scripts/validate_dataset.py --data <export> --visual wilor`
  on the exact model-ready export and preserve the validation report.
- [ ] Independently inspect representative files and verify the file manifest
  and checksums after upload before announcing the release.

## Documentation and handoff

- [ ] Complete `docs/huggingface-dataset-card-template.md`; remove every TODO
  and verify all counts and claims against the approved release manifest.
- [ ] Include a data statement, collection and annotation methodology, schema,
  intended uses, out-of-scope uses, limitations, privacy notes, license and
  contact for access or corrections.
- [ ] Link the code repository, project website and paper. Add the arXiv URL
  only after the preprint exists.
- [ ] Record the Hub repository ID, immutable commit/revision, release date,
  package version and matching code commit in the release notes.
- [ ] Update the project website only after the release is approved and the
  public/gated access behavior has been tested from a logged-out browser.

## Current status

This file is a release checklist, not a release approval. The source package
must pass its explicit release-eligibility checks before any dataset upload.
The code and documentation can be published independently of the data.
