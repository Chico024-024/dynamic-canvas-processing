# Licensing and Commercial Considerations

> This document records engineering context and open questions. It is not legal
> advice. Licensing, third-party rights, royalty definitions, and commercial
> agreements require review by qualified counsel before launch.

## Current status

- **No license has been selected or granted.**
- The repository is a pre-release prototype.
- `LICENSE.md` is a placeholder; default copyright applies until an approved
  license is added.
- DCP Core has not been published, and no commercial terms have been offered.
- Nothing in this repository should be read as permission to redistribute,
  modify, or commercially exploit the code.

Selecting a license, publishing the project, accepting contributions under a
contribution policy, and entering commercial agreements require the project
owner's explicit authorization and legal review.

## The intended split

- **DCP Core**: a public, permissively licensed foundation that is useful on its
  own.
- **DCP Advanced**: proprietary code in a separate repository and distribution,
  licensed commercially.

Core exposes stable public interfaces (`TransitionContext`, `TransitionPlan`,
`StemSet`, `TrackAnalysis`, `TransitionStrategy`) and a local plugin entry-point
group. Advanced must remain an optional package. Core has no dependency on it,
and its absence, failure, or broken plugin cannot disable Core rendering.

Do not copy proprietary Advanced source into Core. Do not copy Core code into
Advanced except as allowed by the eventual Core license and applicable notices.
The public/private boundary must be maintained in source control and release
artifacts.

## Open-source options for Core

### MIT

Advantages:

- short and widely understood;
- permits commercial use, modification, sublicensing, and distribution;
- requires preservation of the copyright and permission notice;
- easy for downstream products to adopt.

Trade-offs:

- no express patent grant (though patent rights are not automatically waived);
- minimal naming/trademark guidance;
- less formal contribution and notice structure.

### Apache License 2.0

Advantages:

- permissive commercial use with an express patent grant and patent-termination
  clause;
- explicit trademark carve-out and clearer contribution terms through the
  license;
- familiar to many enterprise legal teams;
- compatible with proprietary separate modules when their own terms and Core's
  notices are respected.

Trade-offs:

- longer and more formal than MIT;
- requires including the license and preserving notices/NOTICE content where
  applicable;
- still requires review of how a separately distributed proprietary Advanced
  package is packaged and linked.

### Recommendation to discuss, not apply

Apache-2.0 is a reasonable starting recommendation if patent clarity and
enterprise adoption matter, while MIT may be preferable if maximum simplicity and
minimum compliance friction are the priority. Both are permissive and can
coexist with a separately licensed proprietary Advanced edition, subject to
counsel's review of distribution and notice obligations.

GPL-family licenses are not recommended for a Core intended to be embedded in
proprietary commercial products without a deliberate copyleft strategy and
legal review.

**No option above has been selected. Do not add a license file or publish based
on this document without authorization.**

## Third-party dependencies

The prototype installs:

| Component | Version used | Typical license / consideration |
| --- | --- | --- |
| Python | 3.12.15 | PSF license; not bundled with this repository. |
| NumPy | 2.5.4 | BSD-3-Clause. |
| SciPy | 1.18.1 | BSD-3-Clause. |
| SoundFile (python-soundfile) | 0.14.0 | BSD-3-Clause. |
| libsndfile | 1.2.2 (system binary) | LGPL-2.1-or-later; dynamic linking, notices, and relinking/source-offer obligations need review for distribution. |
| pytest | 9.2.0 | MIT; development only. |
| setuptools | build dependency | MIT; build tooling only. |

Version and license facts should be rechecked against the exact artifacts that
will be distributed. A permissive Python package does not automatically settle
the terms of the native library it loads. Redistributing a bundled libsndfile
binary may trigger LGPL obligations that differ from using a system-installed
library.

### Optional analysis extra (installed, not required)

librosa 0.11.0 is installed in the verified development environment and is
integrated as an optional backend behind the `analysis` extra. Core does not
import it, and its absence cannot disable rendering or the basic analyzer.
Installation pulls a transitive dependency tree whose principal licenses are:

| Component | Version installed | License metadata |
| --- | --- | --- |
| librosa | 0.11.0 | ISC |
| audioread | 3.1.0 | MIT |
| numba | 0.68.1 | BSD-2-Clause |
| llvmlite | 0.50.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| joblib | 1.6.0 | BSD-3-Clause |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| decorator | 5.3.1 | BSD-2-Clause |
| pooch | 1.9.0 | BSD-3-Clause |
| soxr (python-soxr) | 1.1.0 | LGPL-2.1-or-later |
| lazy_loader | 0.6 | BSD-3-Clause |
| msgpack | 1.2.4 | Apache-2.0 |
| typing-extensions | 4.16.0 | PSF-2.0 |

The list is not a complete notice file; `pip`/`uv` also resolve shared
dependencies such as NumPy, SciPy, SoundFile, requests, and platformdirs.
**soxr is LGPL-2.1-or-later**, so bundling a frozen application that includes
librosa's dependency tree raises notice, dynamic-linking, and relinking/source
offer questions similar to libsndfile. Keeping the extra optional and
user-installed may reduce exposure compared with shipping those binaries, but
distribution method must be reviewed by counsel. No wheels, native libraries,
model weights, or third-party source are included in this repository.
Installing the extra is a normal package-manager download initiated by the
user, not an upload of audio or project data.

Other candidate tools are **not integrated** in this prototype. They must be
evaluated as code, native libraries, model weights, and training-data terms
separately:

| Tool | Typical license | Intended fit | Main concern |
| --- | --- | --- | --- |
| librosa | ISC | Optional analysis, beat/tempo, chroma/key | **Integrated as an optional extra.** Pulls in numba/llvmlite and LGPL-2.1-or-later soxr; keep optional and audit distribution. |
| Demucs | MIT for code; pretrained weights are not covered by that license | Optional local stem separation behind `StemSeparator` | Reported weight terms are research/non-commercial; review before any commercial use and do not bundle weights by default. |
| aubio | GPL-3.0 | Onset/beat/tempo | Strong copyleft; not compatible with a permissive Core plus proprietary Advanced without a deliberate GPL strategy and legal review. |
| Essentia | AGPL-3.0 for the open-source edition; commercial license available | Analysis features | AGPL network copyleft is incompatible with the intended proprietary split unless a commercial license is obtained. |
| FFmpeg | LGPL-2.1-or-later or GPL-2.0+/GPL-3.0 depending on build | Import/export of additional formats | The installed Windows build is GPLv3; use only as a user-installed external process, never bundled, and review distribution terms. |

librosa was the lowest-risk first integration because its own ISC license is
compatible with either MIT or Apache-2.0 Core, but its transitive soxr
dependency still requires distribution review. Demucs code is MIT, but its
commonly used pretrained weights require separate legal review and should be
user-supplied, not downloaded or redistributed by DCP. aubio and Essentia are
not part of Core under the current licensing plan.

The repository does not bundle or ship a model, weight file, FFmpeg binary, or
third-party Python package. Installing the `analysis` extra asks the package
manager to fetch librosa and its dependencies from PyPI under their own terms.
A future hosted model or service would require its own terms, privacy review,
and explicit user authorization before audio is transmitted.

Dependencies that are dynamically loaded or invoked as separate processes may
have different obligations from vendored code. Distribution method matters.

## Copyright in audio

- DCP must not be used as a tool to bypass copyright or access controls.
- Do not distribute copyrighted commercial recordings with the project.
- The synthetic demo material is generated locally and is reproducible from
  code; that does not automatically clear every jurisdiction's rights in
  generated outputs, so review remains prudent.
- Users are responsible for having the rights to the recordings and stems they
  process.
- Model training data, model weights, source code, and generated audio are
  distinct assets with potentially different terms.

## Commercial model options (not implemented)

The architecture is intended to support future commercial structures without
adding any of them now:

1. **Commercial integration license** - a paid license to embed DCP Advanced in
   a product.
2. **Annual business license** - time-limited organizational access and support.
3. **Usage-based billing** - metered by processed minutes, renders, or active
   users.
4. **Per-transition or per-stream royalties** - potentially practical in some
   distribution models, but difficult to define, audit, and reconcile with
   privacy expectations.
5. **Enterprise agreements** - negotiated support, indemnity, security, and
   deployment terms.

None of these are active. No billing, payment processing, license enforcement,
DRM, royalty collection, or telemetry exists in the prototype. Do not add fake
license checks or artificial Core limitations.

## Metering interface concepts (design only)

If a future commercial edition needs metering, it should be:

- outside DCP Core;
- opt-in and disclosed;
- local by default;
- free of audio content and personal data unless explicitly required and
  authorized;
- deterministic and documented, so customers can audit counts;
- reviewed for privacy, security, and contract law before use.

A plausible event schema would count only technical events, for example:
`transitions_rendered`, `audio_seconds_processed`, `strategy_id`,
`engine_version`, and a customer-provided installation identifier. No counters,
storage, network calls, or identifiers are implemented in this prototype.

Any per-stream or per-transition royalty definition must specify what counts as
a transition, how retries and previews are treated, how offline use is handled,
and how audits respect user privacy. Those questions are unresolved.

## Contributions and governance

Before accepting external contributions, the owner should decide:

- the Core license and contribution policy (DCO, CLA, or another model);
- copyright ownership and employer-contribution handling;
- a code of conduct and security-reporting process;
- trademark and naming policy;
- whether Advanced-related contributions can be accepted into Core at all.

No contribution policy is active yet.

## Name and trademarks

"DCP" and "Dynamic Canvas Processing" have not been cleared as trademarks. Do
not assume availability in any market. Naming, branding, and domain use require
separate review.

## Pre-launch checklist

- [ ] Owner authorizes a Core license in writing.
- [ ] Counsel reviews Core license, Advanced terms, and notices.
- [ ] Dependency and native-library obligations are audited for the actual
      distribution format, including libsndfile and any optional extra's
      LGPL-licensed soxr dependency.
- [ ] Model weights and training-data terms are reviewed before any separator is
      bundled or downloaded.
- [ ] Commercial pricing, royalty definitions, and metering are defined.
- [ ] Privacy and security review is completed for any hosted or metered
      feature.
- [ ] Trademark and naming clearance is completed.
- [ ] Publication and repository visibility are explicitly approved.

Until then, keep the project local and unpublished, and do not add a final
license without authorization.
