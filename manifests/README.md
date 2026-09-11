# Manifests

- `tasks/` contains the exact synthetic-task and LiveCodeBench slice definitions used by retained scripts.
- `source_files.json` records original allowlisted source sizes and SHA-256 values before transformation.
- `anonymization.json` records syntax-preserving adaptations and scanner scope.
- `exclusions.json` records material intentionally outside the release.
- `archive_manifest.json` at the archive root is authoritative for final exported bytes.

Generated control replay manifests cover anonymous bytes and are intentionally absent from `source_files.json`.
