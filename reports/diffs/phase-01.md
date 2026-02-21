# Phase 01 Diff Summary

- Migrated core pipeline modules from `quant_platform` into `gbdt_agent` package.
- Reworked orchestration to fixed stages `stage_00` to `stage_80`.
- Standardized target label to `future_return_20d` (configurable by horizon).
- Simplified training stack to GBDT-only and added backend fallback paths.
- Added mandatory artifact manifest generation per stage.
