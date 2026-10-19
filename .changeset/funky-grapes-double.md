---
'@mastra/core': patch
---

Fixed compareExperiments averaging only one attempt per item instead of all repeated trials.

compareExperiments grouped scores by scorer and item only, so an item run across several trial attempts (via `attempt` on submitExperimentResult or the native runner's repeated-trial support) had its score silently overwritten by whichever attempt's row was read last, instead of being averaged across attempts.

Comparisons involving repeated trials now average every attempt's score into one effective score per item before computing per-scorer stats and deltas, so `avgScore`, `delta`, and regression detection reflect the item's true mean across trials.
