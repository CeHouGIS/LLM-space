# Run notes

- V0 cache and 1,398 transformed image paths passed the manifest audit.
- E1–E4 were run on all cached scenes with seeds 17, 42, 73.
- A position feature revision removed direct bounding-box coordinates before final metrics were generated.
- The only visually reviewed test scene contains two objects. Reviewed-only confidence intervals are unavailable.
- Three two-step image transforms were extracted with the local frozen model snapshot; only one test scene supports shift-right then shift-up.
- No extra manual annotation review was performed. Visibility matching in E3 was unavailable because the annotation schema has no verified visibility field.
- No causal patching or extra vision models were run, as specified by the V0.1 scope.
