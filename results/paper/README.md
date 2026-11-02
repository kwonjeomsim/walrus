# Data behind the paper

This directory holds the measurements reported in "Deciding What to Compile
Before Running It: Memory-Aware JIT Selection for Embedded WebAssembly Using
Decision Trees" (Journal of Systems Architecture submission). The training data
and the trained model live elsewhere in the repository and are listed below.

All held-out results were measured on a Raspberry Pi 5 (AArch64) unless the
file name says `x86`. Times are seconds of total run time, including module
loading, selection and JIT compilation. Code sizes are bytes of the buffer the
JIT reserves for function bodies.

## Result files

| File | Contents | Used in the paper |
|---|---|---|
| `test60_arm.json` | 60 held-out benchmarks, eight modes, ten runs each, AArch64 | speed-up and code ratio table, peak RSS table, group table, Figures 4 and 5 |
| `test60_x86.json` | the same on the x86-64 machine | transfer table, x86-64 comparisons |
| `ksweep_arm.json`, `ksweep_x86.json` | tier-up with K from 1 to 16384 | tier-up curve in Figure 4 |
| `ablation_x86.json`, `ablation_cum_x86.json` | single and cumulative feature removal, three runs each | Figure 6 |
| `decision_cost_arm.json`, `decision_cost_x86.json` | interpreter, counters only, load only, load and predict | selection-overhead table |
| `paper_numbers.txt`, `speedup_tables.txt`, `decision_summary_*.txt` | numbers printed by the scripts below | text and tables |
| `app_miss/` | profiles, tree paths and compile-list runs for `janet` and `xz` | analysis of missed functions |
| `apps15/` | build script and labels for the six applications with enlarged inputs | held-out applications |
| `figure1/` | per-function training profile and plotting script | Figure 1 |

Each benchmark entry in `test60_*.json` and `ksweep_*.json` maps a mode name
to `t` (median time), `reps` (all run times), `code` (machine-code bytes), `n`
(compiled functions) and `rss_reps` (peak RSS of each run in bytes). The modes
are `interp`, `alljit`, `lazy`, `tierup`, `O1`, `O2`, `O3` and `oracle`.

`results/test60_arm_v6.json` at the repository root is an earlier run, made
before six held-out applications were rebuilt with larger inputs. The paper
uses `test60_arm.json` in this directory.

## Training data and model

| Path | Contents |
|---|---|
| `src/jit/JITModelData.h` | the deployed model (three trees, 47, 47 and 45 nodes) |
| `profile/train6_t{4.0,2.0,1.0}/` | per-function training profiles labelled at each code-cost weight |
| `profile/test6_t{4.0,2.0,1.0}/` | the same for the held-out benchmarks |
| `profile/codesize6/` | All-JIT machine-code bytes per function |
| `profile/oracle6_t1.0/` | oracle compile lists |
| `profile/baselines_train232.json` | interpreter and All-JIT times of the training benchmarks |
| `profile/cv_full_v6.json` | cross-validation scores of all 8,400 configurations |
| `profile/split_v6.json`, `profile/benches.tsv` | training and held-out split, run commands |

## Regenerating the numbers

From the repository root, with Python 3.12, NumPy and scikit-learn:

```
cd tools/jit-model
python paper_numbers.py --arm ../../results/paper/test60_arm.json \
    --x86 ../../results/paper/test60_x86.json --ksweep ../../results/paper/ksweep_arm.json
python speedup_tables.py --arm ../../results/paper/test60_arm.json \
    --x86 ../../results/paper/test60_x86.json \
    --ablation ../../results/paper/ablation_x86.json \
    --cumulative ../../results/paper/ablation_cum_x86.json \
    --ksweep ../../results/paper/ksweep_arm.json --ksweep-x86 ../../results/paper/ksweep_x86.json
python decision_summary.py ../../results/paper/decision_cost_arm.json --interp-total 2065.2
```

`tools/jit-model/retrain_v6.sh` retrains the model from `profile/train6_t*`
and reproduces `src/jit/JITModelData.h` exactly. It needs a release build of
this branch (`WALRUS_BIN`) and the benchmark modules named in the profiles,
because features are read from the WebAssembly binaries.

## Not included

The WebAssembly benchmark binaries are not included. They come from BEEBS,
TACLeBench, Embench-IoT, CoreMark-PRO, the LLVM test-suite and open-source
applications, each under its own license. `apps15/build_scaled.sh` shows how
the six enlarged application inputs were built.
