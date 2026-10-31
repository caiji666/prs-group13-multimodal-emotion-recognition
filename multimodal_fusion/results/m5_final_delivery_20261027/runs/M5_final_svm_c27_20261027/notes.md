# M5_final_svm_c27_20261027 — Pipeline Experiment Record v2

Owner/member: M5 / Jiang Yushan. Structured record: [run.json](run.json)

## 1. Question and comparison

Final-input replay: svm_kernel_fusion; audio=learned, visual=hc, seed=None. Compare within the matching representation and method family. [Locked plan](../../plan.json).

## 2. Pipeline at a glance

Frozen audio: WavLM continuation20261021 Round05, 128D attentive-statistics pooled and T×128 sequence. Audio handcrafted ablations use separately paired48D; combined176D contains NEW WavLM, not historical MFCC features. Visual=hc (320D), fixed16 valid frames. Train-only standardization.

Classifier: SVC probability=False, OVR decision scores; {"modality": "kernel_fusion", "kernel": "precomputed", "C": 10.0, "alpha": 0.75}

## 3. Results and selection

Validation Macro-F1=0.833345, Accuracy=0.841667. Test Macro-F1=0.828148, Accuracy=0.829167. Best epoch=None. Test completed after evaluation_lock.json; no test reselection.

## 4. Interpretation and limitations

Lowest test class F1: sad=0.6429. Largest confusion: sad→fearful (8 samples). These are descriptive test findings, not a new tuning objective. Upstream feature selection, representation and classifier effects are distinct. Historical test was already exposed, so this replay is exploratory. Physical time alignment remains unverified; only relative token positions are used.

## 5. Handoff and next step

Checkpoint/hash, source-run identity, normalizers, history and per-split predictions are linked in run.json. No new encoder feature export is claimed. SVM probability columns are empty with separate OVR scores. Audio train probabilities are in-sample; no train stacking is fitted. Class order: neutral, calm, happy, sad, angry, fearful, disgust, surprise. Preserve all versions and candidates; new sources require new experiments.
