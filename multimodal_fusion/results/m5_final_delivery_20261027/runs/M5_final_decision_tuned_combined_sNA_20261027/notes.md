# M5_final_decision_tuned_combined_sNA_20261027 — Pipeline Experiment Record v2

Owner/member: M5 / Jiang Yushan. Structured record: [run.json](run.json)

## 1. Question and comparison

Final-input replay: decision_tuned; audio=learned, visual=combined, seed=None. Compare within the matching representation and method family. [Locked plan](../../plan.json).

## 2. Pipeline at a glance

Frozen audio: WavLM continuation20261021 Round05, 128D attentive-statistics pooled and T×128 sequence. Audio handcrafted ablations use separately paired48D; combined176D contains NEW WavLM, not historical MFCC features. Visual=combined (896D), fixed16 valid frames. Train-only standardization.

Classifier: No new classifier; fixed supplied probabilities and validation-only scalar mixing.

## 3. Results and selection

Validation Macro-F1=0.849631, Accuracy=0.858333. Test Macro-F1=0.828621, Accuracy=0.833333. Best epoch=None. Test completed after evaluation_lock.json; no test reselection.

## 4. Interpretation and limitations

Lowest test class F1: sad=0.6780. Largest confusion: sad→fearful (6 samples). These are descriptive test findings, not a new tuning objective. Upstream feature selection, representation and classifier effects are distinct. Historical test was already exposed, so this replay is exploratory. Physical time alignment remains unverified; only relative token positions are used.

## 5. Handoff and next step

Checkpoint/hash, source-run identity, normalizers, history and per-split predictions are linked in run.json. No new encoder feature export is claimed. SVM probability columns are empty with separate OVR scores. Audio train probabilities are in-sample; no train stacking is fitted. Class order: neutral, calm, happy, sad, angry, fearful, disgust, surprise. Preserve all versions and candidates; new sources require new experiments.
