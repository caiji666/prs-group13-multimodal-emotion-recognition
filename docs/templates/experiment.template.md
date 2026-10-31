# <run_id> — Pipeline Experiment Record v2

Owner/member: <M1–M5 and name>
Structured configuration and results: [run.json](run.json)

## 1. Question and comparison

What question does this experiment answer? Which run(s) are the controls? What changes, and what stays fixed? For a reused baseline, link its original record.

Comparison type: <feature extractor / representation / classifier / fusion / complete pipeline>.

## 2. Pipeline at a glance

Fill brief descriptions below; put exact parameters in `run.json`. Copy rows for multiple extractors. Do not assume extraction and classification were trained separately.

| Component | Method / structure | Input → output | Source run / trainability in this run |
|---|---|---|---|
| Feature extraction | <handcrafted method or encoder; include temporal encoder> | <raw input → sequence / pooled shape> | <source run or trained here; fixed / frozen / partial / trainable> |
| Representation | <pooling, normalization, concatenation> | <component dimensions → final feature dimension> | <normalizer fitted on train only> |
| Multimodal fusion | <not applicable / decision / concat / gated / cross-attention> | <audio/visual features or probabilities → output> | <upstream audio + visual run IDs; trainable?> |
| Classification head | <MLP full layer widths, activations, dropout / SVM kernel, C, gamma / none> | <D → 8 classes> | <source run or trained here; probability method> |

Full pipeline name: <e.g. MFCC → Transformer → mean pooling → MLP>.
Training regime: <handcrafted then classifier / frozen encoder then classifier / joint training / partial fine-tuning / probability fusion>.

## 3. Results and selection

Summarize the main validation finding. State the comparison group, selected checkpoint/epoch and whether this configuration is selected for the next stage. Test not run / completed / failed must be explicit in run.json. Do not infer test scores from validation scores.

All classification scores evaluate the full pipeline above. Extracted feature shape or feature-export success alone is not an emotion-classification result. If the head or its training changes along with the extractor, disclose that limitation before attributing improvement to the features.

## 4. Interpretation and limitations

Write 3–6 concrete sentences: which emotions improve or regress; major confusion pairs; possible explanations and supporting evidence; remaining uncertainty. Distinguish hypotheses from measured findings. Mention classifier/input changes that limit attribution.

## 5. Handoff and next step

For selected configurations: give feature/checkpoint links and hashes in run.json, actual array dimensions, class order, preprocessing, sequence length/padding-mask meaning, and any temporary learned-only input. State who consumes the output and what experiment follows.

Identify which tensor is exported as a feature (normally before the final classifier) and which tensor is the eight-class probability output. If the classifier trained the encoder jointly, identify the selected full checkpoint and extract features from that checkpoint in evaluation mode. If the encoder and head have separate checkpoints, link and hash both.

If blocked/failed: record error, missing dependency or upstream artifact. If any sample is excluded or a local code patch is used, link the ID/reason list or patch.
