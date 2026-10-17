# Group 13 — Multimodal Emotion Recognition

PRS Practice Module: **Multimodal Emotion Recognition Using Speech and Facial Expressions**.
This private repository is the team's shared code workspace for the RAVDESS speech subset (24 actors, eight emotions).

## Project layout

| Directory | Purpose | Current state |
|---|---|---|
| `audio/` | M1 handcrafted and M2 learned audio encoders | Implemented under `handcrafted/` and `learned/` |
| `visual/` | Face preprocessing, handcrafted features and learned encoders | Reserved for teammates' code |
| `multimodal_fusion/` | Decision, concatenation, gated and cross-attention fusion | Existing implementation |
| `demo/` | External audio-video prediction interface | Reserved; not implemented |
| `docs/` | Submission checklist, integration and collaboration notes | Initial guidance |

The current upload contains the fusion implementation, tests and shared actor split. The full end-to-end system is not complete. Synthetic experiments are engineering checks, not RAVDESS results. A real M2 learned-only audio pilot has been performed locally; real visual fusion results are still pending. Generated results, datasets and checkpoints are not included in this initial upload.

## Run the fusion engineering check

```bash
git clone https://github.com/caiji666/prs-group13-multimodal-emotion-recognition.git
cd prs-group13-multimodal-emotion-recognition/multimodal_fusion
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fusion demo --output results/my_first_demo --epochs 30
```

See [fusion documentation](multimodal_fusion/README.md) for feature export and training. Download and prepare RAVDESS separately; the code-only checkout does not contain data or trained weights.

## Shared integration contract

- Use the agreed [actor split](multimodal_fusion/configs/actor_splits.csv): 16 train / 4 validation / 4 test actors. Do not randomly repartition samples.
- For **every real-data fusion command**, explicitly use `--split configs/actors_m2.json` from inside `multimodal_fusion/`. The CLI now defaults to the formal `actors_m2.json`; `actors.json` retains the historical development split.
- Derive `sample_id` by removing only the modality field and extension: `03-01-06-01-02-01-12.wav` → `01-06-01-02-01-12`. Preserve leading zeros, and join modalities by ID.
- Preserve labels, actor/split metadata and probability order: `neutral, calm, happy, sad, angry, fearful, disgust, surprise` (labels 0–7).
- Select models and fusion weights on validation macro-F1. Reserve the test actors for final evaluation.
- Compare audio-only, visual-only and all four fusion methods. Cross-attention is included in the team's later agreed experiment scope.

Detailed interfaces and experiment plans are in [the integration document](multimodal_fusion/实验规划与当前代码说明.md).

## Team workflow

Each member works on a module branch, such as `feature/audio-handcrafted`, `feature/audio-learned`, `feature/visual`, `feature/fusion` or `feature/evaluation-demo`, and opens a pull request into `main`. These are suggested branch names, not a reassignment of the team's existing responsibilities.

Before merging, include the entry command, dependencies, input/output fields, validation results and any interface changes. Keep machine-specific dataset paths in ignored local config. Store large datasets and trained weights separately with retrieval instructions and checksums; they still need to be accounted for in the final course submission.

Confirmed from earlier chats: Jiang Yushan owns multimodal fusion; M2 supplies learned audio features, and M1 supplies handcrafted/final audio output. Other member-to-module assignments are not fully available in the inspected chats and are not guessed here.

## Course submission

GitHub is a collaboration workspace. The course handout does not require a GitHub repository or a public repository. Final submission is **one team ZIP on Canvas by 31 October 2026**, with individual reports submitted separately. See [the requirements checklist](docs/course-requirements.md).

## M2 learned audio module

See [audio/learned/README.md](audio/learned/README.md) for full Stage 1–6 source, selected tuned Transformer weights, WAV emotion inference, training and M1/M5 exports. The shipped model has validation Accuracy 54.58% and Macro-F1 0.5324 (round 05); final test generalization remains unmeasured. Six-field IDs and per-sample sequences are unchanged; class 7 is named `surprise`.
