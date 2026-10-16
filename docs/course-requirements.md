# Course requirements and project commitments

Reviewed against the local `PRS-PatternRecognitionSystems-Practice-Module 7.0 - FT.pdf` (pages 8, 9, 15 and 16), `Project_Proposal_Group13.pdf` (sections 1–5), and the user's later project messages on 16 October 2026. The course documents are reference material; this checklist distinguishes their requirements from team decisions.

## Course handout requirements

- Maximum five team members, enrolled in a Canvas project group.
- Develop, integrate and demonstrate at least three of: supervised/unsupervised learning; machine/deep learning; hybrid/ensemble approaches; intelligent sensing/sense making. The proposal appears to cover these through supervised classification, deep encoders, multimodal fusion and signal processing; make the coverage explicit in the report.
- Submit a runnable pattern recognition system, datasets, source code, model files and relevant supporting documents.
- Final report covers tools/techniques, system design/models, performance, findings and discussion.
- Include a 10–15 minute presentation video and slides from both presentations.
- Each member submits a separate 1–2 page individual reflection covering contributions, useful learning and application elsewhere, plus peer review through the peer review system.
- Deadline stated in the handout: **31 October 2026**. Destination: Canvas → Assignments → Practice Module. One ZIP per team, except the individual reports.
- No GitHub requirement, mandatory repository name, mandatory repository layout or public-visibility requirement was found in the handout. A repository URL alone does not replace the stated ZIP submission.

## Proposal commitments

- RAVDESS speech only; 24 actors and eight emotion classes; paired audio and visual inputs.
- Actor-independent train/validation/test partitions shared by all modules.
- Audio: handcrafted, learned and combined representations; compare CNN, LSTM/BiLSTM and Transformer encoders.
- Visual: DWT-SVM baseline; compare learned configurations and handcrafted/learned/combined representations.
- Fusion: decision-level, feature concatenation and gated fusion. The written proposal describes cross-attention as resource-dependent; the user's later presentation instruction includes it as an experiment, so the repository scope includes all four.
- Primary metric: macro-F1; also accuracy, precision, recall, F1 and confusion matrices. Select configurations on validation data; reserve test data for final evaluation.
- Deliver external manually labelled sample evaluation and a lightweight demonstration interface.
- Repeated actor splits are conditional in the proposal. The later team agreement fixes the main actor split; any supplementary split study must be kept separate.

## Later team agreements

- `multimodal_fusion/configs/actor_splits.csv` is the fixed main-experiment partition (960 train / 240 validation / 240 test speech trials).
- Shared six-field sample IDs, label mapping and probability order must survive all processing stages. Reject duplicates, missing pairs and label/split mismatches before fusion.
- Include interpretation of fusion results, following the user's report of the professor's feedback. Planned analyses are described in the fusion experiment document; they are not yet completed results.

## Final packaging checklist

- [ ] Integrate and verify audio, visual, fusion and demonstration entry points.
- [ ] Record exact environment/dependencies and reproducible run commands.
- [ ] Include required datasets and model files in the submission plan; resolve packaging/access details with course staff if size prevents the prescribed ZIP format.
- [ ] Include actual final metrics, confusion matrices, comparisons and interpretation.
- [ ] Include group report, both slide decks and 10–15 minute video.
- [ ] Submit the team ZIP to Canvas; submit individual reports and peer reviews separately.

Large artifacts are excluded from Git history for development. That exclusion does not remove the course's dataset/model deliverables.
