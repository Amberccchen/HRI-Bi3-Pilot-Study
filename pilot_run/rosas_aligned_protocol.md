# Questionnaire-aligned model prompt

Created: 2026-10-07. This prompt was used for the blinded LAAS12 pilot, but **not** for the earlier LAAS3–11 pilots. The Bi³ papers report the RoSAS items and nine-point scale but do not publish the exact online questionnaire form; the wording below is an adaptation, not a verbatim Bi³ survey instruction.

## Fixed model prompt

> You will see an anonymized recording or representation of one completed robot navigation trial. Two people shared the workspace with the robot. Give **your own impression of the robot's behavior in this trial**. Using a scale from **1 = definitely not associated** to **9 = definitely associated**, how closely is each word below associated with the robot as it behaved in this trial?
>
> Aggressive; awful; scary; awkward; dangerous; strange; knowledgeable; interactive; responsive; capable; reliable; competent.
>
> Give one integer from 1 to 9 for each word. Judge each word separately. Use only the trial material provided. Do not infer or use the controller's identity, and do not assume that a measured distance directly reveals either person's feelings. If the material is insufficient for an attribute, still give your best rating and flag uncertainty separately. Do not predict, combine, or average the two participants' answers.
>
> Return JSON with exactly these keys: `aggressive`, `awful`, `scary`, `awkward`, `dangerous`, `strange`, `knowledgeable`, `interactive`, `responsive`, `capable`, `reliable`, `competent`, `uncertain_attributes`, `brief_behavioral_evidence`. Each of the twelve attribute values must be an integer. `uncertain_attributes` must be an array of item names; `brief_behavioral_evidence` must be one sentence about observed behavior, without speculation about the participants' internal state.

## Administration and scoring

- Present the same fixed prompt for every anonymous trial. Keep controller names and survey ratings sealed until all predictions are committed.
- Use one fresh model context per trial and information condition when testing independent modalities. Record exact model identifier, sampling settings, input files, prompt, outputs, and run time. If using one cumulative context instead, label it an incremental information study rather than an independent modality comparison.
- Keep the original twelve item predictions. Compute model discomfort as the mean of the six discomfort items and model competence as the mean of the six competence items **after** the response is saved. Compare each with both individual participant responses and their pair mean; report human disagreement.
- Keep the factual perception probe separate from this rating prompt. Score its event answers against mocap, then ask whether factual accuracy relates to rating error. Do not provide survey labels or controller names to the perception probe.
- For frame-based input, record frame cadence and any mocap-selected event windows. Sampled frames cannot be described as full-video input.

## Why this replaces the earlier prompt

The earlier pilots asked the model to predict the **average answer of two people** and used anchors “not at all” and “very much.” The Bi³ study instead collected **individual** ratings of how much attributes applied to the robot, using a nine-point adapted RoSAS instrument. The original RoSAS scale uses “definitely not associated” and “definitely associated” as endpoints. This prompt aligns the model's *question and response format* with that construct; it does not recreate the unavailable exact Bi³ form or the participants' embodied experience.

Sources: [Bi³ dataset paper](https://arxiv.org/html/2605.06863), [Bi³ user-study paper](https://fluentrobotics.com/pdfs/stratton2026hmp2nav.pdf), [original RoSAS paper](https://www.researchgate.net/publication/314159812_The_Robotic_Social_Attributes_Scale_RoSAS_Development_and_Validation).
