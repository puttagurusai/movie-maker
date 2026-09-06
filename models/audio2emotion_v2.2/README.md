---
license: other
license_name: nvidia-audio2emotion-license
license_link: >-
  https://huggingface.co/nvidia/Audio2Emotion-v2.2/blob/main/LICENSE
extra_gated_prompt: "By clicking “Agree and Access,” you agree that you will use the NVIDIA Audio2Emotion Model consistent with the [License Agreement for NVIDIA Audio2Emotion Model](https://huggingface.co/nvidia/Audio2Emotion-v2.2/blob/main/LICENSE), which allows you to use the Model only with the NVIDIA Audio2Face Project and prohibits use of the Model or any of its components for emotion recognition."
extra_gated_fields:
  I confirm that this model will be used in accordance with its license: checkbox
---
## Model Overview

### Description

This model is a speech emotion recognition (SER) classifier that can predict six emotions from speech: **anger**, **disgust**, **fear**, **joy**, **neutral**, and **sadness**. It is based on the Wav2Vec2 architecture and is trained to classify emotions in a sequence of audio frames. This model is ready for commercial/non-commercial use.
For source code, documentation, helper scripts, packaged builds, and links to all components in the Audio2Face-3D technology stack, visit the [Audio2Face-3D GitHub repository](https://github.com/NVIDIA/Audio2Face-3D)

### License/Terms of Use

Use of this model is governed by the [License Agreement for NVIDIA Audio2Emotion Model for Use with Audio2Face Project](https://huggingface.co/nvidia/Audio2Emotion-v3.0/blob/main/LICENSE)

AUDIO2EMOTION MODEL NOTICE: This model and any technology included with this model may only be used in connection with the [NVIDIA Audio2Face project](https://docs.omniverse.nvidia.com/audio2face/latest/overview.html) consistent with all applicable documentation. You may not use this model and any technology included with it outside of the Audio2Face project. You may not use this model or any of its components for the purpose of emotion recognition.

### Deployment Geography:
Global

### Use Case:
**IMPORTANT:** This Model and any technology included with this Model may only be used in connection with the NVIDIA Audio2Face project (https://docs.omniverse.nvidia.com/audio2face/latest/overview.html) consistent with all applicable documentation. You may not use this Model and any technology included with it outside of the Audio2Emotion model outside the Audio2Face project. You may not use this Model or any of its components for the purpose of emotion recognition.

This speech emotion recognition model is specifically designed and optimized for the NVIDIA Audio2Face project to generate realistic facial expressions for 3D characters. The model's primary and intended use case is converting speech audio into emotional states that drive realistic 3D facial animations. The model is not intended for standalone emotion recognition applications or general-purpose audio analysis. It has been specifically trained and optimized to work as a component within the Audio2Face pipeline to produce high-quality, emotionally accurate 3D facial expressions that enhance the realism of virtual characters and digital humans.

### Release Date:
Release Date: 09/24/2025 [HuggingFace](https://huggingface.co/nvidia/Audio2Emotion-v2.2)

---

## Model Architecture

- **Architecture Type:** Transformer
- **Network Architecture:** Wav2Vec2
- **This model was developed based on**: [Wav2Vec2-Large-LV60](https://huggingface.co/facebook/wav2vec2-large-lv60)
- **Number of model parameters**: 3.1 x 10^8

### Input

- **Input Type(s):** Audio
- **Input Format(s):** Raw audio input - an array of `float32`
- **Input Parameters:** 2D
- **Other Properties Related to Input:** A batch of input waveforms for classification

### Output

- **Output Type(s):** Probabilities of emotional classes
- **Output Format:** An array of `float32`
- **Output Parameters:** 2D
- **Other Properties Related to Output:** The model can predict six emotions from speech: Anger, disgust, fear, joy, neutral, and sadness.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems [or name equivalent hardware preference]. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

---

## Software Integration

### Runtime Engine(s)

- **NeMo** - 1.0.0

### Supported Hardware Microarchitecture Compatibility

- NVIDIA Ampere <br>
- NVIDIA Blackwell <br>
- NVIDIA Hopper <br>
- NVIDIA Lovelace <br>
- NVIDIA Pascal <br>
- NVIDIA Turing <br>

### [Preferred/Supported] Operating System(s)

- Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  
This AI model can be embedded as an Application Programming Interface (API) call into the software environment described above.

---

## Model Version(s)
Audio2Emotion-v2.2

---

## Training, Testing, and Evaluation Datasets

### Training Dataset

**Data Modality**
- Audio

**Audio Training Data Size**
- Less than 10,000 Hours

#### Link

- Internal datasets
- [RAVDESS](https://zenodo.org/records/1188976)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [JL Corpus](https://www.kaggle.com/datasets/tli725/jl-corpus)
- [EMO-DB](https://audeering.github.io/datasets/datasets/emodb.html)
- [Emozionalmente](https://zenodo.org/records/6569824)


#### Data Collection Method by dataset

- Automated

#### Labeling Method by dataset

- Human

#### Properties (Quantity, Dataset Descriptions, Sensor(s))

- Multiple datasets, including RAVDESS, CREMA-D, JL, EMO-DB, Emozionalmente, TTS GPT 4o (internal), Lindy & Rodney (internal)
- Quantity: 30029 samples

### Testing Dataset

#### Link

- Internal dataset

#### Data Collection Method by dataset

- Automated

#### Labeling Method by dataset

- Human

#### Properties (Quantity, Dataset Descriptions, Sensor(s))

- Internal crowdsourced dataset
- **Quantity:** 1350 samples

### Evaluation Dataset

#### Link

- Internal dataset

#### Data Collection Method by dataset

- Automated

#### Labeling Method by dataset

- Human

#### Properties (Quantity, Dataset Descriptions, Sensor(s))

- Internal crowdsourced dataset
- **Quantity:** 1350 samples

---

### Inference

#### Engine

- Tensor(RT)

#### Test Hardware
- T4, T10, A10, A40, L4, L40S, A100 <br>
- RTX 6000ADA, A6000, Pro 6000 Blackwell  <br>
- RTX 3080, 3090, 4080, 4090, 5090  <br>

---

### Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

This Model and any technology included with this Model may only be used in connection with the NVIDIA Audio2Face project (https://docs.omniverse.nvidia.com/audio2face/latest/overview.html) consistent with all applicable documentation. You may not use this Model and any technology included with it outside of the Audio2Emotion model outside the Audio2Face project. You may not use this Model or any of its components for the purpose of emotion recognition.

# Bias

Field                                                                                               |  Response
:---------------------------------------------------------------------------------------------------|:---------------
Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing:  |  Age, Gender, Linguistic Background, Accent, Speech Patterns, and Cultural Context
Measures taken to mitigate against unwanted bias:                                                   |  Training data includes diverse speakers across multiple datasets (RAVDESS, CREMA-D, JL, Lindy & Rodney, EMO-DB, Emozionalmente, TTS GPT 4o) to reduce demographic bias


# Explainability

Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                   |  Speech Emotion Recognition, Audio Analysis, Human-Computer Interaction, and Audio2Face Integration
Model Type:                                                                                            |  Speech emotion recognition classifier
Intended Users:                                                                                        |  Audio2Face developers, Speech analysis researchers, Human-computer interaction developers, Affective computing researchers
Output:                                                                                                |  Emotion probabilities (six classes: anger, disgust, fear, joy, neutral, and sadness)
Describe how the model works:                                                                          |  Audio input is processed through Wav2Vec2 architecture to classify emotions from speech, outputting probability scores for six emotional states
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  People with speech disorders or non-native accents, Non-English speakers or those with strong regional accents, Elderly individuals with age-related speech changes
Technical Limitations & Mitigation:                                                                    |  Model requires clear audio input at 16kHz sampling rate, may struggle with overlapping speech or very noisy environments
Verified to have met prescribed NVIDIA quality standards:  |  Yes - Model achieves high accuracy on clean audio inputs, validated on internal crowdsourced dataset
Performance Metrics:                                                                                   |  Accuracy (Top-1) - 80%+ on clean audio, Throughput & Latency, Emotion classification confidence scores
Potential Known Risks:                                                                                 |  Model may misclassify emotions in edge cases, should not be used for standalone emotion analysis without Audio2Face integration
Licensing:                                                                                             |  Use of this model is governed by the [License Agreement for NVIDIA Audio2Emotion Model for Use with Audio2Face Project](https://huggingface.co/nvidia/Audio2Emotion-v3.0/blob/main/LICENSE)


# Privacy

Field                                                                                                                              |  Response
:----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------
Generatable or reverse engineerable personal data?                                                     |  Emotion classification probabilities from audio input
Personal data used to create this model?                                                                                       |  Yes - Audio recordings containing human speech and emotional expressions
Was consent obtained for any personal data used?                                                                                             |  Yes
How often is dataset reviewed?                                                                                                     |  Before Every Release
Is a mechanism in place to honor data subject right of access or deletion of personal data?                                        |  Yes
If personal data was collected for the development of the model, was it collected directly by NVIDIA?                                            |  Yes
If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects?  |  Yes
If personal data was collected for the development of this AI model, was it minimized to only what was required?                                 |  Yes - Only audio features necessary for emotion recognition are processed
Is there provenance for all datasets used in training?                                                                                |  Yes
Does data labeling (annotation, metadata) comply with privacy laws?                                                                |  Yes
Is data compliant with data subject requests for data correction or removal, if such a request was made?                           |  Yes
Applicable Privacy Policy        | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/


## Safety & Security
Field                                               |  Response
:---------------------------------------------------|:----------------------------------
Model Application Field(s):                               |  Speech emotion recognition for driving Audio2Face 3D facial animations
Describe the life critical impact (if present).   |  Not Applicable - Model is designed for entertainment and communication applications, not life-critical systems
Use Case Restrictions:                              |  This Model and any technology included with this Model may only be used in connection with the NVIDIA Audio2Face project (https://docs.omniverse.nvidia.com/audio2face/latest/overview.html) consistent with all applicable documentation. You may not use this Model and any technology included with it outside of the Audio2Emotion model outside the Audio2Face project. You may not use this Model or any of its components for the purpose of emotion recognition. Abide by [License Agreement for NVIDIA Audio2Emotion Model for Use with Audio2Face Project](https://huggingface.co/nvidia/Audio2Emotion-v3.0/blob/main/LICENSE).
Model and dataset restrictions:            |  The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to.

## Citation
```
@misc{nvidia2025audio2face3d,
      title={Audio2Face-3D: Audio-driven Realistic Facial Animation For Digital Avatars},
      author={Chaeyeon Chung and Ilya Fedorov and Michael Huang and Aleksey Karmanov and Dmitry Korobchenko and Roger Ribera and Yeongho Seol},
      year={2025},
      eprint={2508.16401},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2508.16401},
      note={Authors listed in alphabetical order}
}
```