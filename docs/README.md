# Documentation

This is the index of Potato’s documentation. The advanced reader will find fine-grained technical documentation in code comments.

## Contents

| Symbol | Meaning |
| ---- | ---- |
| 🔰 | Suitable for laypeople |
| ⚙️ | Technical details |
| 🤔 | Theories, interpretations, and opinions |

0. [Toplevel readme](../README.md). 🔰 _The front page introduction to the project: examples, license, and credits. You probably already saw this._
1. [**Quickstart**](quickstart.md). ⚙️ _To pansharpen something in as few steps as possible._
3. [Potato’s features](concepts.md). _The unusual parts of this project compared to other pansharpening approaches, divided into:_
    1. [Preface](concepts.md#preface). 🔰 _What Potato tries to show – conceptual grounding for the project._
    2. [No per-sample normalization](concepts.md#no-per-sample-normalization). ⚙️ _We use the images’ absolute calibration._
    3. [All-band color conversion](concepts.md#all-band-color-conversion). ⚙️ _We use the images’ rich spectral information._
    4. [Artifact injection](concepts.md#point-spread-functions-and-band-misalignment). ⚙️🤔 _We teach the model to correct some sensor-specific problems._
    5. [Minor features](concepts.md#minor-features). ⚙️🤔 _Ideas not worth lengthy analysis._
4. [CLI tool guide](cli.md). ⚙️ _Utilities for applying, training, and evaluation._
5. [Notes on CID selection](../ancillary-data/cids/). ⚙️ _How to make a scene allow-list._
6. [Personal reflections](personal.md). 🔰🤔 _Subjective notes on what this all means._
