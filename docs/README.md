# Documentation

## Contents

| Symbol | Meaning |
| ---- | ---- |
| 🔰 | Suitable for laypeople |
| ⚙️ | Technical details |
| 🤔 | Theories, interpretations, and opinions |

0. [Toplevel readme](../README.md). 🔰 _The introduction to the project, including examples, license, and credits._
1. [Quickstart](quickstart.md). ⚙️ _If you just want to see the code work, without worrying about what it’s doing._
2. [Potato’s main features](features.md). _Specifics on the unusual features of this project compared to other pansharpening approaches._
  1. Preface: [Beyond aspatial images](features.md#preface-beyond-aspatial-images). 🔰 _Satellite images have affordances and artifacts that “normal” images don’t._
  2. [No per-sample normalization](features.md#no-per-sample-normalization). ⚙️ _We use the images’ absolute calibration._
  3. [All-band color conversion](features.md#all-band-color-conversion). ⚙️ _We use the images’ rich spectral information._
  4. [Artifact injection](features.md#point-spread-functions-and-band-misalignment). ⚙️🤔 _We teach the model to correct some sensor-specific problems._
  5. Appendix: [Minor techniques](features.md#appendix-minor-techniques). ⚙️ _Various ideas not needing lengthy explanations._
3. [Personal reflections](personal.md). 🔰🤔 _Informal notes on this project’s motivations._