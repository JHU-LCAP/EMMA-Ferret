# EMMA-Ferret
Single stream EMMA framework used for Ferret Stimuli Analysis

## Introduction
This repository implements a computational model inspired by the Explicit Memory Multi-resolution Adaptive (EMMA) framework for speech segregation. The model is designed to explore and provide a framework for understanding the role of temporal coherence in segregating complex auditory scenes, as observed in biological experiments.

## Overview
The model aims to segregate a target voice from a mixture of voices by incorporating a directive or attentional focus, similar to how ferrets were trained to direct their attention to a specific voice. The model leverages two key principles to achieve this:

High-dimensional Nonlinear Mapping:
The input voice mixture is mapped to a nonlinear high-dimensional space, simulating the diverse selectivity of cortical neurons to attributes like frequency, pitch, and location. This stage is implemented using deep-learning neural embeddings referred to as pre-attentional model embeddings (Mp), which capture all the characteristics of the input mixture.

Attentional Focus for Voice Segregation:
The attentional focus stage (Ma) selectively enhances the representation of the attended target voice while suppressing other competing sources. This is achieved by aligning the Mp embeddings based on their temporal coherence with the target voice. This stage enables the model to gate the Mp embeddings in a way that emphasizes the target voice features, allowing for successful segregation.

## Installation
To run this code, follow these steps:
1. Clone the repository.
2. Create a new conda env using the environment.yml.
3. Run the code.

## Usage
Here are the main functionalities of the code:
- Functionality 1
- Functionality 2
- ...

## Code Structure
The code is organized as follows:
- `src/`: Contains the source code files.
- `data/`: Contains the dataset used.
- `results/`: Contains the output results.

## Contributors
- John Doe
- Jane Smith

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
