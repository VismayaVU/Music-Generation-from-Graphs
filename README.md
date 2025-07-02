# 🎵 Graph Neural Networks for Harmonious Music Composition

This project explores how **Graph Neural Networks (GNNs)** can be applied to musical composition by learning patterns from real-world MIDI files. The goal is to use GNNs to generate emotionally resonant and structurally harmonious music by modeling musical elements as graph data.

## 📌 Project Overview

- **Course**: Social Computing
- **Project Title**: Graph Neural Networks for Harmonious Music Composition
- **Team Members**:
  - Vishwanath Sridhar [PES1UG22AM194]
  - Venkat Subramanian [PES1UG22AM188]
  - Shri Hari [PES1UG22AM154]
  - Vismaya Vadana [PES1UG22AM195]

## 🧠 What We Did

We built a music generation pipeline using:

- The **MAESTRO v3.0.0 dataset** (a large collection of classical piano MIDI recordings).
- **PyTorch Geometric** for graph construction and deep learning.
- **music21** for MIDI processing and symbolic music analysis.

Our pipeline includes:
1. Extracting melodic, rhythmic, and harmonic features from MIDI files.
2. Representing musical sequences as graphs.
3. Training a GNN to predict pitch and duration from graph features.
4. Generating new music based on learned musical patterns.

## 🧩 Key Technologies

- **Python**
- **PyTorch Geometric** – for Graph Neural Networks
- **networkx** – for building graph representations
- **music21** – for MIDI parsing
- **Pandas / Matplotlib** – for data handling and visualization

## 🗃️ Dataset

- **MAESTRO v3.0.0**: Used for extracting MIDI files and metadata.
- Each MIDI file is converted into a graph based on:
  - Nodes: Notes/Chords with attributes like pitch, duration, and offset.
  - Edges: Connections based on temporal flow and pitch proximity.

## ⚙️ Model Architecture

We use a simple GNN based on Graph Convolutional Networks (GCN):

- **Input features**: Pitch, duration, offset
- **2 GCN layers** followed by **global mean pooling**
- **Linear layer** to output pitch and duration predictions

```python
class MusicGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        ...
```

## 🚀 Music Generation

Once trained, the GNN can be seeded with a real music graph, and used to **generate new music** by predicting pitch and duration while preserving rhythmic and melodic patterns.

The output can be exported to MIDI for playback.

## 🛠️ Installation

Install required packages:

```bash
pip install torch_geometric music21 matplotlib pandas networkx
```

Unzip the dataset:

```bash
unzip maestro-v3.0.0-midi.zip
```

Run the project:

```bash
python social_computing_course_project.py
```

## 🎧 Output

The model generates MIDI files based on seed graphs. Sample outputs are created for durations of 5 and 10 seconds. These can be played back using a MIDI player or viewed using music21's built-in tools.

## 🙏 Acknowledgments

- MAESTRO Dataset by Google Magenta
- PyTorch Geometric Team
- music21 Library Contributors

---

**:D THANK YOU! :D**
