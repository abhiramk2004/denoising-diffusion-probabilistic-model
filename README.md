## 🌀 Diffusion Model on Flower Dataset

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** using PyTorch, trained on the Flowers dataset. It features a custom UNet with residual blocks, time-step embeddings, and self-attention for generating realistic 32×32 flower images.

---

### 📁 Directory Structure

```
.
├── Logs_Checkpoints/
│   ├── Checkpoints/version_X/
│   └── Inference/version_X/
├── Samples/
│   └── version_X/
├── flowers/
│   └── class folders with images
├── diffusion_model.py
└── README.md
```

---

### 🚀 Features

* ✔️ Forward and Reverse Diffusion processes
* ✔️ UNet backbone with residual connections & attention
* ✔️ Sinusoidal time-step embeddings
* ✔️ Mixed precision training (AMP)
* ✔️ Dynamic logging, checkpointing, and sample generation
* ✔️ Modular code structure for easy extension

---

### 📦 Dependencies

Make sure you have the following Python libraries installed:

```bash
pip install torch torchvision torchaudio torchmetrics matplotlib tqdm pillow
```

---

### 📂 Dataset

Place your flower image dataset in the `flowers/` directory structured like this:

```
flowers/
├── daisy/
│   ├── img1.jpg
│   └── ...
├── rose/
└── ...
```

The dataset will be:

* Resized to **32x32** using bicubic interpolation
* Normalized between `[-1, 1]`
* Randomly horizontally flipped

---

### 🏋️‍♂️ Training

Run the script to start training:

```bash
python diffusion_model.py
```

* Model checkpoints will be saved every 10 epochs in:

  ```
  Logs_Checkpoints/Checkpoints/version_X/
  ```
* Sample images will be saved in:

  ```
  Samples/version_X/
  ```

---

### 🧪 Sampling

Before training begins, reverse diffusion will sample images using the latest checkpoint (if available). Images will be saved to `Samples/`.

You can also call the `reverse_diffusion()` function manually to generate new samples.

---

### 🛠 Configuration

All configuration options are in the `BaseConfig` and `TrainingConfig` dataclasses:

```python
@dataclass
class TrainingConfig:
    TIMESTEPS = 1000
    IMG_SHAPE = (3, 32, 32)
    NUM_EPOCHS = 800
    BATCH_SIZE = 32
    LR = 1e-3
```

You can change `BATCH_SIZE`, `TIMESTEPS`, or model parameters like `base_channels` in `UNet`.

---

### 🧼 Clean Training

The script includes:

* AMP mixed precision (`torch.cuda.amp`)
* Gradient clipping (`max_norm=1.0`)
* Checkpoint resume support
* GPU-agnostic device selection

---

### 📈 TODOs & Extensions

* [ ] Add FID/IS score calculation
* [ ] Add larger image support (e.g., 64x64, 128x128)
* [ ] Support DDIM sampling
* [ ] Add TensorBoard or W\&B logging
* [ ] Implement video generation using `frame2vid()`

