# GAN (Generative Adversarial Network) — PyTorch Implementation

A minimal, well-commented implementation of a **vanilla GAN** trained on the
MNIST handwritten digit dataset. The goal is to train a **Generator** network
that learns to produce realistic-looking digit images from random noise,
by having it compete against a **Discriminator** network that tries to
tell real digits apart from generated ("fake") ones.

---

## 1. What is a GAN?

A GAN consists of two neural networks trained simultaneously, in a
minimax game:

| Network | Job |
|---|---|
| **Generator (G)** | Takes random noise `z` and outputs a synthetic image, trying to fool the Discriminator into thinking it's real. |
| **Discriminator (D)** | Takes an image (real or generated) and outputs a probability that it's real. |

They are trained with opposing objectives:

- **Discriminator** wants to correctly classify real images as real (label `1`) and generated images as fake (label `0`).
- **Generator** wants the Discriminator to classify its generated images as real (label `1`).

Formally, the two networks play the minimax game:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

where `x` is a real image and `z` is a random noise vector. Over many
iterations, the Generator gets better at producing convincing fakes, and
the Discriminator gets better at spotting them — pushing both networks to
improve together.

---

## 2. Files

```
gan_project/
├── gan.py        # Full implementation: models, training loop, CLI
└── README.md      # This file
```

---

## 3. Code Walkthrough (`gan.py`)

### 3.1 Imports and setup
Standard PyTorch imports (`torch`, `torch.nn`, `torch.optim`), the
`torchvision` package for the MNIST dataset and image utilities, and
`matplotlib` (non-interactive `Agg` backend) for plotting the loss curve.
A fixed random seed (`SEED = 42`) is set for reproducibility.

### 3.2 `Generator` class
```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        ...
```
- **Input:** a noise vector `z` of size `latent_dim` (default 100), sampled
  from a standard normal distribution `N(0, 1)`.
- **Architecture:** a fully-connected (MLP) network that progressively
  upsamples the vector: `100 → 128 → 256 → 512 → 1024 → 784` (784 = 28×28
  pixels).
- Each hidden layer uses:
  - `nn.Linear` — a fully connected layer.
  - `nn.BatchNorm1d` — stabilizes training by normalizing activations
    (skipped on the first layer).
  - `nn.LeakyReLU(0.2)` — allows a small gradient when the unit is not
    active, which helps avoid "dead" neurons.
- **Output layer:** `nn.Tanh()` squashes pixel values into `[-1, 1]`,
  matching the normalization applied to the real MNIST images.
- The flat 784-length output is reshaped into a `(1, 28, 28)` image tensor.

### 3.3 `Discriminator` class
```python
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        ...
```
- **Input:** a `(1, 28, 28)` image, flattened to a 784-length vector.
- **Architecture:** `784 → 512 → 256 → 1`, using `LeakyReLU` activations
  in the hidden layers.
- **Output layer:** `nn.Sigmoid()` squashes the final scalar into `[0, 1]`,
  interpreted as "probability the image is real."

### 3.4 `train()` function
This is the core training loop:

1. **Data loading** — Downloads MNIST (if not already present) and applies
   `ToTensor()` + `Normalize([0.5], [0.5])` so pixel values lie in `[-1, 1]`
   (matching the Generator's `Tanh` output range).
2. **Model & optimizer setup** — Instantiates `Generator` and
   `Discriminator`, moves them to GPU if available, and creates two
   separate Adam optimizers (one per network) with the momentum
   parameters commonly used in GAN papers (`betas=(0.5, 0.999)`).
3. **Loss function** — `nn.BCELoss()` (Binary Cross-Entropy), used for
   both networks since this is a binary real-vs-fake classification
   problem.
4. **Per-batch training steps**, repeated for every batch of real images:
   - **Step A — Train the Generator:**
     - Sample random noise `z`.
     - Generate fake images: `gen_imgs = generator(z)`.
     - Compute how well the Discriminator was fooled: compare
       `discriminator(gen_imgs)` against the label `valid` (i.e., the
       Generator's loss is *low* when the Discriminator is fooled).
     - Backpropagate and update only the Generator's weights.
   - **Step B — Train the Discriminator:**
     - Compute loss on real images vs. label `valid`.
     - Compute loss on the (detached) generated images vs. label `fake`.
       `.detach()` is important — it stops gradients from also flowing
       back into the Generator during this step.
     - Average the two losses and update only the Discriminator's
       weights.
5. **Logging & checkpoints:**
   - Prints `D loss` / `G loss` every `log_interval` batches.
   - After each epoch, saves a 5×5 grid of freshly generated sample
     images (`generated_epoch_XX.png`) so you can visually track
     progress over time.
6. **After training:**
   - Saves final model weights: `generator.pth`, `discriminator.pth`.
   - Saves a plot of Generator/Discriminator loss over epochs
     (`loss_curve.png`).

### 3.5 Command-line arguments
The script is configurable via CLI flags (all optional, with sensible
defaults):

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 50 | Number of full passes over the dataset |
| `--batch_size` | 64 | Images per training batch |
| `--lr` | 0.0002 | Adam learning rate (standard GAN value) |
| `--b1`, `--b2` | 0.5, 0.999 | Adam momentum parameters |
| `--latent_dim` | 100 | Size of the input noise vector `z` |
| `--log_interval` | 200 | How often (in batches) to print loss |
| `--data_dir` | `./data` | Where MNIST is downloaded/cached |
| `--output_dir` | `./gan_output` | Where samples/models/plots are saved |

---

## 4. How to Run

### 4.1 Install dependencies
```bash
pip install torch torchvision matplotlib
```

### 4.2 Train the GAN
```bash
python gan.py
```

Train for a custom number of epochs / batch size:
```bash
python gan.py --epochs 100 --batch_size 128 --lr 0.0001
```

### 4.3 Outputs
After (or during) training, check the `./gan_output/` folder for:
- `generated_epoch_01.png` … `generated_epoch_NN.png` — sample grids
  showing how generated digits improve over training.
- `generator.pth` / `discriminator.pth` — trained model weights,
  loadable later with `model.load_state_dict(torch.load(path))`.
- `loss_curve.png` — Generator vs. Discriminator loss over epochs.

### 4.4 Generating new images after training
```python
import torch
from gan import Generator

generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load("gan_output/generator.pth"))
generator.eval()

z = torch.randn(16, 100)
with torch.no_grad():
    fake_images = generator(z)  # shape: (16, 1, 28, 28)
```

---

## 5. Tips for Better Results / Common Issues

- **Mode collapse** (Generator produces very similar/limited outputs):
  try lowering the learning rate, adding label smoothing, or using a
  different architecture (e.g., DCGAN with convolutional layers instead
  of fully-connected ones).
- **Discriminator overpowers the Generator** (D loss → 0 quickly): reduce
  the Discriminator's capacity, or train G more frequently than D.
- **Training is unstable / losses oscillate wildly:** this is normal
  for GANs to some degree — watch the *generated image quality* over
  epochs rather than relying solely on the loss values, since GAN losses
  don't monotonically decrease the way typical supervised losses do.
- **Want higher quality images?** Swap the fully-connected layers for
  convolutional ones (a "DCGAN"), which generally produces sharper,
  more coherent images for image data.

---

## 6. Key Takeaways

- A GAN is trained as a **two-player minimax game** between a Generator
  and a Discriminator.
- The Generator never sees real images directly — it only learns from
  the gradient signal passed back through the Discriminator.
- Careful details matter: using `.detach()` when training D on fake
  images, normalizing images to match the Generator's output activation
  (`Tanh` → `[-1, 1]`), and using separate optimizers per network.
- This vanilla, fully-connected GAN is a good learning example; for
  production-quality image generation, convolutional variants (DCGAN)
  or more advanced architectures (WGAN-GP, StyleGAN, etc.) are typically
  used.
