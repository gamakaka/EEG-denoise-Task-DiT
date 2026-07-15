# TaskEEGDiT

TaskEEGDiT is a diffusion-based EEG denoising project. The main model uses a DDPM framework, with a TimeMixer-based noise predictor for estimating Gaussian noise during the reverse denoising process.

The project supports both unconditional denoising and task-conditioned denoising. In the conditional setting, task-wise spectral prototypes can be used as compact task-level guidance for sleep EEG and motor imagery EEG.

## Features

- DDPM-based EEG denoising.
- TimeMixer noise prediction network.
- Optional task-wise spectral prototype conditioning.
- Optional auxiliary content and spectral style losses.
- Unified YAML configuration for training and testing.
- Built-in evaluation metrics including CC, SNR, time-domain RRMSE, and frequency-domain RRMSE.

## Project Structure

```text
config/
  config.yaml                  Main experiment configuration.

src/
  data/                        Dataset loaders.
  models/                      DDPM, model factory, and noise predictors.
  models/noise_predictors/     TimeMixer-based noise prediction network.
  utils/                       Config, logging, checkpoint, and seed utilities.
  evaluate.py                  Evaluation metrics and test-time utilities.
  train_loop.py                Training and validation loop.

timemixer/
  config.py                    TimeMixer configuration.
  exp/, layers/, models/, utils/
                               TimeMixer implementation used by the noise predictor.

Data_Preparation/
  Dataset preparation utilities retained for future dataset processing.

train.py                       Training entry point.
test.py                        Testing entry point.
requirements.txt               Python dependencies.
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The project is developed with PyTorch. Use a CUDA-enabled PyTorch installation if GPU training or inference is required.

## Configuration

The main configuration file is:

```text
config/config.yaml
```

Important sections:

- `project`: random seed and device.
- `data`: train, validation, and test data paths.
- `model`: TimeMixer settings and task-conditioning options.
- `train`: optimizer, batch size, epochs, and auxiliary loss weights.
- `eval`: sampler type and DDIM sampling parameters.
- `diffusion`: DDPM noise schedule.
- `output`: output roots for checkpoints, logs, and evaluation results.

Task conditioning is controlled by:

```yaml
model:
  task_condition: true
  spectral_prototype_path: data/spectral_prototypes.pt
```

Set `task_condition: false` for unconditional denoising.

Auxiliary losses are controlled by:

```yaml
train:
  lambda_content: 0.005
  lambda_style: 0.001
  encoder_ckpt_path: path/to/encoder_checkpoint.pt
```

When an auxiliary loss weight is set to `0`, that loss is skipped. When both weights are `0`, training reduces to the original DDPM objective.

## Training

Run training with the default configuration:

```bash
python train.py
```

Common command-line overrides:

```bash
python train.py --device cuda:0
python train.py --conditional
python train.py --no-conditional
python train.py --lambda-content 3e-5 --lambda-style 350
```

To run testing immediately after training:

```bash
python train.py --run-test-after-train
```

## Testing

Run testing with the default configuration:

```bash
python test.py
```

Use DDIM sampling:

```bash
python test.py --sampler ddim --ddim-num-steps 20 --ddim-eta 0.0
```

Test a conditional checkpoint trained with specific auxiliary loss weights:

```bash
python test.py \
  --conditional \
  --lambda-content 3e-5 \
  --lambda-style 350 \
  --sampler ddim \
  --ddim-num-steps 20
```

The test script evaluates both configured test sets sequentially.

## Spectral Prototype Conditioning

TaskEEGDiT can use task-wise spectral prototypes as condition vectors. A prototype file should contain two class prototypes with shape `[2, F]`, where each row is a normalized spectral distribution.

Conceptually:

- class `0`: sleep EEG prototype;
- class `1`: motor imagery EEG prototype.

During conditional inference, the label selects the corresponding prototype, which is encoded and fused into the TimeMixer noise predictor through a residual conditioning path.

## Data Format

The dataset loader expects NumPy dictionary files containing at least:

```text
clean_eeg
noisy_eeg
label
```

EEG signals should be shaped as `[N, 1, 512]` or `[N, 512]`. Labels should be integer class IDs.

## Notes

- Keep data paths, checkpoint paths, and output roots consistent with `config/config.yaml`.
- For conditional testing, the auxiliary loss weights passed to `test.py` should match the weights used when training the checkpoint.
- Dataset preparation utilities are retained for future data processing but are not required when preprocessed dataset files are already available.
