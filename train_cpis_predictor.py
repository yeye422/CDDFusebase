"""Train the paired and shuffled CPIS R1.1 prediction heads."""

import argparse
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from cpis_r11 import (CONFIG, ROOT, SOURCES, features, frozen_e2, heads, load_images,
                      protocol, read_json, register, runtime, sha256_file, write_json)
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT


def train(model, manifest, config, device):
    scenes = manifest['splits']['train']
    batch_size = config['batch_size']
    if batch_size < 2 or len(scenes) % batch_size:
        raise ValueError('Batch size must divide train1024 and allow a within-batch derangement')
    torch.manual_seed(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    channels = 3 * model.lwpd.channels
    paired = heads(channels, config['hidden_channels']).to(device)
    shuffled = heads(channels, config['hidden_channels']).to(device)
    shuffled.load_state_dict(paired.state_dict())
    groups = (paired, shuffled)
    optimizers = [torch.optim.AdamW(group.parameters(), lr=config['learning_rate']) for group in groups]
    sums = torch.zeros(len(SOURCES), channels, dtype=torch.float64, device=device)
    count = 0
    history = []
    for epoch in range(config['epochs']):
        order = torch.randperm(len(scenes), generator=generator).tolist()
        epoch_losses = torch.zeros(len(groups), len(SOURCES), device=device)
        for start in range(0, len(scenes), batch_size):
            batch = [scenes[i] for i in order[start:start + batch_size]]
            values = features(model, load_images(manifest, batch, device))
            if epoch == 0:
                sums += torch.stack([value.sum(dim=(0, 2, 3), dtype=torch.float64) for value in values])
                count += values[0].shape[0] * values[0].shape[2] * values[0].shape[3]
            # Random batch order changes the partners each epoch. Each scene appears
            # exactly once per direction as a target and once as a conditioning input.
            inputs = (values[::-1], tuple(value.roll(1, dims=0) for value in values[::-1]))
            for i, (group, optimizer, other) in enumerate(zip(groups, optimizers, inputs)):
                optimizer.zero_grad(set_to_none=True)
                losses = torch.stack([F.mse_loss(head(x), target) for head, x, target in zip(group, other, values)])
                losses.sum().backward()
                optimizer.step()
                epoch_losses[i] += losses.detach()
        means = (epoch_losses / (len(scenes) // batch_size)).tolist()
        row = {f'{method}_{source}_mse': means[i][j]
               for i, method in enumerate(('p1', 'p2')) for j, source in enumerate(SOURCES)}
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError(f'Nonfinite prediction loss at epoch {epoch + 1}')
        history.append(row)
        print(f'Epoch {epoch + 1}/{config["epochs"]}: {row}', flush=True)
    return dict(paired=paired.state_dict(), shuffled=shuffled.state_dict(),
                p0_means=(sums / count).float().cpu(), channels=channels), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output_dir.resolve().relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f'Choose a new result directory: {output}')
    config, manifest = read_json(args.config), protocol()
    if min(config['epochs'], config['hidden_channels'], config['learning_rate']) <= 0:
        raise ValueError('Epochs, hidden width and learning rate must be positive')
    device = runtime(args.device)
    e2_checkpoint = ORIGINAL_LOSS_PILOT.final
    e2_hash = sha256_file(e2_checkpoint)
    model = frozen_e2(device)
    output.mkdir(parents=True, exist_ok=False)
    split_path = output / 'split_manifest.json'
    write_json(split_path, manifest)
    state, history = train(model, manifest, config, device)
    checkpoint = output / 'predictors.pth'
    torch.save(dict(**state, config=config, e2_checkpoint_sha256=e2_hash,
                    split_manifest_sha256=sha256_file(split_path)), checkpoint)
    write_json(output / 'summary.json', dict(epoch_mse=history, registry_metrics=history[-1]))
    write_json(output / 'run_metadata.json', dict(
        config=config, device=str(device), torch=str(torch.__version__), cuda=torch.version.cuda,
        checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
        e2_checkpoint=e2_checkpoint.as_posix(), e2_checkpoint_sha256=e2_hash,
        split_manifest=split_path.as_posix(), split_manifest_sha256=sha256_file(split_path),
        model_source_sha256=sha256_file(ROOT / 'cpis_r11.py'), script_sha256=sha256_file(Path(__file__))))
    register(output, 'train', split_path, checkpoint, history[-1], e2_checkpoint)


if __name__ == '__main__':
    main()
