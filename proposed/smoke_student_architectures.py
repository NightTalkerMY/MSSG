"""Fast forward/backward smoke test for every retrained student architecture."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proposed"))

from student_model import ARCHITECTURES, CausalStudentASD  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for architecture in ARCHITECTURES:
        model = CausalStudentASD(architecture=architecture).to(device).train()
        visual = torch.randn(1, 5, 50, 896, device=device)
        audio = torch.randn(1, 5, 50, 128, device=device)
        spatial = torch.randn(1, 5, 50, 4, device=device)
        mask = torch.ones(1, 5, 50, device=device)
        labels = torch.randint(0, 2, (1, 5, 50), device=device)
        logits, features = model(visual, audio, spatial, mask)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        loss = loss + 0.01 * features.square().mean()
        loss.backward()
        print(
            architecture,
            f"parameters={sum(parameter.numel() for parameter in model.parameters())}",
            f"logits={tuple(logits.shape)}",
            f"features={tuple(features.shape)}",
            f"loss={loss.item():.6f}",
        )

    retained = ROOT / "realtime" / "experiments" / "ce_mse_alpha_08_seed_42" / "best_ava.pt"
    if retained.is_file():
        model = CausalStudentASD(architecture="full")
        model.load_state_dict(torch.load(retained, map_location="cpu"))
        print("Existing full checkpoint compatibility: passed")
    else:
        print("Existing full checkpoint compatibility: skipped (checkpoint not present)")


if __name__ == "__main__":
    main()
