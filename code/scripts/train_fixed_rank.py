"""Launch unchanged native fine-tuning with explicitly authorized fixed ranks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--policy", required=True)
    args, native_args = parser.parse_known_args()
    policy = json.loads((ROOT / args.policy).read_text())
    for name, expected in policy["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + name)
    if native_args[native_args.index("--stage") + 1] != "finetune":
        raise ValueError("Rank freeze applies to fine-tuning only")
    if native_args[native_args.index("--run") + 1] != policy["run"]:
        raise ValueError("Fixed-rank run mismatch")
    config = ROOT / native_args[native_args.index("--config") + 1]
    if hashlib.sha256(config.read_bytes()).hexdigest() != policy["config_sha256"]:
        raise ValueError("Fixed-rank run configuration changed")
    import train
    from fixed_rank_training import install
    install(train, policy["fixed_B_rank_by_layer"])
    print(json.dumps({"rank_policy": "fixed", "run": policy["run"],
                      "B_rank": sorted(set(policy["fixed_B_rank_by_layer"].values())),
                      "C_rank": policy["selected_C_rank"], "weights_remain_trainable": True}), flush=True)
    sys.argv = [str(ROOT / "train.py"), *native_args]
    train.main()


if __name__ == "__main__":
    main()
