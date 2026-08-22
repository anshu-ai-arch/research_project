import os
import wfdb
import yaml
from pathlib import Path


def download_mitdb(config_path: str = "config.yaml"):
    """Downloads selected recordings from MIT-BIH Arrhythmia Database (PhysioNet)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Gather all requested recordings (DS1 train + DS2 test)
    ds1 = config["split"]["ds1_train_recordings"]
    ds2 = config["split"]["ds2_test_recordings"]
    all_recordings = sorted(list(set(ds1 + ds2)))

    print(f"[*] Downloading MIT-BIH records ({len(all_recordings)} files) to '{data_dir}'...")

    downloaded = 0
    for rec in all_recordings:
        dat_file = data_dir / f"{rec}.dat"
        hea_file = data_dir / f"{rec}.hea"
        atr_file = data_dir / f"{rec}.atr"

        if dat_file.exists() and hea_file.exists() and atr_file.exists():
            downloaded += 1
            continue

        print(f" -> Downloading record {rec} from PhysioNet mitdb...")
        try:
            wfdb.dl_database(
                db_dir="mitdb",
                dl_dir=str(data_dir),
                records=[rec],
                overwrite=False
            )
            downloaded += 1
        except Exception as e:
            print(f" [!] Error downloading record {rec}: {e}")

    print(f"[*] Download complete. {downloaded}/{len(all_recordings)} records available.")
    return data_dir


if __name__ == "__main__":
    download_mitdb()
