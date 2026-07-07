import os
import tarfile
import json
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cache_dir = r"C:\Users\Lenovo\.cache\huggingface\hub\datasets--0x3--vocal-bursts\snapshots\6aac12ac8bc6f26229ba6ab18b7c591657c43ff0\data"
output_dir = os.path.join(BASE_DIR, "data", "raw", "vocal_bursts")

TAR_TO_CLASS = {
    "Cough.tar.gz": ("Cough", 0),
    "Crying.tar.gz": ("Crying", 1),
    "Moan.tar.gz": ("Moan", 2),
    "Pant.tar.gz": ("Pant", 3),
    "Breath.tar.gz": ("Pant", 3),
}

for class_name in set(v[0] for v in TAR_TO_CLASS.values()):
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

for tar_name, (class_name, label_id) in TAR_TO_CLASS.items():
    tar_path = os.path.join(cache_dir, tar_name)
    if not os.path.exists(tar_path):
        print(f"Skipping {tar_name} - not found")
        continue

    class_dir = os.path.join(output_dir, class_name)
    existing = set(os.listdir(class_dir))
    count = 0

    with tarfile.open(tar_path, "r:gz") as t:
        members = [m for m in t.getmembers() if m.name.endswith(".flac")]
        print(f"{tar_name}: {len(members)} audio files")

        for member in members:
            basename = os.path.basename(member.name)
            out_fname = f"{class_name}_{basename}"
            if out_fname in existing:
                continue

            try:
                f = t.extractfile(member)
                out_path = os.path.join(class_dir, out_fname)
                with open(out_path, "wb") as out_f:
                    out_f.write(f.read())
                count += 1
            except Exception as e:
                pass

    print(f"  Saved {count} new files to {class_name}/")

print("\nFinal counts:")
for class_name in set(v[0] for v in TAR_TO_CLASS.values()):
    class_dir = os.path.join(output_dir, class_name)
    n = len([f for f in os.listdir(class_dir) if f.endswith(('.wav', '.flac'))])
    print(f"  {class_name}: {n}")
