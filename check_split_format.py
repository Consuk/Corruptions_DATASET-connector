from pathlib import Path

def check_txt_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    print(f"\n🧾 Revisando: {file_path.name} ({len(lines)} líneas)")
    prev_seq = None
    frame_ids = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            print(f"[!] Formato incorrecto: {line.strip()}")
            return

        path, frame_id, label = parts
        assert label == "l", f"[!] Label no es 'l': {label}"

        # Extraer secuencia (rectifiedXX)
        seq = path.split("/")[0]
        frame_ids.append((seq, int(frame_id)))

    # Verificar secuencialidad por secuencia
    from collections import defaultdict
    seq_frames = defaultdict(list)
    for seq, fid in frame_ids:
        seq_frames[seq].append(fid)

    for seq, fids in seq_frames.items():
        diffs = [b - a for a, b in zip(fids, fids[1:])]
        if any(d != 1 for d in diffs):
            print(f"[!] Secuencia desordenada o discontinua en {seq}")
        else:
            print(f"[✓] {seq}: Secuencia continua ({len(fids)} frames)")

if __name__ == "__main__":
    split_dir = Path("/workspace/datasets/hamlyn/splits")
    for name in ["train", "val", "test"]:
        check_txt_format(split_dir / f"{name}_files_hamlyn.txt")
