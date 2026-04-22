import random
from pathlib import Path


def prune_images(target_path, target_count=2000, dry_run=True):
    # 1. Define valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 2. Collect all image files
    path = Path(target_path)
    all_files = [f for f in path.iterdir() if f.suffix.lower() in valid_extensions]

    current_count = len(all_files)
    print(f"Total images found: {current_count}")

    if current_count <= target_count:
        print("Folder already has 2000 or fewer images. No action needed.")
        return

    # 3. Calculate how many to remove
    num_to_remove = current_count - target_count
    print(f"Preparing to remove {num_to_remove} random images...")

    # 4. Shuffle and select the "victims"
    random.shuffle(all_files)
    to_delete = all_files[:num_to_remove]

    # 5. Execution
    for i, file_path in enumerate(to_delete):
        if dry_run:
            print(f"[DRY RUN] Would delete: {file_path.name}")
        else:
            try:
                file_path.unlink()
                print(f"[{i + 1}/{num_to_remove}] Deleted: {file_path.name}", end="\r")
            except Exception as e:
                print(f"\nError deleting {file_path.name}: {e}")

    if dry_run:
        print("\n" + "=" * 30)
        print("DRY RUN COMPLETE. No files were actually deleted.")
        print("Set dry_run=False in the script to perform the actual removal.")
        print("=" * 30)
    else:
        print(f"\nDone! Target count of {target_count} images reached.")


if __name__ == "__main__":
    # Change this to your actual folder path
    MY_PATH = "dataset/Plains"

    # Run once with dry_run=True first to be safe!
    prune_images(MY_PATH, target_count=2000, dry_run=True)
