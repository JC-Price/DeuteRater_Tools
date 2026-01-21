
#!/usr/bin/env python3
import os
import hashlib
import time
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label
from concurrent.futures import ThreadPoolExecutor

def hash_py_file(file_path: Path, root: Path) -> bytes:
    """Hash a single .py file (contents + relative path)."""
    sha = hashlib.sha256()
    rel_path = file_path.relative_to(root).as_posix()
    sha.update(rel_path.encode())
    with file_path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.digest()

def compute_py_files_fingerprint(root: Path) -> bytes:
    """Compute combined hash of selected .py files."""
    root = root.resolve()

    EXCLUDED_DIRS = {"DeuteRater_python", "shared_python"}
    ROOT_ALLOWED_FILES = {"Lipid_Kinetics_Workflow2.py"}

    py_files = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        # Remove excluded directories from traversal
        dirnames[:] = sorted(
            d for d in dirnames if d not in EXCLUDED_DIRS
        )

        # Case 1: root directory → only allow specific files
        if dirpath == root:
            for name in sorted(filenames):
                if name in ROOT_ALLOWED_FILES:
                    py_files.append(dirpath / name)

        # Case 2: subdirectories → include all .py files
        else:
            for name in sorted(filenames):
                if name.endswith(".py"):
                    py_files.append(dirpath / name)

    combined_sha = hashlib.sha256()
    with ThreadPoolExecutor() as executor:
        for file_hash in executor.map(lambda f: hash_py_file(f, root), py_files):
            combined_sha.update(file_hash)

    return combined_sha.digest()


def derive_key_from_fingerprint(fp: bytes) -> str:
    """Return a hex representation of the derived key."""
    return hashlib.sha256(fp).hexdigest()


def show_loading_popup(mode="normal"):
    popup = Toplevel()
    popup.title("Processing")
    popup.geometry("300x100")
    if mode == "normal":
        text = "Generating Security Key...\nLoading..."
    else:  # read mode
        text = "Welcome to Kinetic Lipidomics...\nLoading..."
    Label(popup, text=text, font=("Arial", 12)).pack(expand=True)
    popup.update()
    return popup


def normal_mode():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select a folder to analyze")
    if not folder:
        messagebox.showinfo("Cancelled", "No folder selected.")
        return

    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        messagebox.showerror("Error", "Invalid folder selected.")
        return

    loading_popup = show_loading_popup(mode = "normal")
    start_time = time.time()
    fingerprint = compute_py_files_fingerprint(folder_path)
    key = derive_key_from_fingerprint(fingerprint)
    elapsed = time.time() - start_time
    loading_popup.destroy()

    misc_folder = folder_path / "misc"
    misc_folder.mkdir(exist_ok=True)
    key_file = misc_folder / "key.txt"
    key_file.write_text(f"Derived Key: {key}\nTime Taken: {elapsed:.4f} seconds", encoding="utf-8")

    messagebox.showinfo("Success", f"Key saved to:\n{key_file}")

def read_mode(path: str) -> bool:
    folder_path = Path(path)
    if not folder_path.exists() or not folder_path.is_dir():
        return False

    # Create a hidden root for popup
    root = tk.Tk()
    root.withdraw()
    loading_popup = show_loading_popup(mode = "read")

    start_time = time.time()
    fingerprint = compute_py_files_fingerprint(folder_path)
    key = derive_key_from_fingerprint(fingerprint)
    elapsed = time.time() - start_time
    loading_popup.destroy()

    # Check misc/key.txt
    key_file = folder_path / "misc" / "key.txt"
    if not key_file.exists():
        return False

    stored_key = None
    with key_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Derived Key:"):
                stored_key = line.split(":", 1)[1].strip()
                break

    return stored_key == key

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments → Normal Mode
        normal_mode()
    elif len(sys.argv) == 3 and sys.argv[1] == "--read":
        # Read Mode
        result = read_mode(sys.argv[2])
        print(result)
    else:
        print("Usage:")
        print("  python program.py            # Normal Mode")
        print("  python program.py --read <path>  # Read Mode")
