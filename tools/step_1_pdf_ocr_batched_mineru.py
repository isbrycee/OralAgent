#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

def detect_lang_from_dirname(dirname: str) -> Optional[Tuple[str, str]]:
    """Return (mineru_lang_code, tag) based on folder name; None if not CN/EN."""
    name = dirname.upper()
    if "CH" in name:
        return ("ch", "CN")
    if "EN" in name:
        return ("en", "EN")
    return None

def find_pdf_jobs(base_path: Path) -> List[Tuple[Path, str, Path]]:
    """
    Scan the base directory structure and collect jobs.
    Structure:
      base_path/
        subdir_A/
          ... folder contains 'CN' ...
          ... folder contains 'EN' ...
        subdir_B/
          ...
    Returns a list of tuples: (pdf_path, lang_code, rel_output_subdir)
      - rel_output_subdir is used to keep a clear structure under OUTPUT_DIR to avoid name collisions.
    """
    jobs = []
    for first_level in sorted([p for p in base_path.iterdir() if p.is_dir()]):
        # Inside each subfolder, look for folders whose names contain CN or EN (case-insensitive).
        for lang_dir in sorted([p for p in first_level.iterdir() if p.is_dir()]):
            lang_info = detect_lang_from_dirname(lang_dir.name)
            if not lang_info:
                continue
            lang_code, lang_tag = lang_info
            # Collect PDFs directly under this CN/EN directory (non-recursive)
            pdfs = sorted([p for p in lang_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
            # If you need recursive search, replace the above line with:
            # pdfs = sorted([p for p in lang_dir.rglob("*.pdf") if p.is_file()])
            for pdf in pdfs:
                rel_output_subdir =  Path(first_level.name) / lang_dir.name
                jobs.append((pdf, lang_code, rel_output_subdir))
    return jobs

def process_pdf(pdf_path: Path, lang_code: str, output_dir: Path, device: str, quiet: bool = False) -> bool:
    """Process single PDF file with mineru. When quiet=True, no print (for parallel workers)."""
    if not quiet:
        print(f"Processing: {pdf_path}")
        print(f"Language: {lang_code}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "--output", str(output_dir),
        "--ocr",
        "--lang", lang_code,  # "ch" for CN; "en" for EN
        "--device", device
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not quiet:
            print(f"✓ {pdf_path.name} processing completed")
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"✗ {pdf_path.name} processing failed: {e}")
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        if not quiet:
            print(f"✗ {pdf_path.name} exception occurred: {e}")
        return False
    finally:
        if not quiet:
            print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)


def _process_one_job(args: Tuple) -> Tuple[bool, Path]:
    """Worker entry: (pdf_path, lang_code, output_dir, device) -> (success, pdf_path)."""
    pdf_path, lang_code, output_dir, device = args
    ok = process_pdf(pdf_path, lang_code, output_dir, device, quiet=True)
    return (ok, pdf_path)

def main():
    parser = argparse.ArgumentParser(description="Batch process PDFs under CN/EN folders with mineru.")
    parser.add_argument("-r", "--root", type=str, default="/data/OralGPT/OralGPT-text-corpus", help="Data root directory containing subfolders with CN/EN folders.")
    parser.add_argument("-o", "--output", type=str, default="/data/OralGPT/OralGPT-text-corpus-processed", help="Output directory to save all processed results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for mineru (e.g., cuda:0 or cpu).")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Number of parallel workers (default: CPU count).")
    args = parser.parse_args()

    base_path = Path(args.root).expanduser().resolve()
    output_base = Path(args.output).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    if not base_path.exists() or not base_path.is_dir():
        print(f"Root path does not exist or is not a directory: {base_path}")
        return

    print(f"Scanning root: {base_path}")
    jobs = find_pdf_jobs(base_path)
    total_files = len(jobs)

    if total_files == 0:
        print("No PDF files found under CN/EN folders.")
        return

    n_workers = args.jobs if args.jobs is not None else (os.cpu_count() or 4)
    print(f"Found {total_files} PDF file(s) to process.")
    print(f"Using {n_workers} parallel worker(s).")
    print("=" * 60)

    successful = 0
    failed = 0
    start_time = time.time()

    job_args = []
    for pdf_path, lang_code, rel_output_subdir in jobs:
        out_dir = output_base / rel_output_subdir
        job_args.append((pdf_path, lang_code, out_dir, args.device))

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_process_one_job, ja): i for i, ja in enumerate(job_args)}
        for future in as_completed(future_to_idx):
            completed += 1
            idx = future_to_idx[future]
            pdf_path = jobs[idx][0]
            try:
                ok, _ = future.result()
                if ok:
                    successful += 1
                    status = "OK"
                else:
                    failed += 1
                    status = "FAILED"
            except Exception as e:
                failed += 1
                status = f"ERROR: {e}"
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_files - completed) if completed < total_files else 0
            print(f"[{completed}/{total_files}] {pdf_path.name} -> {status} (elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)", flush=True)

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("Processing summary:")
    print(f"Root: {base_path}")
    print(f"Output base: {output_base}")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()