
import sys
import argparse
import pandas as pd
from pathlib import Path
import subprocess as sp
from datetime import datetime, timezone


def command_runner(cmd: list[str], cwd: str | Path, log_file: str | Path, verbose: bool = False):
    """Run command and save outputs to log file."""
    for i,c in enumerate(cmd):
        cmd[i] = str(c)
    
    with open(log_file, "a") as f:
        f.write("Time: " + str(datetime.now(timezone.utc)) + "\n")
        f.write("Command: " + " ".join(cmd) + "\n\n")
        with sp.Popen(
            cmd,  # noqa: S603
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf-8",
            cwd=cwd,
        ) as p:
            while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
                if verbose:
                    print(buffered_output, end="", flush=True)
                f.write(buffered_output)
                f.flush()

def run(
    csv_path: str,
    pdb_dir: str,
    out_dir: str,
    model_weight_dir: str,
    model_name: str,
    mpnn_home: str,
):
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    pdb_dir = Path(pdb_dir)
    print(out_dir)
    for _,r in df.iterrows():
        name = r["pdb_name"]
        pdb = str(pdb_dir / f"{name}.pdb")
        vl = r.get("vl_chain", None)
        if not pd.isna(vl):
            chain = f"{r['vh_chain']} {r['vl_chain']}"
        else:
            chain = r["vh_chain"]
        out_dir_i = out_dir / name
        out_dir_i.mkdir(exist_ok=True)
        cmd = [
            "python", f"{mpnn_home}/protein_mpnn_run.py",
            "--pdb_path", pdb,
            "--pdb_path_chains", chain,
            "--out_folder", out_dir_i,
            "--num_seq_per_target", 100,
            "--sampling_temp", 0.2,
            "--seed", 37,
            "--batch_size", 1,
            "--path_to_model_weights", model_weight_dir,
            "--model_name", model_name,
        ]
        command_runner(cmd, cwd=str(out_dir_i), log_file=out_dir_i / f"{name}.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run mpnn design")
    parser.add_argument('--csv_path', type=str, help='csv path, must contains "pdb_name"')
    parser.add_argument('--pdb_dir', type=str, help='dir where the input pdb files saved, file name must same with pdb_name in csv_path')
    parser.add_argument('--weight_dir', type=str, help='parent dir of mpnn checkpoint files, like v_48_002.pt, abmpnn.pt ...')
    parser.add_argument('--mpnn_home', type=str, help='dir to mpnn repo')
    parser.add_argument('--model_name', type=str, help='name of the checkpoint file', default="abmpnn")
    parser.add_argument('--out_dir', type=str, help='save dir')

    args = parser.parse_args()

    run(args.csv_path, args.pdb_dir, args.out_dir, args.weight_dir, args.model_name, args.mpnn_home)
