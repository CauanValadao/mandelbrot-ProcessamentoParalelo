#!/usr/bin/env python3
"""
Compara byte a byte o binario (int32, row-major) da versao serial com o de cada
estrategia MPI, para varios numeros de processos (inclusive P que nao divide a altura).

Uso: python verificar_corretude_mpi.py [--res 1000] [--preset cavalos] [--procs 2 3 4 7]
"""
import argparse
import filecmp
import os
import shlex
import subprocess
import sys

MPIEXEC = os.environ.get("MPIEXEC", "mpiexec")
EXTRA = shlex.split(os.environ.get("MPIEXEC_EXTRA", ""))
EXE = os.path.abspath("./mandelbrot_mpi.exe")
BIN_SEQ = "saida/mandelbrot_sequencial.bin"
BIN_PAR = "saida/mandelbrot_mpi.bin"

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=1000)
ap.add_argument("--preset", default="cavalos")
ap.add_argument("--procs", type=int, nargs="+", default=[2, 3, 4, 7])
args = ap.parse_args()

os.makedirs("saida", exist_ok=True)


def rodar(procs, extra):
    base = ["--preset", args.preset, "--res", str(args.res), "--salvar", "--runs", "1",
            "--csv", "saida/_verificacao.csv"]
    cmd = [MPIEXEC, *EXTRA, "-n", str(procs), EXE, *base, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Erro ao executar:", " ".join(cmd), "\n", r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)


print("Gerando referencia serial...")
rodar(1, ["--modo", "seq"])

falhas = 0
for est, chunk in (("block", 0), ("cyclic", 1), ("cyclic", 16), ("master", 1), ("master", 16)):
    for p in args.procs:
        rodar(p, ["--modo", "mpi", "--estrategia", est, "--chunk", str(chunk)])
        ok = filecmp.cmp(BIN_SEQ, BIN_PAR, shallow=False)
        falhas += (not ok)
        print(f"{'OK  ' if ok else 'FALHA'} {est:7s} chunk={chunk:<3} P={p}")

print("\n" + ("SUCESSO: todas as saidas sao identicas a versao serial." if not falhas else
              f"{falhas} combinacao(oes) divergiram - veja a tolerancia da Secao 5.5 (0,01% dos pixels, diferenca <= 1)."))
sys.exit(1 if falhas else 0)