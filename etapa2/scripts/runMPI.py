#!/usr/bin/env python3
"""
Orquestrador MPI - DEC107 (Etapa 2).

Casos:
  V  corretude (--comparar) de cada estrategia, incluindo P que nao divide a altura
  A  regiao padrao       - strong scaling, 3 estrategias de distribuicao
  B  cavalos-marinhos    - strong scaling + desbalanceamento (5 configuracoes)
  C  escalabilidade fraca (resolucao cresce com o numero de processos)
  D  efeito do chunk (P maximo, cavalos-marinhos): cyclic x master
  baselines: tempo serial (mpiexec -n 1 --modo seq) usado para Speedup/Eficiencia

Variaveis de ambiente opcionais:
  MPIEXEC        (padrao 'mpiexec')
  MPIEXEC_EXTRA  flags extras, ex.: "--oversubscribe" ou "--bind-to core" (OpenMPI)
  MPICC          compilador MPI (padrao: mpicc do PATH; senao gcc + MS-MPI via MSMPI_INC/MSMPI_LIB64)
"""
import argparse
import csv
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time

os.makedirs("saida", exist_ok=True)

SRC_FILE = "src/mandelbrot_mpi.c"
BIN_FILE = os.path.abspath("./mandelbrot_mpi.exe")
PROGRAM_CSV = "saida/benchmark_mpi.csv"
MASTER_CSV = "saida/resultados_mpi.csv"

MPIEXEC = os.environ.get("MPIEXEC", "mpiexec")
MPIEXEC_EXTRA = shlex.split(os.environ.get("MPIEXEC_EXTRA", ""))
N_CORES = os.cpu_count() or 4

PROCS = [p for p in (1, 2, 4, 8, 16) if p <= N_CORES]
P_MAX = max(PROCS)

# (estrategia, chunk)
STRATS_A = [("block", 0), ("cyclic", 1), ("master", 1)]
STRATS_B = [("block", 0), ("cyclic", 1), ("cyclic", 16), ("master", 1), ("master", 16)]
STRATS_C = [("cyclic", 1), ("master", 1)]
CHUNKS_D = [1, 2, 4, 16, 64, 256, 1024]
WEAK_PAIRS = [(2, 5793), (4, 8192), (8, 11585), (16, 16384)]
WEAK_PAIRS = [(p, r) for p, r in WEAK_PAIRS if p <= N_CORES]

REGIAO_PADRAO = dict(preset="vista", max_iter=1000)
REGIAO_CAVALOS = dict(preset="cavalos", max_iter=5000)

BASELINES = {
    "A_padrao_4096":  dict(REGIAO_PADRAO, res=4096),
    "B_cavalos_4096": dict(REGIAO_CAVALOS, res=4096),
    **{f"C_weak_{r}": dict(REGIAO_PADRAO, res=r) for _, r in WEAK_PAIRS},
}

N_SEQ_REPETICOES = 10
RUN_COUNT_SEQ = 3
RUN_COUNT_PAR = 30


def compilar():
    mpicc = os.environ.get("MPICC") or shutil.which("mpicc")
    if mpicc:
        cmd = [mpicc, "-O3", "-march=native", "-o", BIN_FILE, SRC_FILE, "-lm"]
    elif os.environ.get("MSMPI_INC") and os.environ.get("MSMPI_LIB64"):
        inc, lib = os.environ["MSMPI_INC"], os.environ["MSMPI_LIB64"]
        cmd = ["gcc", "-O3", "-march=native", f"-I{inc}", f"-I{inc}/x64", f"-L{lib}",
               "-o", BIN_FILE, SRC_FILE, "-lmsmpi", "-lm"]
    else:
        print("Nenhum compilador MPI encontrado (defina MPICC ou instale mpicc/MS-MPI SDK).")
        sys.exit(1)
    print("Compilando:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Erro de compilacao:\n", r.stderr)
        sys.exit(1)
    print("Compilado com sucesso.\n")


def registrar_ambiente():
    """Registra hardware/SO/compilador/MPI (exigido na Secao 6, item 9 e Secao 11 do enunciado)."""
    def cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        except Exception:
            return "indisponivel"
    linhas = [
        f"SO: {platform.platform()}",
        f"CPU: {platform.processor()}",
        f"Nucleos logicos: {N_CORES}",
        f"Python: {platform.python_version()}",
        f"gcc: {cmd(['gcc', '--version'])}",
        f"mpiexec: {cmd([MPIEXEC, '--version'])}",
        f"Flags: -O3 -march=native (double, sem simetria)",
        f"PROCS={PROCS} WEAK_PAIRS={WEAK_PAIRS}",
        f"RUN_COUNT_PAR={RUN_COUNT_PAR} N_SEQ_REPETICOES={N_SEQ_REPETICOES} RUN_COUNT_SEQ={RUN_COUNT_SEQ}",
    ]
    with open("saida/ambiente.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("Ambiente registrado em saida/ambiente.txt")


def _ler_linhas():
    if not os.path.exists(PROGRAM_CSV):
        return []
    with open(PROGRAM_CSV, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rodar_um(nome, cfg, procs=1, estrategia=None, chunk=0, runs=1, modo="mpi",
             comparar=False, timeout=7200):
    args = ["--preset", cfg["preset"], "--res", str(cfg["res"]), "--max-iter", str(cfg["max_iter"]),
            "--nome", nome, "--runs", str(runs), "--modo", modo]
    if modo == "mpi":
        args += ["--estrategia", estrategia, "--chunk", str(chunk)]
    if comparar:
        args += ["--comparar"]
    comando = [MPIEXEC, *MPIEXEC_EXTRA, "-n", str(procs), BIN_FILE, *args]

    antes = len(_ler_linhas())
    t0 = time.time()
    r = subprocess.run(comando, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    if r.returncode != 0:
        print("### Falha na execucao ###\nCMD:", " ".join(comando), "\nSTDOUT:\n", r.stdout[-2000:],
              "\nSTDERR:\n", r.stderr[-2000:])
        raise RuntimeError(f"Processo terminou com codigo {r.returncode}")
    linhas = _ler_linhas()
    if len(linhas) <= antes:
        print(r.stdout[-2000:])
        raise RuntimeError("Nenhuma linha nova em " + PROGRAM_CSV)
    print(f"  [{modo:3s}] P={procs:<3} {estrategia or '-':<7} C={chunk:<5} "
          f"res={cfg['res']:<6} tempo_wall={dt:8.2f}s")
    return linhas[-1]


class Coletor:
    def __init__(self):
        self.linhas = []

    def registrar(self, linha, caso, baseline_key, extra=None):
        linha = {k.strip(): (v or "").strip() for k, v in linha.items()}
        linha["caso"] = caso
        linha["baseline_key"] = baseline_key
        if extra:
            linha.update(extra)
        self.linhas.append(linha)

    def salvar(self, path=MASTER_CSV):
        if not self.linhas:
            print("Nada para salvar.")
            return
        todas = list(self.linhas)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", newline="") as f:
                todas = list(csv.DictReader(f)) + todas
        colunas = list(dict.fromkeys(k for row in todas for k in row.keys()))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colunas)
            w.writeheader()
            w.writerows(todas)
        print(f"\n{len(self.linhas)} novas linhas anexadas em '{path}'.")


def rodar_baselines(col):
    for key, cfg in BASELINES.items():
        print(f"\n=== Baseline serial: {key} ({N_SEQ_REPETICOES} execucoes x run_count={RUN_COUNT_SEQ}) ===")
        for rep in range(1, N_SEQ_REPETICOES + 1):
            l = rodar_um(key, cfg, procs=1, runs=RUN_COUNT_SEQ, modo="seq")
            col.registrar(l, key, key, {"repeticao": rep})


def rodar_corretude(col):
    print("\n=== CASO V: corretude (--comparar) ===")
    ps = [p for p in (2, 3, 4, 7) if p <= N_CORES]
    for preset_cfg, nome in ((dict(REGIAO_CAVALOS, res=1024), "V_cavalos_1024"),
                             (dict(REGIAO_PADRAO, res=1000), "V_vista_1000")):   # 1000 nao divide por 3/7
        for est, ch in STRATS_B:
            for p in ps:
                l = rodar_um(nome, preset_cfg, procs=p, estrategia=est, chunk=ch, runs=1, comparar=True)
                col.registrar(l, nome, "", {"threads_alvo": p})
    p = min(4, N_CORES)
    for cfg, nome in ((dict(REGIAO_PADRAO, res=4096), "V_vista_4096"),
                      (dict(REGIAO_CAVALOS, res=4096), "V_cavalos_4096")):
        for est, ch in STRATS_A:
            l = rodar_um(nome, cfg, procs=p, estrategia=est, chunk=ch, runs=1, comparar=True)
            col.registrar(l, nome, "", {"threads_alvo": p})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--apenas", default=None, help="lista separada por virgula: V,A,B,C,D,baselines")
    args = ap.parse_args()
    apenas = set(args.apenas.split(",")) if args.apenas else None

    if (not apenas or {"A", "B"} <= apenas) and os.path.exists(PROGRAM_CSV):
        bak = PROGRAM_CSV + f".bak.{int(time.time())}"
        shutil.move(PROGRAM_CSV, bak)
        print(f"CSV interno existente movido para '{bak}'.")

    if not args.no_compile:
        compilar()
    registrar_ambiente()
    col = Coletor()

    try:
        if not apenas or "V" in apenas:
            rodar_corretude(col)
        if not apenas or "baselines" in apenas:
            rodar_baselines(col)

        if not apenas or "A" in apenas:
            print("\n=== CASO A: regiao padrao (strong scaling, 3 estrategias) ===")
            for est, ch in STRATS_A:
                for p in PROCS:
                    l = rodar_um("A_padrao", dict(REGIAO_PADRAO, res=4096), procs=p, estrategia=est,
                                 chunk=ch, runs=RUN_COUNT_PAR)
                    col.registrar(l, "A_padrao", "A_padrao_4096", {"threads_alvo": p})

        if not apenas or "B" in apenas:
            print("\n=== CASO B: cavalos-marinhos (strong scaling + desbalanceamento) ===")
            for est, ch in STRATS_B:
                for p in PROCS:
                    l = rodar_um("B_cavalos", dict(REGIAO_CAVALOS, res=4096), procs=p, estrategia=est,
                                 chunk=ch, runs=RUN_COUNT_PAR)
                    col.registrar(l, "B_cavalos", "B_cavalos_4096", {"threads_alvo": p})

        if not apenas or "C" in apenas:
            print("\n=== CASO C: escalabilidade fraca ===")
            for est, ch in STRATS_C:
                for p, res in WEAK_PAIRS:
                    l = rodar_um("C_weak", dict(REGIAO_PADRAO, res=res), procs=p, estrategia=est,
                                 chunk=ch, runs=RUN_COUNT_PAR)
                    col.registrar(l, "C_weak", f"C_weak_{res}", {"threads_alvo": p})

        if not apenas or "D" in apenas:
            print(f"\n=== CASO D: efeito do chunk (P={P_MAX}, cavalos-marinhos) ===")
            for est in ("cyclic", "master"):
                for ch in CHUNKS_D:
                    l = rodar_um("D_chunk", dict(REGIAO_CAVALOS, res=4096), procs=P_MAX, estrategia=est,
                                 chunk=ch, runs=RUN_COUNT_PAR)
                    col.registrar(l, "D_chunk", "B_cavalos_4096", {"threads_alvo": P_MAX})
    finally:
        col.salvar()   # salva o que ja foi coletado mesmo se algo falhar no meio

    reprov = [l for l in col.linhas if l.get("Aprovado") == "Nao"]
    if reprov:
        print("\n!!! CORRETUDE REPROVADA em:")
        for l in reprov:
            print("   ", l["caso"], l["Estrategia"], "P=", l["Procs"], "chunk=", l["Chunk"])
    else:
        print("\nTodas as checagens de corretude realizadas foram aprovadas.")
    print("Concluido.")


if __name__ == "__main__":
    main()