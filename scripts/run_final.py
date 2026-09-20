#!/usr/bin/env python3
"""
Orquestrador FINAL - DEC107.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

os.makedirs("saida", exist_ok=True) # Garante que a pasta existe

SRC_FILE = "src/mandelbrot.c"
BIN_FILE = "./saida/mandelbrot"
PROGRAM_CSV = "saida/benchmark_resultados.csv"
MASTER_CSV = "saida/resultados_finais.csv"

SCHED_CODE = {"static": 1, "dynamic": 2, "guided": 3}
REGIAO_PADRAO = dict(re_min=-2.0, re_max=1.0, im_min=-1.5, im_max=1.5)
MAX_ITER_PADRAO = 1000

_cx, _cy, _half = -0.743643887, 0.131825904, 1.5e-3
REGIAO_CAVALOS = dict(re_min=_cx - _half, re_max=_cx + _half,
                       im_min=_cy - _half, im_max=_cy + _half)
MAX_ITER_CAVALOS = 5000

CONFIG_A = [(1, "dynamic", 4), (2, "dynamic", 4), (4, "dynamic", 8), (8, "dynamic", 16), (16, "dynamic", 16)]
CONFIG_B = [
    (1, "static", 0), (2, "static", 0), (4, "static", 0), (8, "static", 0), (16, "static", 0),
    (1, "dynamic", 16), (2, "dynamic", 16), (4, "dynamic", 16), (8, "dynamic", 16), (16, "dynamic", 16),
    (1, "guided", 16), (2, "guided", 16), (4, "guided", 16), (8, "guided", 16), (16, "guided", 16),
    (1, "static", 2), (2, "static", 2), (4, "static", 2), (8, "static", 2), (16, "static", 2)
]
CONFIG_C_SCHED_CHUNK = ("dynamic", 16)
WEAK_SCALING_PAIRS_NAO_TRIVIAIS = [(2, 5793), (4, 8192), (8, 11585), (16, 16384)]
CONFIG_D = [(16, "dynamic", c) for c in [1, 2, 4, 16, 64, 256, 1024]]

BASELINES = {
    "A_padrao_4096":  dict(regiao=REGIAO_PADRAO,  width=4096,  height=4096,  max_iter=MAX_ITER_PADRAO),
    "B_cavalos_4096": dict(regiao=REGIAO_CAVALOS, width=4096,  height=4096,  max_iter=MAX_ITER_CAVALOS),
    "C_weak_5793":    dict(regiao=REGIAO_PADRAO,  width=5793,  height=5793,  max_iter=MAX_ITER_PADRAO),
    "C_weak_8192":    dict(regiao=REGIAO_PADRAO,  width=8192,  height=8192,  max_iter=MAX_ITER_PADRAO),
    "C_weak_11585":   dict(regiao=REGIAO_PADRAO,  width=11585, height=11585, max_iter=MAX_ITER_PADRAO),
    "C_weak_16384":   dict(regiao=REGIAO_PADRAO,  width=16384, height=16384, max_iter=MAX_ITER_PADRAO),
}

N_SEQ_REPETICOES = 10   
RUN_COUNT_SEQ = 3      
RUN_COUNT_PAR = 30     

# --- A CORREÇÃO CRÍTICA ESTÁ AQUI ---
HEADER_ESPERADO = [
    "Modo", "Cenario", "Threads", "Escalonamento", "Chunk", "Resolucao",
    "MaxIter", "Vezes", "T_Med_Glob", "T_Min_Glob", "T_Max_Glob",
    "T_Med_Seq", "Speedup", "Eficiencia", "FatorBal", "Simetria",
    "T_Min_Thr", "T_Max_Thr", "T_Med_Thr",
    "Acuracia_%", "Diferentes", "Dif_Alem_Tol", "Aprovado", "Medias_Threads"
]

def compilar():
    print(f"Compilando {SRC_FILE} com otimizacoes de arquitetura...")
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-o", "mandelbrot", SRC_FILE, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Erro de compilacao:\n", r.stderr)
        sys.exit(1)
    print("Compilado com sucesso.\n")

def _fmt(x):
    return repr(x) if isinstance(x, float) else str(x)

def montar_input(caso, regiao, width, height, max_iter, is_parallel, threads=None,
                  schedule=None, chunk=None, run_count=1):
    linhas = ["2", caso,
              _fmt(regiao["re_min"]), _fmt(regiao["re_max"]),
              _fmt(regiao["im_min"]), _fmt(regiao["im_max"]),
              str(width), str(height), str(max_iter),
              "0"]
    if is_parallel:
        linhas += ["1", str(threads), str(SCHED_CODE[schedule]), str(chunk), "0", "0"]
    else:
        linhas += ["2"]
    linhas += ["1", "0", str(run_count), "3"]
    return "\n".join(linhas) + "\n"

def _contar_linhas(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)

def _ultima_linha_nova(path, linhas_antes):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        todas = f.readlines()
    return todas[-1] if len(todas) > linhas_antes else None

def rodar_um(caso, regiao, width, height, max_iter, is_parallel, threads=None,
             schedule=None, chunk=None, run_count=1, timeout=7200):
    entrada = montar_input(caso, regiao, width, height, max_iter, is_parallel,
                            threads, schedule, chunk, run_count)
    linhas_antes = _contar_linhas(PROGRAM_CSV)

    t0 = time.time()
    r = subprocess.run([BIN_FILE], input=entrada, text=True,
                        capture_output=True, timeout=timeout)
    dt = time.time() - t0

    if r.returncode != 0:
        print("### Falha na execucao ###\nSTDOUT:\n", r.stdout[-2000:],
              "\nSTDERR:\n", r.stderr[-2000:])
        raise RuntimeError(f"Processo terminou com codigo {r.returncode}")

    linha = _ultima_linha_nova(PROGRAM_CSV, linhas_antes)
    if linha is None:
        print(r.stdout[-2000:])
        raise RuntimeError("Nenhuma linha nova encontrada em " + PROGRAM_CSV)

    modo = "Paralelo" if is_parallel else "Sequencial"
    print(f"  [{modo:10s}] threads={threads or '-':<3} sched={schedule or '-':<8} "
          f"chunk={chunk if chunk is not None else '-':<4} res={width}x{height} "
          f"max_iter={max_iter:<5} tempo_wall={dt:7.2f}s")
    return linha

def parse_linha_csv(linha_bruta):
    leitor = csv.reader([linha_bruta], skipinitialspace=True)
    campos = [c.strip() for c in next(leitor)]
    return dict(zip(HEADER_ESPERADO, campos))

class Coletor:
    def __init__(self):
        self.linhas = []

    def registrar(self, linha_bruta, caso, baseline_key, extra=None):
        campos = parse_linha_csv(linha_bruta)
        campos["caso"] = caso
        campos["baseline_key"] = baseline_key
        if extra:
            campos.update(extra)
        self.linhas.append(campos)

    def salvar(self, path=MASTER_CSV):
        if not self.linhas:
            print("Nada para salvar.")
            return
        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                existentes = list(r)
            todas_linhas = existentes + self.linhas
        else:
            todas_linhas = self.linhas
            
        colunas = list(dict.fromkeys(k for row in todas_linhas for k in row.keys()))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colunas)
            w.writeheader()
            w.writerows(todas_linhas)
        print(f"\n{len(self.linhas)} novas linhas anexadas em '{path}'.")

def rodar_baselines(coletor, apenas=None):
    for baseline_key, cfg in BASELINES.items():
        if apenas and baseline_key not in apenas:
            continue
        print(f"\n=== Baseline sequencial: {baseline_key} "
              f"({N_SEQ_REPETICOES} execucoes independentes x run_count={RUN_COUNT_SEQ}) ===")
        for rep in range(1, N_SEQ_REPETICOES + 1):
            linha = rodar_um(baseline_key, cfg["regiao"], cfg["width"], cfg["height"],
                              cfg["max_iter"], is_parallel=False, run_count=RUN_COUNT_SEQ)
            coletor.registrar(linha, caso=baseline_key, baseline_key=baseline_key,
                               extra={"repeticao": rep})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--apenas", default=None,
                     help="lista separada por virgula: A,B,C,D,baselines")
    args = ap.parse_args()
    apenas = set(args.apenas.split(",")) if args.apenas else None

    if os.path.exists(PROGRAM_CSV):
        backup = PROGRAM_CSV + f".bak.{int(time.time())}"
        shutil.move(PROGRAM_CSV, backup)
        print(f"CSV interno existente movido para '{backup}'.")

    if not args.no_compile:
        compilar()

    coletor = Coletor()

    if not apenas or "baselines" in apenas:
        rodar_baselines(coletor)

    if not apenas or "A" in apenas:
        print("\n=== CASO A: regiao padrao (Strong Scaling) ===")
        for t, sched, chunk in CONFIG_A:
            linha = rodar_um("A_padrao_threads", REGIAO_PADRAO, 4096, 4096, MAX_ITER_PADRAO,
                              is_parallel=True, threads=t, schedule=sched, chunk=chunk,
                              run_count=RUN_COUNT_PAR)
            coletor.registrar(linha, caso="A_padrao_threads", baseline_key="A_padrao_4096",
                               extra={"threads_alvo": t})

    if not apenas or "B" in apenas:
        print("\n=== CASO B: cavalos-marinhos (Strong Scaling - Otimizacoes e Overheads) ===")
        for t, sched, chunk in CONFIG_B:
            linha = rodar_um("B_cavalos_threads", REGIAO_CAVALOS, 4096, 4096, MAX_ITER_CAVALOS,
                              is_parallel=True, threads=t, schedule=sched, chunk=chunk,
                              run_count=RUN_COUNT_PAR)
            coletor.registrar(linha, caso="B_cavalos_threads", baseline_key="B_cavalos_4096",
                               extra={"threads_alvo": t})

    if not apenas or "C" in apenas:
        print("\n=== CASO C: escalabilidade fraca ===")
        sched, chunk = CONFIG_C_SCHED_CHUNK
        for t, res in WEAK_SCALING_PAIRS_NAO_TRIVIAIS:
            baseline_key = f"C_weak_{res}"
            linha = rodar_um("C_weak_scaling", REGIAO_PADRAO, res, res, MAX_ITER_PADRAO,
                              is_parallel=True, threads=t, schedule=sched, chunk=chunk,
                              run_count=RUN_COUNT_PAR)
            coletor.registrar(linha, caso="C_weak_scaling", baseline_key=baseline_key,
                               extra={"threads_alvo": t})

    if not apenas or "D" in apenas:
        print("\n=== CASO D: Efeito do tamanho do Chunk (16t, Dynamic, Cavalos-Marinhos) ===")
        for t, sched, chunk in CONFIG_D:
            linha = rodar_um("D_chunk_effect", REGIAO_CAVALOS, 4096, 4096, MAX_ITER_CAVALOS,
                              is_parallel=True, threads=t, schedule=sched, chunk=chunk,
                              run_count=RUN_COUNT_PAR)
            coletor.registrar(linha, caso="D_chunk_effect", baseline_key="B_cavalos_4096",
                               extra={"threads_alvo": t})

    coletor.salvar()
    print("\nConcluido.")

if __name__ == "__main__":
    main()
