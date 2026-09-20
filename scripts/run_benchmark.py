#!/usr/bin/env python3
"""
Orquestrador de testes de benchmark HÍBRIDO - DEC107.

Casos Primarios (A, B, C): Focados na variacao base pedida no enunciado.
Casos Secundarios (A2, B2, C2): "Bateria de Estresse", cruzando todas as 
combinacoes possiveis de Threads x Schedules x Chunks.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

# =============================================================================
# CONFIGURACAO -- BATERIA RICA (OVERNIGHT)
# =============================================================================

os.makedirs("saida", exist_ok=True)

# Atualizado para refletir a sua estrutura exata de pastas
SRC_FILE = "src/gera_mandelbrot.c"          
BIN_FILE = "./saida/mandelbrot"
PROGRAM_CSV = "saida/benchmark_resultados.csv"   
MASTER_CSV = "saida/resultados_finais.csv"

N_CORES = os.cpu_count() or 4

# Threads testadas na escalabilidade
THREADS_STRONG = sorted(set([1, 2, 4, min(8, N_CORES), N_CORES]))

# Variações de escalonamento para os casos de estresse (A2, B2, C2)
SCHEDULES = ["static", "dynamic", "guided"]
CHUNKS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]   

# Pares (threads, resolucao) da escalabilidade fraca
WEAK_SCALING_PAIRS = [
    (1, 4096),
    (4, 8192),
    (16, 16384),
]

RUN_COUNT_DEFAULT = 15        
RUN_COUNT_WEAK = 15           
RUN_COUNT_CORRECTNESS = 15    

# Regiao padrao exigida
REGIAO_PADRAO = dict(re_min=-2.0, re_max=1.0, im_min=-1.5, im_max=1.5)
MAX_ITER_PADRAO = 1000
RES_PADRAO = 4096

# Vale dos cavalos-marinhos (desbalanceamento)
_cx, _cy, _half = -0.743643887, 0.131825904, 1.5e-3
REGIAO_CAVALOS = dict(re_min=_cx - _half, re_max=_cx + _half,
                       im_min=_cy - _half, im_max=_cy + _half)
MAX_ITER_CAVALOS = 5000
RES_CAVALOS = 4096

# Reaproveita a medicao sequencial pesada na mesma regiao/resolucao
PAREAR_SEQUENCIAL_SEMPRE = False

# =============================================================================
# LOGICA DE EXECUCAO E ORQUESTRACAO
# =============================================================================

SCHED_CODE = {"static": 1, "dynamic": 2, "guided": 3}

HEADER_ESPERADO = ["Modo", "Cenario", "Threads", "Escalonamento", "Chunk", "Resolucao",
                    "MaxIter", "Vezes", "T_Med_Glob", "T_Min_Glob", "T_Max_Glob",
                    "T_Med_Seq", "Speedup", "Eficiencia", "FatorBal", "Simetria",
                    "T_Min_Thr", "T_Max_Thr", "T_Med_Thr", "Acuracia_%", "Diferentes",
                    "Dif_Alem_Tol", "Aprovado", "Medias_Threads"]

_seq_cache = {}


def compilar():
    print(f"Compilando {SRC_FILE} ...")
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-o", BIN_FILE, SRC_FILE, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Erro de compilacao:\n", r.stderr)
        sys.exit(1)
    print("Compilado com sucesso.\n")


def _fmt(x):
    return repr(x) if isinstance(x, float) else str(x)


def montar_input(caso, regiao, width, height, max_iter, is_parallel, threads=None,
                  schedule=None, chunk=None, comparar=0, run_count=1):
    linhas = ["2",  
              caso,  # NOVO: Envia o nome do cenário para o scanf("%31s") do C
              _fmt(regiao["re_min"]), _fmt(regiao["re_max"]),
              _fmt(regiao["im_min"]), _fmt(regiao["im_max"]),
              str(width), str(height), str(max_iter),
              "0"]  

    if is_parallel:
        linhas += ["1", str(threads), str(SCHED_CODE[schedule]), str(chunk),
                   str(1 if comparar else 0),
                   "0"]  
    else:
        linhas += ["2"]  

    linhas += ["1",               
               "0",               
               str(run_count),
               "3"]               
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
             schedule=None, chunk=None, comparar=0, run_count=RUN_COUNT_DEFAULT,
             timeout=3600):
    entrada = montar_input(caso, regiao, width, height, max_iter, is_parallel,
                            threads, schedule, chunk, comparar, run_count)
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
        self._pair_id = 0

    def novo_par(self):
        self._pair_id += 1
        return self._pair_id

    def registrar(self, linha_bruta, caso, pair_id, extra=None):
        campos = parse_linha_csv(linha_bruta)
        campos["caso"] = caso
        campos["pair_id"] = pair_id
        if extra:
            campos.update(extra)
        self.linhas.append(campos)

    def salvar(self, path=MASTER_CSV):
        if not self.linhas:
            print("Nada para salvar.")
            return
        colunas = list(self.linhas[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colunas)
            w.writeheader()
            w.writerows(self.linhas)
        print(f"\n{len(self.linhas)} linhas salvas em '{path}'.")


def _seq_key(regiao, width, height, max_iter, run_count):
    return (regiao["re_min"], regiao["re_max"], regiao["im_min"], regiao["im_max"],
            width, height, max_iter, run_count)


def rodar_par(coletor, caso, regiao, width, height, max_iter, threads, schedule,
              chunk, comparar=0, run_count=RUN_COUNT_DEFAULT, extra=None):
    pair_id = coletor.novo_par()

    linha_par = rodar_um(caso, regiao, width, height, max_iter, is_parallel=True,
                          threads=threads, schedule=schedule, chunk=chunk,
                          comparar=comparar, run_count=run_count)
    coletor.registrar(linha_par, caso, pair_id, extra)

    key = _seq_key(regiao, width, height, max_iter, run_count)
    if not PAREAR_SEQUENCIAL_SEMPRE and key in _seq_cache:
        linha_seq = _seq_cache[key]
        #print("  [Sequencial] (reaproveitado via cache da resolucao/iteracao)")
    else:
        linha_seq = rodar_um(caso, regiao, width, height, max_iter, is_parallel=False,
                              run_count=run_count)
        _seq_cache[key] = linha_seq

    coletor.registrar(linha_seq, caso, pair_id, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apenas", default=None,
                     help="lista separada por virgula: A,A2,B,B2,C,C2")
    args = ap.parse_args()
    apenas = set(args.apenas.split(",")) if args.apenas else None

    if args.dry_run:
        imprimir_plano(apenas)
        return

    if os.path.exists(PROGRAM_CSV):
        backup = PROGRAM_CSV + f".bak.{int(time.time())}"
        shutil.move(PROGRAM_CSV, backup)
        print(f"CSV interno existente movido para '{backup}' (comecando do zero).")

    if not args.no_compile:
        compilar()

    coletor = Coletor()

    # ================= CASOS PRIMARIOS (VARIAÇÃO BASE) =================

    if not apenas or "A" in apenas:
        print(f"\n=== CASO A: Regiao Padrao (Strong Scaling Básico) ===")
        for i, t in enumerate(THREADS_STRONG):
            rodar_par(coletor, "A_padrao_threads", REGIAO_PADRAO, RES_PADRAO, RES_PADRAO,
                      MAX_ITER_PADRAO, threads=t, schedule="dynamic", chunk=0,
                      comparar=(1 if i == 0 else 0),
                      run_count=(RUN_COUNT_CORRECTNESS if i == 0 else RUN_COUNT_DEFAULT),
                      extra={"threads_alvo": t})

    if not apenas or "B" in apenas:
        print(f"\n=== CASO B: Cavalos-Marinhos (Desbalanceamento Básico) ===")
        for i, t in enumerate(THREADS_STRONG):
            rodar_par(coletor, "B_cavalos_threads", REGIAO_CAVALOS, RES_CAVALOS, RES_CAVALOS,
                      MAX_ITER_CAVALOS, threads=t, schedule="dynamic", chunk=0,
                      comparar=(1 if i == 0 else 0),
                      run_count=(RUN_COUNT_CORRECTNESS if i == 0 else RUN_COUNT_DEFAULT),
                      extra={"threads_alvo": t})

    if not apenas or "C" in apenas:
        print(f"\n=== CASO C: Escalabilidade Fraca (Básico) ===")
        for t, res in WEAK_SCALING_PAIRS:
            rodar_par(coletor, "C_weak_scaling", REGIAO_PADRAO, res, res, MAX_ITER_PADRAO,
                      threads=t, schedule="dynamic", chunk=0, comparar=0,
                      run_count=RUN_COUNT_WEAK, extra={"threads_alvo": t})


    # ================= CASOS SECUNDÁRIOS (VARIAÇÃO COMPLETA) =================

    if not apenas or "A2" in apenas:
        print(f"\n=== CASO A2: Regiao Padrao (ESTRESSE - Threads x Sched x Chunk) ===")
        for t in THREADS_STRONG:
            for sched in SCHEDULES:
                for chunk in CHUNKS:
                    rodar_par(coletor, "A2_padrao_completo", REGIAO_PADRAO, RES_PADRAO, RES_PADRAO,
                              MAX_ITER_PADRAO, threads=t, schedule=sched, chunk=chunk,
                              comparar=0, run_count=RUN_COUNT_DEFAULT, extra={"threads_alvo": t})

    if not apenas or "B2" in apenas:
        print(f"\n=== CASO B2: Cavalos-Marinhos (ESTRESSE - Threads x Sched x Chunk) ===")
        for t in THREADS_STRONG:
            for sched in SCHEDULES:
                for chunk in CHUNKS:
                    rodar_par(coletor, "B2_cavalos_completo", REGIAO_CAVALOS, RES_CAVALOS, RES_CAVALOS,
                              MAX_ITER_CAVALOS, threads=t, schedule=sched, chunk=chunk,
                              comparar=0, run_count=RUN_COUNT_DEFAULT, extra={"threads_alvo": t})

    if not apenas or "C2" in apenas:
        print(f"\n=== CASO C2: Escalabilidade Fraca (ESTRESSE - Pares x Sched x Chunk) ===")
        for t, res in WEAK_SCALING_PAIRS:
            for sched in SCHEDULES:
                for chunk in CHUNKS:
                    rodar_par(coletor, "C2_weak_completo", REGIAO_PADRAO, res, res, MAX_ITER_PADRAO,
                              threads=t, schedule=sched, chunk=chunk, comparar=0,
                              run_count=RUN_COUNT_WEAK, extra={"threads_alvo": t})

    coletor.salvar()

    reprovados = [l for l in coletor.linhas if l.get("Aprovado") not in ("N/A", "Sim")]
    if reprovados:
        print("\n!!! ATENCAO: checagem de corretude NAO aprovada em:")
        for l in reprovados:
            print("   ", l["caso"], "Aprovado=", l.get("Aprovado"))
    else:
        print("\nTodas as checagens de corretude realizadas foram aprovadas.")


def imprimir_plano(apenas):
    partes = []
    # Primarios
    if not apenas or "A" in apenas:
        partes.append(("A", len(THREADS_STRONG)))
    if not apenas or "B" in apenas:
        partes.append(("B", len(THREADS_STRONG)))
    if not apenas or "C" in apenas:
        partes.append(("C", len(WEAK_SCALING_PAIRS)))
    
    # Secundarios (Total Variation)
    if not apenas or "A2" in apenas:
        partes.append(("A2", len(THREADS_STRONG) * len(SCHEDULES) * len(CHUNKS)))
    if not apenas or "B2" in apenas:
        partes.append(("B2", len(THREADS_STRONG) * len(SCHEDULES) * len(CHUNKS)))
    if not apenas or "C2" in apenas:
        partes.append(("C2", len(WEAK_SCALING_PAIRS) * len(SCHEDULES) * len(CHUNKS)))
    
    total_cfg = sum(n for _, n in partes)
    print("Plano de execucao HIFRIDO (Primarios = Basico | Secundarios = Total):")
    for nome, n in partes:
        print(f"  Caso {nome}: {n} configuracoes x 2 execucoes (par+seq) = {n*2} chamadas")
    print(f"\nTOTAL GERAL: {total_cfg} configuracoes combinadas")
    print(f"\nTHREADS_STRONG={THREADS_STRONG}")
    print(f"SCHEDULES={SCHEDULES}  CHUNKS={CHUNKS}")
    print(f"WEAK_SCALING_PAIRS={WEAK_SCALING_PAIRS}")


if __name__ == "__main__":
    main()
