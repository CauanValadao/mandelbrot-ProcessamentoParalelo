#!/usr/bin/env python3
"""
Analises finais da Etapa 2 (MPI) - DEC107.

Uso: python analisar_mpi.py [saida/resultados_mpi.csv] [saida/resultados_finais.csv (OpenMP, opcional)]

Gera em graficos/:
  A_*, B_*   speedup / eficiencia / tempo por estrategia (strong scaling)
  B_fator_balanceamento.png, B_decomposicao_tempo.png (calculo x espera x comunicacao)
  C_weak_scaling.png, D_efeito_chunk.png
  comparacao_openmp_*.png  MPI x OpenMP (OpenMP SEM simetria, para comparacao justa)
  tabela_resumo_mpi.csv    tabela pronta para o relatorio
"""
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as _stats
    def _t(n): return _stats.t.ppf(0.975, df=max(n - 1, 1))
except ImportError:
    def _t(n): return 1.96

CSV_MPI = sys.argv[1] if len(sys.argv) > 1 else "saida/resultados_mpi.csv"
CSV_OMP = sys.argv[2] if len(sys.argv) > 2 else "saida/resultados_finais.csv"
OUT = Path("graficos"); OUT.mkdir(exist_ok=True)

NUM = ["Procs", "Chunk", "T_Med_Glob", "T_Med_Seq", "FatorBal", "Razao_Max_Med", "T_Comp_Med",
       "T_Espera_Med", "T_Comm_Rank0", "T_Comm_Med", "T_Escrita", "repeticao", "threads_alvo"]


def carregar():
    if not os.path.exists(CSV_MPI):
        sys.exit(f"Arquivo '{CSV_MPI}' nao encontrado.")
    df = pd.read_csv(CSV_MPI)
    df.columns = [c.strip() for c in df.columns]
    for c in NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Modo", "Estrategia", "caso", "baseline_key"):
        df[c] = df[c].astype(str).str.strip()
    return df


def baselines(df):
    seq = df[df["Modo"] == "Sequencial"]
    out = []
    for key, g in seq.groupby("baseline_key"):
        n = len(g); m = g["T_Med_Glob"].mean()
        sd = g["T_Med_Glob"].std(ddof=1) if n > 1 else 0.0
        out.append(dict(baseline_key=key, seq_media=m, seq_desvio=sd, seq_n=n,
                        seq_ic95=_t(n) * sd / (n ** 0.5) if n > 1 else 0.0))
    return pd.DataFrame(out)


def tabela(df, base, caso):
    t = df[(df["Modo"] == "MPI") & (df["caso"] == caso)].copy()
    if t.empty:
        return t
    t = t.merge(base, on="baseline_key", how="left")
    t["speedup"] = t["seq_media"] / t["T_Med_Glob"]
    t["sp_min"] = (t["seq_media"] - t["seq_ic95"]) / t["T_Med_Glob"]
    t["sp_max"] = (t["seq_media"] + t["seq_ic95"]) / t["T_Med_Glob"]
    t["efic"] = t["speedup"] / t["Procs"]
    # no mestre-trabalhador so (P-1) processos calculam
    nt = t["Procs"].where(~((t["Estrategia"] == "MasterWorker") & (t["Procs"] > 1)), t["Procs"] - 1)
    t["efic_trab"] = t["speedup"] / nt
    t["config"] = t.apply(lambda r: r["Estrategia"] if r["Estrategia"] == "Block"
                          else f'{r["Estrategia"]} (C={int(r["Chunk"])})', axis=1)
    return t.sort_values(["config", "Procs"])


def salvar(fig, nome):
    fig.tight_layout(); fig.savefig(OUT / nome, dpi=300); plt.close(fig); print("Gerado:", nome)


def strong(t, titulo, prefixo):
    if t.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for cfg, g in t.groupby("config"):
        a1.errorbar(g["Procs"], g["speedup"], yerr=[g["speedup"] - g["sp_min"], g["sp_max"] - g["speedup"]],
                    fmt="o-", capsize=4, label=cfg)
        a2.plot(g["Procs"], g["efic"], "o-", label=cfg)
    ps = sorted(t["Procs"].unique())
    a1.plot(ps, ps, "--", color="gray", label="Ideal")
    a1.set(xlabel="Processos", ylabel="Speedup", title=f"Speedup - {titulo}"); a1.legend(); a1.grid(alpha=.3)
    a2.axhline(1, ls="--", color="gray")
    a2.set(xlabel="Processos", ylabel="Eficiencia", ylim=(0, 1.15), title=f"Eficiencia - {titulo}")
    a2.legend(); a2.grid(alpha=.3)
    salvar(fig, f"{prefixo}_speedup_eficiencia.png")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cfg, g in t.groupby("config"):
        ax.plot(g["Procs"], g["T_Med_Glob"], "o-", label=cfg)
    ax.axhline(t["seq_media"].iloc[0], ls="--", color="black", label=f'Serial ({t["seq_media"].iloc[0]:.2f}s)')
    ax.set(xlabel="Processos", ylabel="Tempo medio (s)", title=f"Tempo de execucao - {titulo}")
    ax.legend(); ax.grid(alpha=.3)
    salvar(fig, f"{prefixo}_tempo.png")


def balanceamento(t, titulo):
    if t.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for cfg, g in t.groupby("config"):
        a1.plot(g["Procs"], g["FatorBal"], "o-", label=cfg)
        a2.plot(g["Procs"], g["Razao_Max_Med"], "o-", label=cfg)
    a1.axhline(0, ls="--", color="gray")
    a1.set(xlabel="Processos", ylabel="(Tmax - Tmin) / Tmax", title=f"Fator de balanceamento - {titulo}")
    a2.axhline(1, ls="--", color="gray")
    a2.set(xlabel="Processos", ylabel="Tmax / Tmedio", title=f"Razao carga maxima/media - {titulo}")
    for a in (a1, a2): a.legend(); a.grid(alpha=.3)
    salvar(fig, "B_fator_balanceamento.png")


def decomposicao(t, titulo):
    g = t[t["Procs"] == t["Procs"].max()]
    if g.empty:
        return
    g = g.sort_values("config")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(g))
    ax.bar(x, g["T_Comp_Med"], label="Calculo")
    ax.bar(x, g["T_Espera_Med"], bottom=g["T_Comp_Med"], label="Espera (ocioso)")
    ax.bar(x, g["T_Comm_Med"], bottom=g["T_Comp_Med"] + g["T_Espera_Med"], label="Comunicacao")
    ax.set_xticks(list(x)); ax.set_xticklabels(g["config"], rotation=20, ha="right")
    ax.set(ylabel="Tempo medio por processo (s)",
           title=f"Decomposicao do tempo - {titulo} ({int(g['Procs'].max())} processos)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    salvar(fig, "B_decomposicao_tempo.png")


def weak(df, base):
    t = tabela(df, base, "C_weak")
    if t.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for cfg, g in t.groupby("config"):
        g = g.sort_values("Procs")
        res = g["Resolucao"].str.split("x").str[0].astype(int)
        a1.plot(res, g["T_Med_Glob"], "o-", label=cfg)
        a2.plot(res, g["T_Med_Glob"].iloc[0] / g["T_Med_Glob"], "o-", label=cfg)
        for x, y1, y2, p in zip(res, g["T_Med_Glob"], g["T_Med_Glob"].iloc[0] / g["T_Med_Glob"], g["Procs"]):
            a1.annotate(f" {int(p)}p", (x, y1), fontsize=8); a2.annotate(f" {int(p)}p", (x, y2), fontsize=8)
    a1.set(xlabel="Resolucao (N x N)", ylabel="Tempo (s)", title="Escala fraca - tempo"); a1.legend(); a1.grid(alpha=.3)
    a2.axhline(1, ls="--", color="gray")
    a2.set(xlabel="Resolucao (N x N)", ylabel="Eficiencia fraca", ylim=(0, 1.15), title="Escala fraca - eficiencia")
    a2.legend(); a2.grid(alpha=.3)
    salvar(fig, "C_weak_scaling.png")


def chunk_effect(df):
    d = df[(df["Modo"] == "MPI") & (df["caso"] == "D_chunk")]
    if d.empty:
        return
    for col, ylab, nome in (("T_Med_Glob", "Tempo medio (s)", "D_efeito_chunk.png"),
                            ("FatorBal", "(Tmax - Tmin) / Tmax", "D_balanceamento_chunk.png")):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for est, g in d.groupby("Estrategia"):
            g = g.sort_values("Chunk")
            ax.plot(g["Chunk"], g[col], "o-", label=est)
        ax.set_xscale("log")
        ax.set(xlabel="Chunk (linhas por bloco / tarefa)", ylabel=ylab,
               title=f"Efeito do chunk - {int(d['Procs'].iloc[0])} processos, cavalos-marinhos")
        ax.legend(); ax.grid(alpha=.3)
        salvar(fig, nome)


def comparar_openmp(df_mpi_tab, caso_mpi, caso_omp, titulo, arq):
    """Melhor configuracao MPI x melhor OpenMP por numero de unidades paralelas.
    Ambos usam o MESMO baseline serial (o do MPI) -> speedup diretamente comparavel."""
    if not os.path.exists(CSV_OMP) or df_mpi_tab.empty:
        return
    o = pd.read_csv(CSV_OMP); o.columns = [c.strip() for c in o.columns]
    for c in ("Threads", "Chunk", "T_Med_Glob"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = o[(o["Modo"].astype(str).str.strip() == "Paralelo") & (o["caso"] == caso_omp)]
    if o.empty:
        print(f"[aviso] OpenMP: caso '{caso_omp}' nao encontrado em {CSV_OMP}")
        return
    omp = o.loc[o.groupby("Threads")["T_Med_Glob"].idxmin()].sort_values("Threads")
    mpi = df_mpi_tab.loc[df_mpi_tab.groupby("Procs")["T_Med_Glob"].idxmin()].sort_values("Procs")
    seq = df_mpi_tab["seq_media"].iloc[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    a1.plot(omp["Threads"], omp["T_Med_Glob"], "s-", label="OpenMP (melhor config.)")
    a1.plot(mpi["Procs"], mpi["T_Med_Glob"], "o-", label="MPI (melhor config.)")
    for cfg, g in df_mpi_tab.groupby("config"):
        a1.plot(g["Procs"], g["T_Med_Glob"], ":", alpha=.5, label=f"MPI {cfg}")
    a1.set(xlabel="Threads / Processos", ylabel="Tempo medio (s)", title=f"Tempo - {titulo}")
    a1.legend(fontsize=7); a1.grid(alpha=.3)

    a2.plot(omp["Threads"], seq / omp["T_Med_Glob"], "s-", label="OpenMP")
    a2.plot(mpi["Procs"], seq / mpi["T_Med_Glob"], "o-", label="MPI")
    ps = sorted(set(omp["Threads"]) | set(mpi["Procs"]))
    a2.plot(ps, ps, "--", color="gray", label="Ideal")
    a2.set(xlabel="Threads / Processos", ylabel="Speedup (vs. serial MPI)", title=f"Speedup - {titulo}")
    a2.legend(); a2.grid(alpha=.3)
    salvar(fig, arq)

    resumo = pd.DataFrame({"P": mpi["Procs"].values,
                           "MPI_melhor": mpi["config"].values, "MPI_tempo_s": mpi["T_Med_Glob"].values})
    resumo = resumo.merge(omp[["Threads", "Escalonamento", "Chunk", "T_Med_Glob"]]
                          .rename(columns={"Threads": "P", "Escalonamento": "OMP_sched", "Chunk": "OMP_chunk",
                                           "T_Med_Glob": "OMP_tempo_s"}), on="P", how="outer")
    resumo["MPI/OMP"] = resumo["MPI_tempo_s"] / resumo["OMP_tempo_s"]
    resumo.to_csv(OUT / f"{arq.replace('.png', '')}.csv", index=False)


def main():
    df = carregar()
    base = baselines(df)
    ta, tb = tabela(df, base, "A_padrao"), tabela(df, base, "B_cavalos")

    strong(ta, "Regiao padrao", "A")
    strong(tb, "Cavalos-marinhos", "B")
    balanceamento(tb, "Cavalos-marinhos")
    decomposicao(tb, "Cavalos-marinhos")
    weak(df, base)
    chunk_effect(df)
    # OpenMP sem simetria (caso A) para comparacao justa; caso B nao usa simetria (regiao assimetrica)
    comparar_openmp(ta, "A_padrao", "A_padrao_sem_simetria", "Regiao padrao", "comparacao_openmp_A.png")
    comparar_openmp(tb, "B_cavalos", "B_cavalos_threads", "Cavalos-marinhos", "comparacao_openmp_B.png")

    cols = ["caso", "config", "Procs", "T_Med_Glob", "seq_media", "speedup", "efic", "efic_trab", "FatorBal",
            "Razao_Max_Med", "T_Comp_Med", "T_Espera_Med", "T_Comm_Rank0", "T_Comm_Med", "T_Escrita"]
    resumo = pd.concat([x[cols] for x in (ta, tb) if not x.empty], ignore_index=True)
    resumo.round(4).to_csv(OUT / "tabela_resumo_mpi.csv", index=False)
    print("Gerado: tabela_resumo_mpi.csv")

    v = df[df["caso"].str.startswith("V_")]
    if not v.empty:
        print("\nCorretude:", (v["Aprovado"] == "Sim").sum(), "aprovadas de", len(v))


if __name__ == "__main__":
    main()