#!/usr/bin/env python3
"""
Le 'resultados_finais.csv' e produz as analises finais:
Speedup, Eficiência, Fator de Balanceamento e Escalabilidade.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as _stats
    def _valor_critico(n):
        return _stats.t.ppf(0.975, df=max(n - 1, 1))
except ImportError:
    def _valor_critico(n):
        return 1.96

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "resultados_finais.csv"
OUT_DIR = Path("graficos_finais")
OUT_DIR.mkdir(exist_ok=True)

def carregar():
    df = pd.read_csv(CSV_PATH)
    # Limpa nomes de colunas
    df.columns = [c.strip() for c in df.columns]
    num_cols = ["Threads", "Chunk", "T_Med_Glob", "T_Min_Glob", "T_Max_Glob", "FatorBal", "repeticao"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["threads_alvo"] = df["Threads"]
    return df

def stats_baseline(df):
    seq = df[df["Modo"].str.strip() == "Sequencial"]
    linhas = []
    for key, grupo in seq.groupby("baseline_key"):
        n = len(grupo)
        media = grupo["T_Med_Glob"].mean()
        desvio = grupo["T_Med_Glob"].std(ddof=1) if n > 1 else 0.0
        erro_padrao = desvio / (n ** 0.5) if n > 1 else 0.0
        ic95 = _valor_critico(n) * erro_padrao
        linhas.append({"baseline_key": key, "seq_media": media, "seq_desvio": desvio,
                        "seq_n": n, "seq_ic95": ic95})
    return pd.DataFrame(linhas)

def montar_tabela(df, baselines, caso):
    par = df[(df["Modo"].str.strip() == "Paralelo") & (df["caso"] == caso)].copy()
    if par.empty:
        return par
    par = par.merge(baselines, on="baseline_key", how="left")
    par["speedup"] = par["seq_media"] / par["T_Med_Glob"]
    par["speedup_min"] = (par["seq_media"] - par["seq_ic95"]) / par["T_Med_Glob"]
    par["speedup_max"] = (par["seq_media"] + par["seq_ic95"]) / par["T_Med_Glob"]
    par["Eficiência"] = par["speedup"] / par["threads_alvo"]
    par["Eficiência_min"] = par["speedup_min"] / par["threads_alvo"]
    par["Eficiência_max"] = par["speedup_max"] / par["threads_alvo"]
    
    # Coluna extra para diferenciar as 4 narrativas
    par["Escalonamento_Formatado"] = par["Escalonamento"].str.strip() + " (C=" + par["Chunk"].astype(str) + ")"
    return par.sort_values("threads_alvo")

def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(OUT_DIR / nome, dpi=130)
    plt.close(fig)
    print("Gerado:", nome)

def grafico_strong_scaling(tabela, titulo, arquivo, arquivo_tempo=None, combine_chunks=False):
    if tabela.empty:
        return
        
    # Agrupa apenas pelo nome do escalonamento (ex: "Dynamic") para unir a linha toda
    group_col = "Escalonamento" if combine_chunks else "Escalonamento_Formatado"

    # Grafico Opcional de Tempo Bruto (Caso A)
    if arquivo_tempo:
        fig_t, ax_t = plt.subplots(figsize=(7, 4.5))
        for esc, grupo in tabela.groupby(group_col):
            grupo = grupo.sort_values("threads_alvo")
            
            if combine_chunks:
                # Linha contínua de fundo
                ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "-", color="gray", alpha=0.5)
                # Pontos coloridos separados por chunk para a legenda
                for chunk, subgrupo in grupo.groupby("Chunk"):
                    ax_t.plot(subgrupo["threads_alvo"], subgrupo["T_Med_Glob"], "o", label=f"Chunk={chunk}")
            else:
                ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "o-", label=esc)
        
        seq_media = tabela["seq_media"].iloc[0]
        ax_t.axhline(seq_media, ls="--", color="black", label=f"Sequencial ({seq_media:.2f}s)")
        ax_t.set_xlabel("Threads"); ax_t.set_ylabel("Tempo Médio (s)")
        ax_t.set_title(f"Tempo de Execução — {titulo}")
        ax_t.legend(); ax_t.grid(alpha=0.3)
        salvar(fig_t, arquivo_tempo)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    for esc, grupo in tabela.groupby(group_col):
        grupo = grupo.sort_values("threads_alvo")
        
        if combine_chunks:
            # Linhas contínuas de fundo
            ax1.plot(grupo["threads_alvo"], grupo["speedup"], "-", color="gray", alpha=0.5)
            ax2.plot(grupo["threads_alvo"], grupo["Eficiência"], "-", color="gray", alpha=0.5)
            
            # Pontos de erro (errorbar) coloridos e agrupados por chunk
            for chunk, subgrupo in grupo.groupby("Chunk"):
                erro_baixo = subgrupo["speedup"] - subgrupo["speedup_min"]
                erro_alto = subgrupo["speedup_max"] - subgrupo["speedup"]
                p = ax1.errorbar(subgrupo["threads_alvo"], subgrupo["speedup"],
                             yerr=[erro_baixo, erro_alto], fmt="o", capsize=4, label=f"Chunk={chunk}")
                             
                erro_baixo_e = subgrupo["Eficiência"] - subgrupo["Eficiência_min"]
                erro_alto_e = subgrupo["Eficiência_max"] - subgrupo["Eficiência"]
                ax2.errorbar(subgrupo["threads_alvo"], subgrupo["Eficiência"],
                             yerr=[erro_baixo_e, erro_alto_e], fmt="o", capsize=4, color=p[0].get_color())
        else:
            erro_baixo = grupo["speedup"] - grupo["speedup_min"]
            erro_alto = grupo["speedup_max"] - grupo["speedup"]
            p = ax1.errorbar(grupo["threads_alvo"], grupo["speedup"],
                         yerr=[erro_baixo, erro_alto], fmt="o-", capsize=4, label=esc)
                         
            erro_baixo_e = grupo["Eficiência"] - grupo["Eficiência_min"]
            erro_alto_e = grupo["Eficiência_max"] - grupo["Eficiência"]
            ax2.errorbar(grupo["threads_alvo"], grupo["Eficiência"],
                         yerr=[erro_baixo_e, erro_alto_e], fmt="o-", capsize=4, color=p[0].get_color(), label=esc)

    ax1.plot(tabela["threads_alvo"].unique(), tabela["threads_alvo"].unique(), "--", color="gray", label="Ideal")
    ax1.set_xlabel("Threads"); ax1.set_ylabel("Speedup")
    ax1.set_title(f"Speedup — {titulo}"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.axhline(1.0, ls="--", color="gray")
    ax2.set_xlabel("Threads"); ax2.set_ylabel("Eficiência")
    ax2.set_ylim(0, 1.15); ax2.set_title(f"Eficiência — {titulo}")
    if not combine_chunks: ax2.legend()
    ax2.grid(alpha=0.3)
    salvar(fig, arquivo)

def grafico_tempo_bruto(tabela, titulo, arquivo, combine_chunks=False):
    if tabela.empty:
        return
    
    group_col = "Escalonamento" if combine_chunks else "Escalonamento_Formatado"
    fig_t, ax_t = plt.subplots(figsize=(7, 4.5))
    
    for esc, grupo in tabela.groupby(group_col):
        grupo = grupo.sort_values("threads_alvo")
        
        if combine_chunks:
            ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "-", color="gray", alpha=0.5)
            for chunk, subgrupo in grupo.groupby("Chunk"):
                ax_t.plot(subgrupo["threads_alvo"], subgrupo["T_Med_Glob"], "o", label=f"Chunk={chunk}")
        else:
            ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "o-", label=esc)
    
    seq_media = tabela["seq_media"].iloc[0]
    ax_t.axhline(seq_media, ls="--", color="black", label=f"Sequencial ({seq_media:.2f}s)")
    ax_t.set_xlabel("Threads"); ax_t.set_ylabel("Tempo Médio (s)")
    ax_t.set_title(f"Tempo de Execução — {titulo}")
    ax_t.legend(); ax_t.grid(alpha=0.3)
    salvar(fig_t, arquivo)

def grafico_balanceamento(tabela, titulo, arquivo):
    if tabela.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for esc, grupo in tabela.groupby("Escalonamento_Formatado"):
        ax.plot(grupo["threads_alvo"], grupo["FatorBal"], "o-", label=esc)
        
    ax.axhline(0.0, ls="--", color="gray", label="Perfeito (0.0)")
    ax.set_xlabel("Threads"); ax.set_ylabel("Fator de Balanço de Carga")
    ax.set_title(f"Desbalanceamento de carga — {titulo}")
    ax.legend(); ax.grid(alpha=0.3)
    salvar(fig, arquivo)

def grafico_weak_scaling(tabela, titulo, arquivo):
    if tabela.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Extrai as resolucoes e o numero de threads
    resolucoes = tabela["Resolucao"].str.split("x").str[0].astype(int)
    threads = tabela["threads_alvo"].astype(int)
    
    # Gráfico 1: Tempo
    ax1.plot(resolucoes, tabela["T_Med_Glob"], "o-", label="Paralelo (Dynamic, C = 16)")
    t1 = tabela["T_Med_Glob"].iloc[0]
    ax1.axhline(t1, ls="--", color="gray", label="Ideal (constante)")
    
    # Anota o número de threads ao lado de cada bolinha no gráfico de Tempo
    for x, y, t in zip(resolucoes, tabela["T_Med_Glob"], threads):
        ax1.annotate(f"  {t}t", (x, y), textcoords="offset points", xytext=(5, -3), ha="left", va="center", fontsize=9, color="black", weight="bold")

    ax1.set_xlabel("Resolução Escalada (N)"); ax1.set_ylabel("Tempo (s)")
    ax1.set_title(f"Tempo — {titulo}"); ax1.legend(); ax1.grid(alpha=0.3)

    # Gráfico 2: Eficiência
    Eficiência_fraca = t1 / tabela["T_Med_Glob"]
    ax2.plot(resolucoes, Eficiência_fraca, "o-", color="darkorange")
    ax2.axhline(1.0, ls="--", color="gray")
    
    # Anota o número de threads ao lado de cada bolinha no gráfico de Eficiência
    for x, y, t in zip(resolucoes, Eficiência_fraca, threads):
        ax2.annotate(f"  {t}t", (x, y), textcoords="offset points", xytext=(5, -3), ha="left", va="center", fontsize=9, color="black", weight="bold")

    ax2.set_xlabel("Resolução Escalada (N)"); ax2.set_ylabel("Eficiência")
    ax2.set_ylim(0, 1.15); ax2.set_title(f"Eficiência de Escala Fraca")
    ax2.grid(alpha=0.3)
    
    salvar(fig, arquivo)
    
def grafico_chunk_effect(tabela, titulo, arquivo):
    if tabela.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Garante que o dataframe esta ordenado numericamente pelo Chunk
    tabela = tabela.sort_values(by="Chunk")
    
    ax.plot(tabela["Chunk"], tabela["T_Med_Glob"], "o-", color="purple", label="16 Threads (Dynamic)")
    
    ax.set_xlabel("Tamanho do Chunk"); ax.set_ylabel("Tempo Médio de Execução (s)")
    ax.set_title(f"Efeito do Chunk — {titulo}")
    ax.set_xscale('log') # Escale logarítmica é ideal para mostrar de 1 até 1024
    ax.set_xticks([1, 2, 4, 16, 64, 256, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Remove notação científica
    ax.legend(); ax.grid(alpha=0.3)
    salvar(fig, arquivo)

def main():
    df = carregar()
    baselines = stats_baseline(df)

    # 1. CASO A
    tabela_a = montar_tabela(df, baselines, "A_padrao_threads")
    grafico_strong_scaling(tabela_a, "Região Padrão", "A_speedup_Eficiência.png", combine_chunks=True)
    grafico_tempo_bruto(tabela_a, "Região Padrão", "A_tempo_bruto.png", combine_chunks=True)

    # 2. CASO B - Adicionado o gráfico de tempo bruto isolado aqui
    tabela_b = montar_tabela(df, baselines, "B_cavalos_threads")
    grafico_strong_scaling(tabela_b, "Cavalos-Marinhos", "B_speedup_Eficiência.png")
    grafico_tempo_bruto(tabela_b, "Cavalos-Marinhos", "B_tempo_bruto.png")
    grafico_balanceamento(tabela_b, "Cavalos-Marinhos", "B_fator_balanceamento.png")

    tabela_c = montar_tabela(df, baselines, "C_weak_scaling")
    grafico_weak_scaling(tabela_c, "Escala Fraca", "C_weak_scaling.png")
    
    tabela_d = df[(df["Modo"].str.strip() == "Paralelo") & (df["caso"] == "D_chunk_effect")].copy()
    grafico_chunk_effect(tabela_d, "Cavalos-Marinhos", "D_efeito_chunk.png")

if __name__ == "__main__":
    main()