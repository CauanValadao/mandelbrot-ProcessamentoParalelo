#!/usr/bin/env python3
"""
Le 'resultados_finais.csv' e produz as analises finais:
Speedup, Eficiência, Fator de Balanceamento e Escalabilidade.
"""
import os
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

os.makedirs("saida", exist_ok=True)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "saida/resultados_finais.csv"
OUT_DIR = Path("graficos")
OUT_DIR.mkdir(exist_ok=True)

def carregar():
    if not os.path.exists(CSV_PATH):
        print(f"Erro: Arquivo '{CSV_PATH}' não encontrado.")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    # Limpa nomes de colunas
    df.columns = [c.strip() for c in df.columns]
    num_cols = ["Threads", "Chunk", "T_Med_Glob", "T_Min_Glob", "T_Max_Glob","T_Med_Serial", "FatorBal", "repeticao"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["threads_alvo"] = df["Threads"]
    return df

def stats_baseline(df):
    seq = df[df["Modo"].str.strip() == "Sequencial"]
    linhas = []
    if "baseline_key" not in seq.columns:
        seq["baseline_key"] = seq["caso"] # Fallback caso nao tenha a coluna
        
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
    fig.savefig(OUT_DIR / nome, dpi=300)
    plt.close(fig)
    print("Gerado:", nome)

def grafico_strong_scaling(tabela, titulo, arquivo, arquivo_tempo=None, combine_chunks=False):
    if tabela.empty:
        return
        
    group_col = "Escalonamento" if combine_chunks else "Escalonamento_Formatado"

    if arquivo_tempo:
        fig_t, ax_t = plt.subplots(figsize=(7, 4.5))
        for esc, grupo in tabela.groupby(group_col):
            grupo = grupo.sort_values("threads_alvo")
            if combine_chunks:
                ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "-", color="gray", alpha=0.5)
                for chunk, subgrupo in grupo.groupby("Chunk"):
                    ax_t.plot(subgrupo["threads_alvo"], subgrupo["T_Med_Glob"], "o", label=f"Chunk={chunk}")
            else:
                ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "o-", label=esc)
        
        if not tabela["seq_media"].isna().all():
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
            ax1.plot(grupo["threads_alvo"], grupo["speedup"], "-", color="gray", alpha=0.5)
            ax2.plot(grupo["threads_alvo"], grupo["Eficiência"], "-", color="gray", alpha=0.5)
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
    
    if not tabela["seq_media"].isna().all():
        seq_media = tabela["seq_media"].iloc[0]
        ax_t.axhline(seq_media, ls="--", color="black", label=f"Sequencial ({seq_media:.2f}s)")
        
    ax_t.set_xlabel("Threads"); ax_t.set_ylabel("Tempo Médio (s)")
    ax_t.set_title(f"Tempo de Execução — {titulo}")
    ax_t.legend(); ax_t.grid(alpha=0.3)
    salvar(fig_t, arquivo)

def grafico_tempo_zoom(tabela, titulo, arquivo):
    if tabela.empty:
        return
    
    # Filtra os dados apenas para 8 e 16 threads
    df_zoom = tabela[tabela["threads_alvo"].isin([8, 16])]
    if df_zoom.empty:
        return

    fig_t, ax_t = plt.subplots(figsize=(7, 4.5))
    
    for esc, grupo in df_zoom.groupby("Escalonamento_Formatado"):
        grupo = grupo.sort_values("threads_alvo")
        ax_t.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], "o-", label=esc)
    
    # Omitimos a linha do sequencial propositalmente para permitir o zoom nos valores baixos
    
    ax_t.set_xlabel("Threads")
    ax_t.set_ylabel("Tempo Médio (s)")
    ax_t.set_title(f"Tempo de Execução (Zoom 8 e 16t) — {titulo}")
    ax_t.set_xticks([8, 16])
    ax_t.legend()
    ax_t.grid(alpha=0.3)
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

def grafico_balanceamento_chunk(tabela, titulo, arquivo):
    if tabela.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tabela = tabela.sort_values(by="Chunk")

    ax.plot(tabela["Chunk"], tabela["FatorBal"], "o-", color="purple", label="16 Threads (Dynamic)")

    ax.set_xlabel("Tamanho do Chunk")
    ax.set_ylabel("Fator de Balanço de Carga")
    ax.set_title(f"Fator de Balanceamento × Tamanho do Chunk — {titulo}")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 16, 64, 256, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(alpha=0.3)
    salvar(fig, arquivo) 

def grafico_weak_scaling(tabela, titulo, arquivo):
    if tabela.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    resolucoes = tabela["Resolucao"].str.split("x").str[0].astype(int)
    threads = tabela["threads_alvo"].astype(int)
    
    ax1.plot(resolucoes, tabela["T_Med_Glob"], "o-", label="Paralelo (Dynamic, C = 16)")
    t1 = tabela["T_Med_Glob"].iloc[0]
    ax1.axhline(t1, ls="--", color="gray", label="Ideal (constante)")
    
    for x, y, t in zip(resolucoes, tabela["T_Med_Glob"], threads):
        ax1.annotate(f"  {t}t", (x, y), textcoords="offset points", xytext=(5, -3), ha="left", va="center", fontsize=9, color="black", weight="bold")

    ax1.set_xlabel("Resolução Escalada (N)"); ax1.set_ylabel("Tempo (s)")
    ax1.set_title(f"Tempo — {titulo}"); ax1.legend(); ax1.grid(alpha=0.3)

    Eficiência_fraca = t1 / tabela["T_Med_Glob"]
    ax2.plot(resolucoes, Eficiência_fraca, "o-", color="darkorange")
    ax2.axhline(1.0, ls="--", color="gray")
    
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
    
    tabela = tabela.sort_values(by="Chunk")
    ax.plot(tabela["Chunk"], tabela["T_Med_Glob"], "o-", color="purple", label="16 Threads (Dynamic)")
    
    ax.set_xlabel("Tamanho do Chunk"); ax.set_ylabel("Tempo Médio de Execução (s)")
    ax.set_title(f"Efeito do Chunk — {titulo}")
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 16, 64, 256, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(); ax.grid(alpha=0.3)
    salvar(fig, arquivo)

def grafico_simetria(tabela, titulo, arquivo):
    if tabela.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Limpa espacos da coluna Simetria gerada pelo C para garantir o agrupamento
    if "Simetria" in tabela.columns:
        tabela["Simetria"] = tabela["Simetria"].astype(str).str.strip()
    
    # Plota agrupando se a simetria foi aplicada ou nao
    for simetria, grupo in tabela.groupby("Simetria"):
        grupo = grupo.sort_values("threads_alvo")
        
        # Estilizacao: Verde continuo se COM simetria, Vermelho tracejado se SEM
        if simetria == "Aplicada":
            label = "Com Simetria"
            marker = "o-"
            color = "green"
        else:
            label = "Sem Simetria"
            marker = "s--"
            color = "red"
            
        ax.plot(grupo["threads_alvo"], grupo["T_Med_Glob"], marker, color=color, label=label)

    ax.set_xlabel("Threads")
    ax.set_ylabel("Tempo Médio de Execução (s)")
    ax.set_title(f"Impacto da Simetria no Tempo — {titulo}")
    ax.legend()
    ax.grid(alpha=0.3)
    salvar(fig, arquivo)

def grafico_composicao_tempo(tabela, titulo, arquivo):
    """
    Compara o tempo médio da parte serial com o tempo médio da parte
    paralela, calculado como:

        T_paralelo = T_total - T_serial

    O eixo Y utiliza escala logarítmica para evidenciar diferenças
    de magnitude entre as duas parcelas.
    """
    if tabela.empty or "T_Med_Serial" not in tabela.columns:
        return

    df_plot = tabela.copy()

    # Somente as quantidades de threads solicitadas.
    df_plot = df_plot[df_plot["threads_alvo"].isin([1, 2, 4, 8, 16])].copy()
    df_plot = df_plot.sort_values("threads_alvo")
    df_plot = df_plot.drop_duplicates(subset=["threads_alvo"])

    # T_Med_Serial é a média do tempo da parte serial.
    df_plot["T_Med_Glob"] = pd.to_numeric(df_plot["T_Med_Glob"], errors="coerce")
    df_plot["T_Med_Serial"] = pd.to_numeric(
        df_plot["T_Med_Serial"], errors="coerce"
    )

    df_plot = df_plot.dropna(subset=["T_Med_Glob", "T_Med_Serial"])

    # Tempo médio da parte paralela:
    # média do tempo total - média do tempo serial.
    df_plot["T_Med_Paralelo"] = (
        df_plot["T_Med_Glob"] - df_plot["T_Med_Serial"]
    )

    # Escala logarítmica não aceita valores <= 0.
    df_plot = df_plot[
        (df_plot["T_Med_Serial"] > 0) &
        (df_plot["T_Med_Paralelo"] > 0)
    ].copy()

    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_plot["threads_alvo"],
        df_plot["T_Med_Paralelo"],
        "o-",
        linewidth=2,
        label="Parte paralela"
    )

    ax.plot(
        df_plot["threads_alvo"],
        df_plot["T_Med_Serial"],
        "s-",
        linewidth=2,
        label="Parte serial"
    )

    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])

    ax.set_xlabel("Número de Threads")
    ax.set_ylabel("Tempo Médio de Execução (s) — escala logarítmica")
    ax.set_title(f"Composição do Tempo de Execução — {titulo}")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Exibe os valores calculados sobre os pontos.
    for _, linha in df_plot.iterrows():
        ax.annotate(
            f'{linha["T_Med_Paralelo"]:.6f}s',
            (linha["threads_alvo"], linha["T_Med_Paralelo"]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8
        )

        ax.annotate(
            f'{linha["T_Med_Serial"]:.6f}s',
            (linha["threads_alvo"], linha["T_Med_Serial"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    salvar(fig, arquivo)

def gerar_tabela_relatorio(df, baselines):
    if df.empty:
        return
        
    # Faz o merge para recuperar o tempo sequencial base de cada cenário
    df_calc = df.merge(baselines[["baseline_key", "seq_media"]], on="baseline_key", how="left")
    
    # Recalcula Speedup e Eficiência matematicamente corretos em relação à base sequencial
    df_calc["Speedup_Real"] = df_calc.apply(
        lambda r: r["seq_media"] / r["T_Med_Glob"] if r["Modo"].strip() == "Paralelo" else 1.0, axis=1
    )
    df_calc["Eficiencia_Real"] = df_calc.apply(
        lambda r: r["Speedup_Real"] / r["Threads"] if r["Modo"].strip() == "Paralelo" else 1.0, axis=1
    )
    
    # Substitui os -1 impressos pelo C pelos cálculos reais do Python
    df_calc["Speedup"] = df_calc["Speedup_Real"]
    df_calc["Eficiencia"] = df_calc["Eficiencia_Real"]
    
    colunas_importantes = [
        "caso", "Modo", "Threads", "Escalonamento", "Chunk", 
        "Resolucao", "T_Med_Serial","T_Med_Glob", "Speedup", "Eficiencia", "FatorBal"
    ]
    
    colunas_usar = [c for c in colunas_importantes if c in df_calc.columns]
    df_relatorio = df_calc[colunas_usar].copy()
    
    cols_numericas = df_relatorio.select_dtypes(include=['float64']).columns
    df_relatorio[cols_numericas] = df_relatorio[cols_numericas].round(4)
    
    df_relatorio = df_relatorio.drop_duplicates(subset=["caso", "Threads", "Escalonamento", "Chunk", "Resolucao"])
    
    # Salva em formato Markdown
    caminho_md = OUT_DIR / "tabela_bruta_relatorio.md"
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(df_relatorio.to_markdown(index=False))
        
    # Salva em formato CSV
    caminho_csv = OUT_DIR / "tabela_bruta_relatorio.csv"
    df_relatorio.to_csv(caminho_csv, index=False)
        
    print(f"Tabela consolidada gerada com sucesso em:\n - {caminho_md}\n - {caminho_csv}")

def main():
    df = carregar()
    baselines = stats_baseline(df)

    tabela_a = montar_tabela(df, baselines, "A_padrao_threads")
    if not tabela_a.empty:
        grafico_strong_scaling(tabela_a, "Região Padrão", "A_speedup_eficiência.png", combine_chunks=True)
        grafico_tempo_bruto(tabela_a, "Região Padrão", "A_tempo_bruto.png", combine_chunks=True)
        grafico_composicao_tempo(tabela_a, "Região Padrão", "A_composicao_tempo.png")

    tabela_simetria = df[(df["Modo"].str.strip() == "Paralelo") & (df["caso"].isin(["A_padrao_threads", "A_padrao_sem_simetria"]))].copy()
    if not tabela_simetria.empty:
        grafico_simetria(tabela_simetria, "Região Padrão", "A_impacto_simetria.png")

    tabela_b = montar_tabela(df, baselines, "B_cavalos_threads")
    if not tabela_b.empty:
        grafico_strong_scaling(tabela_b, "Cavalos-Marinhos", "B_speedup_eficiência.png")
        grafico_tempo_bruto(tabela_b, "Cavalos-Marinhos", "B_tempo_bruto.png")
        grafico_balanceamento(tabela_b, "Cavalos-Marinhos", "B_fator_balanceamento.png")
        grafico_tempo_zoom(tabela_b, "Cavalos-Marinhos", "B_tempo_bruto_zoom.png")

    tabela_c = montar_tabela(df, baselines, "C_weak_scaling")
    if not tabela_c.empty:
        grafico_weak_scaling(tabela_c, "Escala Fraca", "C_weak_scaling.png")
    
    tabela_d = df[(df["Modo"].str.strip() == "Paralelo") & (df["caso"] == "D_chunk_effect")].copy()
    if not tabela_d.empty:
        grafico_chunk_effect(tabela_d, "Cavalos-Marinhos", "D_efeito_chunk.png")
        grafico_balanceamento_chunk(tabela_d, "Cavalos-Marinhos", "D_balanceamento_chunk.png")

    # CHAMADA CORRIGIDA: Envia as métricas base (baselines) para a função de exportação
    gerar_tabela_relatorio(df, baselines)

if __name__ == "__main__":
    main()