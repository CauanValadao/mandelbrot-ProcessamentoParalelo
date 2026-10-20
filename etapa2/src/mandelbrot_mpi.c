/*
 * DEC107 - Etapa 2: Mandelbrot (escape-time) com MPI
 *
 * Estrategias de distribuicao (por LINHAS da imagem, row-major):
 *   1 = block  : faixas contiguas de linhas (decomposicao estatica por blocos)
 *   2 = cyclic : distribuicao ciclica de linhas (block-cyclic com --chunk linhas; chunk=1 -> ciclica pura)
 *   3 = master : mestre-trabalhador com balanceamento dinamico (tarefas de --chunk linhas)
 *                rank 0 = mestre (nao calcula), ranks 1..P-1 = trabalhadores
 *
 * Coleta do resultado:
 *   block / cyclic -> MPI_Gatherv (+ reordenacao no root para cyclic)
 *   master         -> mestre recebe cada tarefa e copia para a imagem
 *
 * Metricas medidas por rank, em cada execucao:
 *   comp   = tempo dentro do kernel de calculo
 *   espera = tempo ocioso bloqueado (barreira antes do Gatherv / Recv bloqueante)
 *   comm   = tempo de comunicacao ativa (Gatherv + reordenacao / Send / copia no mestre)
 *   total  = tempo de parede da execucao (maximo entre os ranks), SEM escrita em disco
 * O tempo de escrita em disco (PPM + binario) e medido e reportado a parte (T_Escrita).
 *
 * Compilacao:  mpicc -O3 -march=native -o mandelbrot_mpi.exe src/mandelbrot_mpi.c -lm
 * Execucao  :  mpiexec -n 8 ./mandelbrot_mpi.exe --preset cavalos --estrategia master --chunk 4 --runs 10
 *              (sem argumentos abre o menu interativo)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <mpi.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

_Static_assert(sizeof(int) == 4, "o formato de saida exige int de 32 bits");

enum { MODO_SEQ = 0, MODO_MPI = 1 };
enum { EST_BLOCK = 1, EST_CYCLIC = 2, EST_MASTER = 3 };
enum { TAG_WORK = 1, TAG_STOP = 2, TAG_RESULT = 3 };

typedef struct {
    char   cenario[32];
    double re_min, re_max, im_min, im_max;
    int    width, height, max_iter;
    int    modo;        /* MODO_SEQ / MODO_MPI */
    int    estrategia;  /* EST_* */
    int    chunk;       /* linhas por bloco (cyclic) ou por tarefa (master) */
    int    runs;        /* execucoes cronometradas */
    int    seq_runs;    /* execucoes seriais para T_Med_Seq (0 = nao mede) */
    int    comparar;    /* 1 = compara com a versao serial (corretude) */
    int    salvar;      /* 1 = grava PPM + binario */
    int    erro;        /* 1 = argumentos invalidos */
    char   csv[160];
} Params;

typedef struct { double total, comp, espera, comm; } TempoRank;

typedef struct {
    double acuracia;
    long long iguais, diferentes, alem_tol;
    int aprovado;
} Comparacao;

/* ------------------------------------------------------------------ */
/* Kernel (identico para serial e paralelo => resultados identicos)    */
/* ------------------------------------------------------------------ */
NOINLINE
static void calcular_linhas(const Params *p, const double *c_real, int j0, int nrows, int *out) {
    const int w = p->width, h = p->height, max_iter = p->max_iter;
    for (int r = 0; r < nrows; r++) {
        int j = j0 + r;
        double c_imag = p->im_min + ((double)j / (h - 1)) * (p->im_max - p->im_min);
        int *lin = out + (size_t)r * w;
        for (int i = 0; i < w; i++) {
            double cr = c_real[i], zr = 0.0, zi = 0.0, zt;
            int k;
            for (k = 0; k < max_iter; k++) {
                zt = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + c_imag;
                zr = zt;
                if (zr * zr + zi * zi > 4.0) break;
            }
            lin[i] = k;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Layouts de distribuicao                                             */
/* ------------------------------------------------------------------ */
static void layout_block(int h, int P, int r, int *start, int *n) {
    int base = h / P, rem = h % P;
    *n = base + (r < rem ? 1 : 0);
    *start = r * base + (r < rem ? r : rem);
}

static int linhas_ciclico(int h, int P, int r, int C) {
    int n = 0, nb = (h + C - 1) / C;
    for (int b = r; b < nb; b += P) {
        int ini = b * C;
        n += (ini + C <= h) ? C : h - ini;
    }
    return n;
}

static void *xmalloc(size_t bytes, MPI_Comm comm) {
    void *ptr = malloc(bytes > 0 ? bytes : 1);
    if (!ptr) {
        fprintf(stderr, "Falha ao alocar %.1f MB\n", bytes / 1048576.0);
        MPI_Abort(comm, 2);
    }
    return ptr;
}

/* ------------------------------------------------------------------ */
/* Execucoes                                                           */
/* ------------------------------------------------------------------ */
static void executar_serial(const Params *p, const double *c_real, int *image, TempoRank *t) {
    double t0 = MPI_Wtime();
    calcular_linhas(p, c_real, 0, p->height, image);
    double t1 = MPI_Wtime();
    t->total = t->comp = t1 - t0;
    t->espera = t->comm = 0.0;
}

/* 1) Faixas contiguas + Gatherv */
static void executar_block(const Params *p, const double *c_real, int rank, int P,
                           int *image, TempoRank *t, MPI_Comm comm) {
    const int w = p->width, h = p->height;
    int start, n;
    layout_block(h, P, rank, &start, &n);

    int *local, *counts = NULL, *displs = NULL;
    if (rank == 0) {
        local = image; /* rank 0 calcula direto na posicao final (start == 0) */
        counts = (int *)xmalloc(sizeof(int) * P, comm);
        displs = (int *)xmalloc(sizeof(int) * P, comm);
        for (int r = 0; r < P; r++) {
            int s, c;
            layout_block(h, P, r, &s, &c);
            counts[r] = c * w;
            displs[r] = s * w;
        }
    } else {
        local = (int *)xmalloc((size_t)n * w * sizeof(int), comm);
    }

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    calcular_linhas(p, c_real, start, n, local);
    double t1 = MPI_Wtime();
    MPI_Barrier(comm); /* isola o tempo ocioso (desbalanceamento) do custo real de comunicacao */
    double t2 = MPI_Wtime();
    if (rank == 0)
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, image, counts, displs, MPI_INT, 0, comm);
    else
        MPI_Gatherv(local, n * w, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, comm);
    double t3 = MPI_Wtime();

    t->comp = t1 - t0; t->espera = t2 - t1; t->comm = t3 - t2; t->total = t3 - t0;

    if (rank == 0) { free(counts); free(displs); } else free(local);
}

/* 2) Ciclico por blocos de C linhas + Gatherv + reordenacao no root */
static void executar_cyclic(const Params *p, const double *c_real, int rank, int P,
                            int *image, TempoRank *t, MPI_Comm comm) {
    const int w = p->width, h = p->height, C = p->chunk;
    const int nb = (h + C - 1) / C;
    int n = linhas_ciclico(h, P, rank, C);
    int *local = (int *)xmalloc((size_t)n * w * sizeof(int), comm);
    int *counts = NULL, *displs = NULL, *tmp = NULL;
    if (rank == 0) {
        counts = (int *)xmalloc(sizeof(int) * P, comm);
        displs = (int *)xmalloc(sizeof(int) * P, comm);
        int acc = 0;
        for (int r = 0; r < P; r++) {
            counts[r] = linhas_ciclico(h, P, r, C) * w;
            displs[r] = acc;
            acc += counts[r];
        }
        tmp = (int *)xmalloc((size_t)h * w * sizeof(int), comm);
    }

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    size_t off = 0;
    for (int b = rank; b < nb; b += P) {
        int ini = b * C;
        int len = (ini + C <= h) ? C : h - ini;
        calcular_linhas(p, c_real, ini, len, local + off);
        off += (size_t)len * w;
    }
    double t1 = MPI_Wtime();
    MPI_Barrier(comm);
    double t2 = MPI_Wtime();
    MPI_Gatherv(local, n * w, MPI_INT, tmp, counts, displs, MPI_INT, 0, comm);
    if (rank == 0) { /* devolve cada bloco para a linha correta da imagem */
        for (int r = 0; r < P; r++) {
            size_t o = (size_t)displs[r];
            for (int b = r; b < nb; b += P) {
                int ini = b * C;
                int len = (ini + C <= h) ? C : h - ini;
                memcpy(image + (size_t)ini * w, tmp + o, (size_t)len * w * sizeof(int));
                o += (size_t)len * w;
            }
        }
    }
    double t3 = MPI_Wtime();

    t->comp = t1 - t0; t->espera = t2 - t1; t->comm = t3 - t2; t->total = t3 - t0;

    free(local);
    if (rank == 0) { free(counts); free(displs); free(tmp); }
}

/* 3) Mestre-trabalhador (dinamico) */
static void executar_master(const Params *p, const double *c_real, int rank, int P,
                            int *image, TempoRank *t, MPI_Comm comm) {
    const int w = p->width, h = p->height, C = p->chunk;

    if (P == 1) { /* sem trabalhadores: o unico processo calcula tudo */
        executar_serial(p, c_real, image, t);
        return;
    }

    int *buf = (int *)xmalloc((2 + (size_t)C * w) * sizeof(int), comm); /* [inicio, nlinhas, dados...] */
    double comp = 0, espera = 0, comm_t = 0;

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    if (rank == 0) {
        int next = 0, pend = 0, dummy = 0;
        for (int wk = 1; wk < P; wk++) {
            double a = MPI_Wtime();
            if (next < h) {
                int s = next;
                MPI_Send(&s, 1, MPI_INT, wk, TAG_WORK, comm);
                next += C; pend++;
            } else {
                MPI_Send(&dummy, 1, MPI_INT, wk, TAG_STOP, comm);
            }
            comm_t += MPI_Wtime() - a;
        }
        while (pend > 0) {
            MPI_Status st;
            double a = MPI_Wtime();
            MPI_Recv(buf, (int)(2 + (size_t)C * w), MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, comm, &st);
            double b = MPI_Wtime();
            espera += b - a;
            memcpy(image + (size_t)buf[0] * w, buf + 2, (size_t)buf[1] * w * sizeof(int));
            pend--;
            if (next < h) {
                int s = next;
                MPI_Send(&s, 1, MPI_INT, st.MPI_SOURCE, TAG_WORK, comm);
                next += C; pend++;
            } else {
                MPI_Send(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_STOP, comm);
            }
            comm_t += MPI_Wtime() - b; /* copia + envio da proxima tarefa = sobrecarga do mestre */
        }
    } else {
        for (;;) {
            MPI_Status st;
            int s;
            double a = MPI_Wtime();
            MPI_Recv(&s, 1, MPI_INT, 0, MPI_ANY_TAG, comm, &st);
            double b = MPI_Wtime();
            espera += b - a;
            if (st.MPI_TAG == TAG_STOP) break;
            int n = (s + C <= h) ? C : h - s;
            calcular_linhas(p, c_real, s, n, buf + 2);
            double c = MPI_Wtime();
            comp += c - b;
            buf[0] = s; buf[1] = n;
            MPI_Send(buf, (int)(2 + (size_t)n * w), MPI_INT, 0, TAG_RESULT, comm);
            comm_t += MPI_Wtime() - c;
        }
    }
    double t1 = MPI_Wtime();
    t->total = t1 - t0; t->comp = comp; t->espera = espera; t->comm = comm_t;
    free(buf);
}

/* ------------------------------------------------------------------ */
/* Corretude                                                           */
/* ------------------------------------------------------------------ */
static Comparacao comparar_matrizes(const int *par, const int *seq, int w, int h) {
    Comparacao res = {0.0, 0, 0, 0, 0};
    long long total = (long long)w * h;
    for (long long k = 0; k < total; k++) {
        if (par[k] == seq[k]) res.iguais++;
        else {
            res.diferentes++;
            if (abs(par[k] - seq[k]) > 1) res.alem_tol++;
        }
    }
    res.acuracia = total > 0 ? 100.0 * (double)res.iguais / (double)total : 0.0;
    double pct = total > 0 ? 100.0 * (double)res.diferentes / (double)total : 0.0;
    res.aprovado = (pct <= 0.01) && (res.alem_tol == 0);
    printf("Validacao de corretude: %s (diferentes=%lld, alem da tolerancia=%lld)\n",
           res.aprovado ? "APROVADO" : "REPROVADO", res.diferentes, res.alem_tol);
    return res;
}

/* ------------------------------------------------------------------ */
/* Saida (PPM + binario int32 row-major)                               */
/* ------------------------------------------------------------------ */
static void mapa_de_cor(int iter, int max_iter, unsigned char *r, unsigned char *g, unsigned char *b) {
    if (iter >= max_iter) { *r = 11; *g = 29; *b = 58; return; }
    double t = log(1.0 + iter) / log(1.0 + max_iter), u = 1.0 - t;
    *r = (unsigned char)(255.0 * 9.0 * u * t * t * t);
    *g = (unsigned char)(255.0 * 15.0 * u * u * t * t);
    *b = (unsigned char)(255.0 * 8.5 * u * u * u * t);
}

static void salvar_ppm(const char *nome, const int *img, int w, int h, int max_iter) {
    FILE *f = fopen(nome, "wb");
    if (!f) { printf("Erro ao abrir '%s' para escrita\n", nome); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    unsigned char *lin = (unsigned char *)malloc((size_t)w * 3);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++)
            mapa_de_cor(img[(size_t)j * w + i], max_iter, &lin[3 * i], &lin[3 * i + 1], &lin[3 * i + 2]);
        fwrite(lin, 1, (size_t)w * 3, f);
    }
    free(lin);
    fclose(f);
}

static void salvar_binario(const char *nome, const int *img, int w, int h) {
    FILE *f = fopen(nome, "wb");
    if (!f) { printf("Erro ao abrir '%s' para escrita\n", nome); return; }
    /* a imagem ja esta em row-major: linha 0 (x=0..w-1), linha 1, ... */
    if (fwrite(img, sizeof(int32_t), (size_t)w * h, f) != (size_t)w * h)
        printf("Erro ao escrever '%s'\n", nome);
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* Estatisticas e CSV                                                  */
/* ------------------------------------------------------------------ */
typedef struct {
    double t_med, t_min, t_max;
    double comp_min, comp_max, comp_med;
    double fator_bal, razao_max_med;
    double espera_med, comm_rank0, comm_med;
    double *medias_comp; /* media por rank do tempo de calculo */
    int P;
} Estat;

static Estat calcular_estatisticas(const double *t_tot, const double *rk, int runs, int P, int master) {
    Estat e;
    memset(&e, 0, sizeof(e));
    e.P = P;
    e.medias_comp = (double *)calloc((size_t)P, sizeof(double));
    int q0 = (master && P > 1) ? 1 : 0; /* no modo mestre-trabalhador o balanceamento so faz sentido entre trabalhadores */
    int nq = P - q0;

    e.t_min = e.t_max = t_tot[0];
    e.comp_min = 1e300; e.comp_max = 0;
    double s_t = 0, s_comp = 0, s_esp = 0, s_comm = 0, s_fator = 0, s_razao = 0, s_c0 = 0;

    for (int r = 0; r < runs; r++) {
        double tt = t_tot[r];
        if (tt < e.t_min) e.t_min = tt;
        if (tt > e.t_max) e.t_max = tt;
        s_t += tt;

        double mx = 0, mn = 1e300, soma = 0;
        for (int q = 0; q < P; q++) {
            const double *x = &rk[((size_t)r * P + q) * 3]; /* comp, espera, comm */
            e.medias_comp[q] += x[0] / runs;
            if (q < q0) continue;
            if (x[0] > mx) mx = x[0];
            if (x[0] < mn) mn = x[0];
            soma += x[0];
            s_esp += x[1];
            s_comm += x[2];
        }
        s_c0 += rk[((size_t)r * P) * 3 + 2];
        if (mx < e.comp_min * 0 + 1e-300) mx = 1e-300;
        if (mn < e.comp_min) e.comp_min = mn;
        if (mx > e.comp_max) e.comp_max = mx;
        s_comp += soma;
        s_fator += (mx - mn) / mx;           /* (Tmax - Tmin) / Tmax, como na Etapa 1 */
        s_razao += mx / (soma / nq);         /* Tmax / Tmedio */
    }
    e.t_med = s_t / runs;
    e.comp_med = s_comp / ((double)runs * nq);
    e.espera_med = s_esp / ((double)runs * nq);
    e.comm_med = s_comm / ((double)runs * nq);
    e.comm_rank0 = s_c0 / runs;
    e.fator_bal = s_fator / runs;
    e.razao_max_med = s_razao / runs;
    return e;
}

static void salvar_csv(const Params *p, const Estat *e, double t_seq, double t_escrita,
                       const Comparacao *cmp, int tem_cmp, int P) {
    FILE *f = fopen(p->csv, "a");
    if (!f) { printf("Erro ao abrir '%s'\n", p->csv); return; }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "Modo,Cenario,Procs,Estrategia,Chunk,Resolucao,MaxIter,Vezes,"
                   "T_Med_Glob,T_Min_Glob,T_Max_Glob,T_Med_Seq,Speedup,Eficiencia,Eficiencia_Trab,"
                   "FatorBal,Razao_Max_Med,T_Comp_Min,T_Comp_Max,T_Comp_Med,T_Espera_Med,"
                   "T_Comm_Rank0,T_Comm_Med,T_Escrita,Acuracia_%%,Diferentes,Dif_Alem_Tol,Aprovado,Medias_Procs\n");

    const char *modo = p->modo == MODO_MPI ? "MPI" : "Sequencial";
    const char *est = "N/A";
    if (p->modo == MODO_MPI)
        est = p->estrategia == EST_BLOCK ? "Block" : p->estrategia == EST_CYCLIC ? "Cyclic" : "MasterWorker";
    double speedup = -1, efic = -1, efic_trab = -1;
    if (p->modo == MODO_MPI && t_seq > 0 && e->t_med > 0) {
        speedup = t_seq / e->t_med;
        efic = speedup / P;
        efic_trab = (p->estrategia == EST_MASTER && P > 1) ? speedup / (P - 1) : efic;
    }
    size_t cap = (size_t)P * 20 + 16;
    char *medias = (char *)calloc(cap, 1);
    size_t off = 0;
    for (int q = 0; q < P; q++) off += (size_t)snprintf(medias + off, cap - off, "%s%.6f", q ? "|" : "", e->medias_comp[q]);

    fprintf(f, "%s,%s,%d,%s,%d,%dx%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.6f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%lld,%lld,%s,%s\n",
            modo, p->cenario, P, est, p->modo == MODO_MPI ? p->chunk : 0, p->width, p->height, p->max_iter, p->runs,
            e->t_med, e->t_min, e->t_max, t_seq, speedup, efic, efic_trab,
            e->fator_bal, e->razao_max_med, e->comp_min, e->comp_max, e->comp_med, e->espera_med,
            e->comm_rank0, e->comm_med, t_escrita,
            tem_cmp ? cmp->acuracia : 0.0, tem_cmp ? cmp->diferentes : 0LL, tem_cmp ? cmp->alem_tol : 0LL,
            tem_cmp ? (cmp->aprovado ? "Sim" : "Nao") : "N/A", medias);
    free(medias);
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* Interface (CLI + menu interativo)                                   */
/* ------------------------------------------------------------------ */
static int aplicar_preset(Params *p, const char *nome) {
    if (!strcmp(nome, "vista") || !strcmp(nome, "mapa")) {
        p->re_min = -2.0; p->re_max = 1.0; p->im_min = -1.5; p->im_max = 1.5;
        p->width = p->height = 4096; p->max_iter = 1000;
        strcpy(p->cenario, !strcmp(nome, "vista") ? "Vista_Completa" : "Mapa_Custo");
    } else if (!strcmp(nome, "cavalos")) {
        p->re_min = -0.743643887 - 1.5e-3; p->re_max = -0.743643887 + 1.5e-3;
        p->im_min = 0.131825904 - 1.5e-3;  p->im_max = 0.131825904 + 1.5e-3;
        p->width = p->height = 4096; p->max_iter = 5000;
        strcpy(p->cenario, "Cavalos_Marinhos");
    } else if (!strcmp(nome, "espiral")) {
        p->re_min = -0.16070135 - 0.004; p->re_max = -0.16070135 + 0.004;
        p->im_min = 1.0375665 - 0.004;   p->im_max = 1.0375665 + 0.004;
        p->width = p->height = 4096; p->max_iter = 3000;
        strcpy(p->cenario, "Espiral");
    } else return 0;
    return 1;
}

static void uso(void) {
    printf("Uso: mpiexec -n P mandelbrot_mpi [opcoes]   (sem opcoes: menu interativo)\n"
           "  --preset vista|cavalos|mapa|espiral   regiao/resolucao/max_iter predefinidos (padrao: vista)\n"
           "  --regiao re_min re_max im_min im_max  regiao customizada\n"
           "  --res N | --width W --height H        resolucao\n"
           "  --max-iter M     --nome ROTULO\n"
           "  --modo seq|mpi   (padrao mpi; seq roda a versao serial no rank 0, use -n 1)\n"
           "  --estrategia block|cyclic|master      (padrao block)\n"
           "  --chunk C        linhas por bloco (cyclic) ou por tarefa (master); 0 = padrao (1)\n"
           "  --runs K         execucoes cronometradas (padrao 1)\n"
           "  --seq-runs K     execucoes seriais p/ calcular speedup dentro do programa (padrao 0)\n"
           "  --comparar       valida contra a versao serial\n"
           "  --salvar         grava PPM e binario em saida/\n"
           "  --csv ARQ        arquivo de resultados (padrao saida/benchmark_mpi.csv)\n");
}

static int parse_args(int argc, char **argv, Params *p) {
    memset(p, 0, sizeof(*p));
    aplicar_preset(p, "vista");
    p->modo = MODO_MPI; p->estrategia = EST_BLOCK; p->runs = 1;
    strcpy(p->csv, "saida/benchmark_mpi.csv");
    int has_reg = 0, has_res = 0, has_w = 0, has_h = 0, has_it = 0, has_nome = 0;
    double reg[4] = {0}; int res = 0, w = 0, h = 0, it = 0; char nome[32] = "";
    const char *preset = "vista";

#define NEED(n) do { if (i + (n) >= argc) { printf("Faltam valores para %s\n", a); return 0; } } while (0)
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--preset")) { NEED(1); preset = argv[++i]; }
        else if (!strcmp(a, "--regiao")) { NEED(4); for (int k = 0; k < 4; k++) reg[k] = atof(argv[++i]); has_reg = 1; }
        else if (!strcmp(a, "--res")) { NEED(1); res = atoi(argv[++i]); has_res = 1; }
        else if (!strcmp(a, "--width")) { NEED(1); w = atoi(argv[++i]); has_w = 1; }
        else if (!strcmp(a, "--height")) { NEED(1); h = atoi(argv[++i]); has_h = 1; }
        else if (!strcmp(a, "--max-iter")) { NEED(1); it = atoi(argv[++i]); has_it = 1; }
        else if (!strcmp(a, "--nome")) { NEED(1); snprintf(nome, sizeof(nome), "%s", argv[++i]); has_nome = 1; }
        else if (!strcmp(a, "--modo")) { NEED(1); const char *m = argv[++i]; p->modo = !strcmp(m, "seq") ? MODO_SEQ : MODO_MPI; }
        else if (!strcmp(a, "--estrategia")) {
            NEED(1); const char *m = argv[++i];
            if (!strcmp(m, "block")) p->estrategia = EST_BLOCK;
            else if (!strcmp(m, "cyclic")) p->estrategia = EST_CYCLIC;
            else if (!strcmp(m, "master")) p->estrategia = EST_MASTER;
            else { printf("Estrategia invalida: %s\n", m); return 0; }
        }
        else if (!strcmp(a, "--chunk")) { NEED(1); p->chunk = atoi(argv[++i]); }
        else if (!strcmp(a, "--runs")) { NEED(1); p->runs = atoi(argv[++i]); }
        else if (!strcmp(a, "--seq-runs")) { NEED(1); p->seq_runs = atoi(argv[++i]); }
        else if (!strcmp(a, "--comparar")) p->comparar = 1;
        else if (!strcmp(a, "--salvar")) p->salvar = 1;
        else if (!strcmp(a, "--csv")) { NEED(1); snprintf(p->csv, sizeof(p->csv), "%s", argv[++i]); }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { uso(); return 0; }
        else { printf("Opcao desconhecida: %s\n", a); uso(); return 0; }
    }
#undef NEED
    if (!aplicar_preset(p, preset)) { printf("Preset invalido: %s\n", preset); return 0; }
    if (has_reg) { p->re_min = reg[0]; p->re_max = reg[1]; p->im_min = reg[2]; p->im_max = reg[3]; }
    if (has_res) p->width = p->height = res;
    if (has_w) p->width = w;
    if (has_h) p->height = h;
    if (has_it) p->max_iter = it;
    if (has_nome) snprintf(p->cenario, sizeof(p->cenario), "%s", nome);
    return 1;
}

static int menu_interativo(Params *p) {
    memset(p, 0, sizeof(*p));
    strcpy(p->csv, "saida/benchmark_mpi.csv");
    int op;
    printf("Caso: 1-Vista completa 2-Cavalos-marinhos 3-Mapa de custo 4-Espiral: ");
    if (scanf("%d", &op) != 1) return 0;
    const char *nomes[] = {"vista", "cavalos", "mapa", "espiral"};
    if (op < 1 || op > 4) return 0;
    aplicar_preset(p, nomes[op - 1]);
    printf("Resolucao (lado, 0 = padrao 4096): ");
    int r; if (scanf("%d", &r) != 1) return 0;
    if (r > 0) p->width = p->height = r;
    printf("Modo: 1-MPI 2-Serial: ");
    if (scanf("%d", &op) != 1) return 0;
    p->modo = (op == 2) ? MODO_SEQ : MODO_MPI;
    if (p->modo == MODO_MPI) {
        printf("Estrategia: 1-Block 2-Cyclic 3-Master/Worker: ");
        if (scanf("%d", &p->estrategia) != 1) return 0;
        printf("Chunk (linhas por bloco/tarefa, 0 = padrao): ");
        if (scanf("%d", &p->chunk) != 1) return 0;
        printf("Comparar com serial? (1/0): ");
        if (scanf("%d", &p->comparar) != 1) return 0;
        printf("Execucoes seriais para speedup (0 = nao medir): ");
        if (scanf("%d", &p->seq_runs) != 1) return 0;
    }
    printf("Salvar PPM e binario? (1/0): ");
    if (scanf("%d", &p->salvar) != 1) return 0;
    printf("Quantas execucoes: ");
    if (scanf("%d", &p->runs) != 1) return 0;
    return 1;
}

static int validar(Params *p) {
    if (p->width < 2 || p->height < 2 || p->max_iter < 1 || p->runs < 1 || p->seq_runs < 0 || p->chunk < 0) return 0;
    if (p->re_max <= p->re_min || p->im_max <= p->im_min) return 0;
    if (p->estrategia < EST_BLOCK || p->estrategia > EST_MASTER) return 0;
    if (p->chunk == 0) p->chunk = 1;
    if ((long long)p->chunk * p->width + 2 > (long long)INT_MAX / 2) return 0;
    if ((long long)p->width * p->height > (long long)INT_MAX) return 0; /* contagens do Gatherv sao int */
    return 1;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    Params p;
    if (rank == 0) {
        int ok = (argc > 1) ? parse_args(argc, argv, &p) : menu_interativo(&p);
        if (ok && !validar(&p)) { printf("Parametros invalidos.\n"); ok = 0; }
        if (!ok) { memset(&p, 0, sizeof(p)); p.erro = 1; }
    }
    MPI_Bcast(&p, (int)sizeof(Params), MPI_BYTE, 0, comm);
    if (p.erro) { MPI_Finalize(); return 1; }

    const int w = p.width, h = p.height;
    const int master_mode = (p.modo == MODO_MPI && p.estrategia == EST_MASTER);

    /* ---------- modo serial: so o rank 0 trabalha ---------- */
    int Pef = (p.modo == MODO_SEQ) ? 1 : P;
    if (p.modo == MODO_SEQ && rank != 0) { MPI_Finalize(); return 0; }
    if (p.modo == MODO_SEQ && P > 1 && rank == 0) printf("Aviso: modo serial; use -n 1 para evitar processos ociosos.\n");

    MKDIR("saida");

    double *c_real = (double *)xmalloc((size_t)w * sizeof(double), comm);
    for (int i = 0; i < w; i++) c_real[i] = p.re_min + ((double)i / (w - 1)) * (p.re_max - p.re_min);

    int *image = NULL;
    if (rank == 0) image = (int *)xmalloc((size_t)w * h * sizeof(int), comm);

    double *t_tot = NULL, *rk = NULL;
    if (rank == 0) {
        t_tot = (double *)xmalloc((size_t)p.runs * sizeof(double), comm);
        rk = (double *)xmalloc((size_t)p.runs * Pef * 3 * sizeof(double), comm);
    }

    if (rank == 0)
        printf("[%s] %dx%d max_iter=%d modo=%s estrategia=%d chunk=%d procs=%d runs=%d\n", p.cenario, w, h,
               p.max_iter, p.modo == MODO_SEQ ? "serial" : "mpi", p.estrategia, p.chunk, Pef, p.runs);

    /* ---------- execucoes cronometradas ---------- */
    for (int r = 0; r < p.runs; r++) {
        TempoRank t = {0, 0, 0, 0};
        if (p.modo == MODO_SEQ) {
            executar_serial(&p, c_real, image, &t);
            t_tot[r] = t.total;
            rk[(size_t)r * 3 + 0] = t.comp; rk[(size_t)r * 3 + 1] = 0; rk[(size_t)r * 3 + 2] = 0;
        } else {
            switch (p.estrategia) {
                case EST_BLOCK:  executar_block(&p, c_real, rank, P, image, &t, comm); break;
                case EST_CYCLIC: executar_cyclic(&p, c_real, rank, P, image, &t, comm); break;
                default:         executar_master(&p, c_real, rank, P, image, &t, comm); break;
            }
            double total_max;
            MPI_Reduce(&t.total, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            double mine[3] = {t.comp, t.espera, t.comm};
            MPI_Gather(mine, 3, MPI_DOUBLE, rank == 0 ? &rk[(size_t)r * P * 3] : NULL, 3, MPI_DOUBLE, 0, comm);
            if (rank == 0) t_tot[r] = total_max;
        }
        if (rank == 0) printf("  execucao %d/%d: %.4f s\n", r + 1, p.runs, t_tot[r]);
    }

    if (rank == 0) {
        /* ---------- escrita em disco (reportada a parte) ---------- */
        double t_escrita = 0.0;
        if (p.salvar) {
            char fn[256];
            double a = MPI_Wtime();
            if (p.modo == MODO_SEQ) {
                snprintf(fn, sizeof(fn), "saida/%s_sequencial.ppm", p.cenario);
                salvar_ppm(fn, image, w, h, p.max_iter);
                salvar_binario("saida/mandelbrot_sequencial.bin", image, w, h);
            } else {
                const char *en = p.estrategia == EST_BLOCK ? "block" : p.estrategia == EST_CYCLIC ? "cyclic" : "master";
                snprintf(fn, sizeof(fn), "saida/%s_mpi_%s.ppm", p.cenario, en);
                salvar_ppm(fn, image, w, h, p.max_iter);
                salvar_binario("saida/mandelbrot_mpi.bin", image, w, h);
            }
            t_escrita = MPI_Wtime() - a;
            printf("Escrita em disco: %.3f s\n", t_escrita);
        }

        /* ---------- baseline serial (speedup) e corretude ---------- */
        double t_seq = -1.0;
        Comparacao cmp = {0, 0, 0, 0, 0};
        int tem_cmp = 0;
        if (p.modo == MODO_MPI && (p.seq_runs > 0 || p.comparar)) {
            int *ref = (int *)xmalloc((size_t)w * h * sizeof(int), comm);
            int n_ref = p.seq_runs > 0 ? p.seq_runs : 1;
            double soma = 0;
            for (int r = 0; r < n_ref; r++) {
                TempoRank ts;
                executar_serial(&p, c_real, ref, &ts);
                soma += ts.total;
            }
            if (p.seq_runs > 0) t_seq = soma / n_ref;
            if (p.comparar) { cmp = comparar_matrizes(image, ref, w, h); tem_cmp = 1; }
            free(ref);
        }

        /* ---------- estatisticas + CSV ---------- */
        Estat e = calcular_estatisticas(t_tot, rk, p.runs, Pef, master_mode);
        if (p.modo == MODO_SEQ) { /* na versao serial, T_Med_Seq = proprio tempo medio */
            t_seq = e.t_med;
        }
        salvar_csv(&p, &e, t_seq, t_escrita, &cmp, tem_cmp, Pef);

        printf("T_medio=%.4f s | FatorBal=%.4f | Tmax/Tmed=%.3f | comp_med=%.4f espera_med=%.4f comm_med=%.4f",
               e.t_med, e.fator_bal, e.razao_max_med, e.comp_med, e.espera_med, e.comm_med);
        if (t_seq > 0 && p.modo == MODO_MPI)
            printf(" | Speedup=%.3f Eficiencia=%.3f", t_seq / e.t_med, t_seq / e.t_med / Pef);
        printf("\nResultados em '%s'\n", p.csv);

        free(e.medias_comp); free(t_tot); free(rk); free(image);
    }
    free(c_real);
    MPI_Finalize();
    return 0;
}