#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <string.h>


typedef struct {
    int is_parallel; // 1 for parallel, 0 for sequential
    int salvar; // 1 to save the image, 0 otherwise
    int run_count; // number of times to run the code
    double re_min;
    double re_max;
    double im_min;
    double im_max;
    int width;
    int height;
    int max_iter;
    int num_threads; // number of threads to use (for parallel execution)
    omp_sched_t escalonamento;
    int C;
    int comparar;
    int calcular_speedup_eficiencia;
    int usar_simetria; // 1 para tentar aproveitar a simetria em torno do eixo real, 0 caso contrario
    char cenario[32];
} MandelbrotParams;

typedef struct {
    double acuracia;
    long long iguais;
    long long diferentes;
    long long diferentes_alem_tolerancia;
    int aprovado; // 1 para sim, 0 para nao
} ComparacaoResultado;

typedef struct {
    double *tempos_execucao;   // tempo de cada execucao, tamanho run_count
    int run_count;
    double **tempos_thread;     // tempo de cada thread, tamanho num_threads
    int num_threads;
    double media_sequencial; // tempo medio do codigo sequencial, valido se comparar == 1
    ComparacaoResultado comparacao; // resultado de comparar_matrizes, valido se comparar == 1
} ResultadoTempos;

typedef struct {
    double tempo_medio_exec;
    double tempo_min_exec;
    double tempo_max_exec;
    double tempo_medio_thread;
    double tempo_min_thread;
    double tempo_max_thread;
    double fator_balanceamento;
    double *medias_por_thread; // NOVO: Armazena a média individual de cada thread
    int num_threads;           // NOVO: Facilita na hora de gravar no arquivo
    double speedup;
    double eficiencia;
    double media_sequencial;
} EstatisticasTempo;

// 1) Vista completa
const MandelbrotParams MANDELBROT_VISTA_COMPLETA = {
    .re_min = -2.0, .re_max = 1.0, .im_min = -1.5, .im_max = 1.5, 
    .width = 4096, .height = 4096, .max_iter = 1000,
    .cenario = "Vista_Completa"
};

// 2) Zoom do vale dos cavalos-marinhos
const MandelbrotParams MANDELBROT_CAVALOS_MARINHOS = {
    .re_min = -0.743643887 - 1.5e-3, .re_max = -0.743643887 + 1.5e-3,
    .im_min = 0.131825904 - 1.5e-3,  .im_max = 0.131825904 + 1.5e-3,
    .width = 4096, .height = 4096, .max_iter = 5000,
    .cenario = "Cavalos_Marinhos"
};

// 3) Mapa de custo por pixel
const MandelbrotParams MANDELBROT_MAPA_CUSTO = {
    .re_min = -2.0, .re_max = 1.0, .im_min = -1.5, .im_max = 1.5, 
    .width = 4096, .height = 4096, .max_iter = 1000,
    .cenario = "Mapa_Custo"
};

// 4) Zoom clássico "espiral"
const MandelbrotParams MANDELBROT_ESPIRAL = {
    .re_min = -0.16070135 - 0.004, .re_max = -0.16070135 + 0.004,
    .im_min = 1.0375665 - 0.004,   .im_max = 1.0375665 + 0.004,
    .width = 4096, .height = 4096, .max_iter = 3000,
    .cenario = "Espiral"
};

int** mandelbrot(double re_min,double re_max,double im_min,double im_max,int width,int height,int max_iter, double* tempos_thread, int usar_simetria);

void free_matriz(int** cont);

void salvar_ppm(const char *nome_arquivo, int **matriz, int width, int height, int max_iter);

void controlador();

ResultadoTempos rodarMandelbrot(MandelbrotParams params);

void free_resultado_tempos(ResultadoTempos *resultado);

int** mandelbrotSequencial(double re_min,double re_max,double im_min,double im_max,int width,int height,int max_iter, int usar_simetria);

// Verifica se a regiao [im_min, im_max] e simetrica em torno do eixo real (im_min == -im_max),
// condicao necessaria para usar mandelbrot(c) == mandelbrot(conj(c)) e economizar metade do calculo.
static int pode_usar_simetria(double im_min, double im_max);

MandelbrotParams interface();

void salvar_csv(MandelbrotParams params, EstatisticasTempo est, ComparacaoResultado comp);

static void mapa_de_cor(int iter, int max_iter, unsigned char *r, unsigned char *g, unsigned char *b);

void salvar_binario(const char *nome_arquivo, int **matriz, int width, int height);

ComparacaoResultado comparar_matrizes(int **paralelo, int **sequencial, int width, int height);

EstatisticasTempo calcular_estatisticas(ResultadoTempos *resultado, int is_parallel);

int main(){
    controlador();
    return 0;
}

void controlador(){
    printf("INTERFACE DE USUARIO\n");
    while(1){
        MandelbrotParams params = interface();
        
        ResultadoTempos resultado = rodarMandelbrot(params);

        EstatisticasTempo est = calcular_estatisticas(&resultado, params.is_parallel);

        salvar_csv(params, est, resultado.comparacao);

        free(est.medias_por_thread);
        free_resultado_tempos(&resultado);
    }
}

MandelbrotParams interface(){
    MandelbrotParams params;
    printf("predefinido: 1 ---------- customizado: 2 ---------- Sair: 3\n");
    int escolha;
    scanf("%d", &escolha);
    while(escolha != 1 && escolha != 2 && escolha != 3 ){
        printf("Opcao invalida, digite novamente: ");
        scanf("%d", &escolha);
    }
    if(escolha == 3) {
        printf("Saindo...\n");
        exit(0);
    }
    if(escolha == 1){
        printf("Escolha o caso de teste:\n");
        printf("1 - Vista completa\n");
        printf("2 - Zoom do vale dos cavalos-marinhos\n");
        printf("3 - Mapa de custo por pixel\n");
        printf("4 - Zoom clássico \"espiral\"\n");
        int caso;
        scanf("%d", &caso);
        while(caso < 1 || caso > 4){
            printf("Opcao invalida, digite novamente: ");
            scanf("%d", &caso);
        }
        switch(caso){
            case 1:
                params = MANDELBROT_VISTA_COMPLETA;
                break;
            case 2:
                params = MANDELBROT_CAVALOS_MARINHOS;
                break;
            case 3:
                params = MANDELBROT_MAPA_CUSTO;
                break;
            case 4:
                params = MANDELBROT_ESPIRAL;
                break;
        }
    }
    else if(escolha == 2){
        printf("Digite o nome do cenario (sem espacos): ");
        scanf("%31s", params.cenario);
        printf("Digite o valor de re_min: ");
        printf("Digite o valor de re_min: ");
        scanf("%lf", &params.re_min);
        printf("Digite o valor de re_max: ");
        scanf("%lf", &params.re_max);
        printf("Digite o valor de im_min: ");
        scanf("%lf", &params.im_min);
        printf("Digite o valor de im_max: ");
        scanf("%lf", &params.im_max);
        printf("Digite a largura (width): ");
        scanf("%d", &params.width);
        printf("Digite a altura (height): ");
        scanf("%d", &params.height);
        printf("Digite o numero maximo de iteracoes (max_iter): ");
        scanf("%d", &params.max_iter);
    }
    escolha = 1;
    while(escolha){
        printf("re_min: %lf, re_max: %lf, im_min: %lf, im_max: %lf, width: %d, height: %d, max_iter: %d\n", params.re_min, params.re_max, params.im_min, params.im_max, params.width, params.height, params.max_iter);
        printf("Deseja alterar algum parametro? (1 - sim, 0 - nao): ");
        scanf("%d", &escolha);
        if(escolha){
            printf("Digite o parametro que deseja alterar (1 - re_min, 2 - re_max, 3 - im_min, 4 - im_max, 5 - width, 6 - height, 7 - max_iter): ");
            int parametro;
            scanf("%d", &parametro);
            switch(parametro){
                case 1:
                    printf("Digite o valor de re_min: ");
                    scanf("%lf", &params.re_min);
                    break;
                case 2:
                    printf("Digite o valor de re_max: ");
                    scanf("%lf", &params.re_max);
                    break;
                case 3:
                    printf("Digite o valor de im_min: ");
                    scanf("%lf", &params.im_min);
                    break;
                case 4:
                    printf("Digite o valor de im_max: ");
                    scanf("%lf", &params.im_max);
                    break;
                case 5:
                    printf("Digite a largura (width): ");
                    scanf("%d", &params.width);
                    break;
                case 6:
                    printf("Digite a altura (height): ");
                    scanf("%d", &params.height);
                    break;
                case 7:
                    printf("Digite o numero maximo de iteracoes (max_iter): ");
                    scanf("%d", &params.max_iter);
                    break;
            }
        }
    }
    printf("Codigo Paralelizado: 1 ---------- codigo Sequencial: 2 ---------- Sair: 3\n");
    scanf("%d", &params.is_parallel);
    while(params.is_parallel != 1 && params.is_parallel != 2 && params.is_parallel != 3){
        printf("Opcao invalida, digite novamente: ");
        scanf("%d", &params.is_parallel);
    }
    if(params.is_parallel == 3) {
        printf("Saindo...\n");
        exit(0);
    }
    if(params.is_parallel == 1){
        printf("Digite a quantidade de threads a serem utilizadas: ");
        scanf("%d", &params.num_threads);
        while(params.num_threads < 1){
            printf("Quantidade de threads invalida, digite novamente: ");
            scanf("%d", &params.num_threads);
        }
        printf("Escolha o tipo de escalonamento (1 - static, 2 - dynamic, 3 - guided): ");
        scanf("%d", &escolha);
        while(escolha < 1 || escolha > 3){
            printf("Tipo de escalonamento invalido, digite novamente: ");
            scanf("%d", &escolha);
        }
        switch(escolha){
            case 1:
                params.escalonamento = omp_sched_static;
                break;
            case 2:
                params.escalonamento = omp_sched_dynamic;
                break;
            case 3:
                params.escalonamento = omp_sched_guided;
                break;
        }
        printf("Digite o valor de C. digite 0 para usar o valor default: ");
        scanf("%d", &params.C);
        while(params.C < 0){
            printf("Valor de C invalido, digite novamente: ");
            scanf("%d", &params.C);
        }
        printf("Deseja comparar o resultado de ambos os codigos? (1 - sim, 0 - nao): ");
        scanf("%d", &params.comparar);
        while(params.comparar != 0 && params.comparar != 1){
            printf("Opcao invalida, digite novamente: ");
            scanf("%d", &params.comparar);
        }
        printf("Deseja calcular o speedup e a eficiencia do codigo paralelo? (1 - sim, 0 - nao): ");
        scanf("%d", &params.calcular_speedup_eficiencia);
        while(params.calcular_speedup_eficiencia != 0 && params.calcular_speedup_eficiencia != 1){
            printf("Opcao invalida, digite novamente: ");
            scanf("%d", &params.calcular_speedup_eficiencia);
        }
    }
    else{
        params.num_threads = 1;
        params.escalonamento = omp_sched_static;
        params.C = 0;
    }

    printf("Deseja tentar usar simetria em torno do eixo real para otimizar o calculo? (1 - sim, 0 - nao): ");
    scanf("%d", &params.usar_simetria);
    while(params.usar_simetria != 0 && params.usar_simetria != 1){
        printf("Opcao invalida, digite novamente: ");
        scanf("%d", &params.usar_simetria);
    }
    if(params.usar_simetria && !pode_usar_simetria(params.im_min, params.im_max)){
        printf("Aviso: a regiao escolhida nao e simetrica em torno do eixo real (im_min != -im_max).\n");
        printf("A simetria sera ignorada nesta execucao.\n");
    }

    printf("Deseja salvar a imagem gerada? (1 - sim, 0 - nao): ");
    scanf("%d", &params.salvar);
    while(params.salvar != 0 && params.salvar != 1){
        printf("Opcao invalida, digite novamente: ");
        scanf("%d", &params.salvar);
    }
    printf("Quantas vezes rodar o codigo: ");
    scanf("%d", &params.run_count);
    while(params.run_count < 1){
        printf("Quantidade de execucoes invalida, digite novamente: ");
        scanf("%d", &params.run_count);
    }
    return params;
}

ResultadoTempos rodarMandelbrot(MandelbrotParams params){
    ResultadoTempos resultado;
    resultado.run_count = params.run_count;
    resultado.num_threads = params.num_threads;
    resultado.tempos_execucao = (double*)malloc(sizeof(double) * params.run_count);
    resultado.tempos_thread = (double**)malloc(sizeof(double*) * params.run_count);
    for(int i = 0; i < params.run_count; i++){
        resultado.tempos_thread[i] = (double*)malloc(sizeof(double) * resultado.num_threads);
    }
    resultado.comparacao = (ComparacaoResultado){0};
    resultado.media_sequencial = -1.0; // Inicializa como -1 para indicar que não foi calculada

    if(params.is_parallel == 1){
        omp_set_num_threads(params.num_threads);
        omp_set_schedule(params.escalonamento, params.C);
        
        for(int i = 0; i < params.run_count; i++){
            double tempo_inicial = omp_get_wtime();

            int** cont = mandelbrot(params.re_min, params.re_max, params.im_min, params.im_max, params.width, params.height, params.max_iter, resultado.tempos_thread[i], params.usar_simetria);
            double tempo_final = omp_get_wtime();
            resultado.tempos_execucao[i] = tempo_final - tempo_inicial;

            if (i == params.run_count - 1) {
                if (params.salvar) {
                    salvar_ppm("mandelbrot_paralelo.ppm", cont, params.width, params.height, params.max_iter);
                    salvar_binario("mandelbrot_paralelo.bin", cont, params.width, params.height);
                }
                double tempo;
                if (params.comparar) {
                    double tempo_inicial_seq = omp_get_wtime();
                    int** cont_sequencial = mandelbrotSequencial(params.re_min, params.re_max, params.im_min, params.im_max, 
                                                                 params.width, params.height, params.max_iter, params.usar_simetria);
                    double tempo_final_seq = omp_get_wtime();
                    tempo = tempo_final_seq - tempo_inicial_seq;
                    resultado.comparacao = comparar_matrizes(cont, cont_sequencial, params.width, params.height);
                    free_matriz(cont_sequencial);
                }
                if(params.calcular_speedup_eficiencia){
                    double tempo_medio_sequencial = 0.0;
                    int i = 0;
                    if(params.comparar){
                        tempo_medio_sequencial = tempo; // Usa o tempo da execução sequencial já calculada
                        i = 1;
                    }
                    for(; i < params.run_count; i++){
                        double tempo_inicial = omp_get_wtime();
                        int** cont_sequencial = mandelbrotSequencial(params.re_min, params.re_max, params.im_min, params.im_max, 
                                                                    params.width, params.height, params.max_iter, params.usar_simetria);
                        double tempo_final = omp_get_wtime();
                        tempo_medio_sequencial += (tempo_final - tempo_inicial);
                        free_matriz(cont_sequencial);
                    }
                    resultado.media_sequencial = tempo_medio_sequencial / params.run_count;
                }
            }

            free_matriz(cont);
        }   

    }
    else{
        for(int i = 0; i < params.run_count; i++){
            double tempo_inicial = omp_get_wtime();

            int** cont = mandelbrotSequencial(params.re_min, params.re_max, params.im_min, params.im_max, params.width, params.height, params.max_iter, params.usar_simetria);
            double tempo_final = omp_get_wtime();
            resultado.tempos_execucao[i] = tempo_final - tempo_inicial;

            if (i == params.run_count - 1 && params.salvar) {
                salvar_ppm("mandelbrot_sequencial.ppm", cont, params.width, params.height, params.max_iter);
                salvar_binario("mandelbrot_sequencial.bin", cont, params.width, params.height);
            }

            free_matriz(cont);
        }
    }
    
    return resultado;
}

void free_resultado_tempos(ResultadoTempos *resultado) {
    if (resultado == NULL) return;
    
    if (resultado->tempos_execucao != NULL) {
        free(resultado->tempos_execucao);
        resultado->tempos_execucao = NULL;
    }
    
    if (resultado->tempos_thread != NULL) {
        for(int i = 0; i < resultado->run_count; i++) {
            if (resultado->tempos_thread[i] != NULL) {
                free(resultado->tempos_thread[i]);
            }
        }
        free(resultado->tempos_thread);
        resultado->tempos_thread = NULL;
    }
}

// Em torno do eixo real, mandelbrot(c) e mandelbrot(conjugado(c)) tem sempre a mesma
// quantidade de iteracoes ate escapar (a orbita de conj(c) e o conjugado da orbita de c).
// Isso so pode ser explorado diretamente por indice de linha quando a grade de alturas
// e simetrica em torno de zero, ou seja, quando im_min == -im_max.
static int pode_usar_simetria(double im_min, double im_max) {
    return fabs(im_min + im_max) < 1e-9;
}

int** mandelbrot(double re_min,double re_max,double im_min,double im_max,int width,int height,int max_iter,double* tempos_thread,int usar_simetria){

    int* matriz = (int*)malloc(sizeof(int)*width*height);
    int** cont = (int**)malloc(sizeof(int*)*width);

    double* c_realVet = (double*)malloc(sizeof(double)*width);
    double* c_imagVet = (double*)malloc(sizeof(double)*height); 

    //#pragma omp parallel for schedule(static)
    for(int i = 0; i < width; i++) {
        cont[i] = &matriz[i * height];
    }

    // 2. Preenchimento do vetor real
    //#pragma omp parallel for schedule(static)
    for(int i = 0; i < width; i++) {
        c_realVet[i] = re_min + ((double)i / (width - 1)) * (re_max - re_min);
    }

    // 3. Preenchimento do vetor imaginário
    //#pragma omp parallel for schedule(static)
    for(int j = 0; j < height; j++) {
        c_imagVet[j] = im_min + ((double)j / (height - 1)) * (im_max - im_min);
    }

    // Com simetria valida, so precisamos calcular metade das linhas (eixo j);
    // a outra metade e espelhada (mesma contagem de iteracoes).
    int simetria_valida = usar_simetria && pode_usar_simetria(im_min, im_max);
    int altura_calculo = simetria_valida ? (height + 1) / 2 : height;

    double tempo_inicial = omp_get_wtime();

    #pragma omp parallel 
    { 
    double tempo_inicio_thread = omp_get_wtime();
    int tid = omp_get_thread_num();
    int k = 0;
    double c_real, c_imag, z_tempReal, z_real, z_imag;

    #pragma omp for schedule(runtime) nowait
    for(int i = 0; i < width; i++){
        c_real = c_realVet[i];
        for(int j = 0; j < altura_calculo; j++){
            c_imag = c_imagVet[j];
            z_real = 0;
            z_imag = 0;
            for(k = 0; k < max_iter; k++){
                z_tempReal = z_real*z_real - z_imag * z_imag + c_real;
                z_imag = 2*z_real*z_imag + c_imag;
                z_real = z_tempReal;

                if(z_real*z_real + z_imag*z_imag > 4)
                    break;
            }
            cont[i][j] = k;
            if(simetria_valida){
                int j_espelho = height - 1 - j;
                if(j_espelho != j){
                    cont[i][j_espelho] = k;
                }
            }
        }
    }
    double tempo_final_thread = omp_get_wtime();
    tempos_thread[tid] = tempo_final_thread - tempo_inicio_thread;
    }

    
    free(c_realVet);
    free(c_imagVet);
    return cont;
}

int** mandelbrotSequencial(double re_min,double re_max,double im_min,double im_max,int width,int height,int max_iter,int usar_simetria){

    int* matriz = (int*)malloc(sizeof(int)*width*height);
    int** cont = (int**)malloc(sizeof(int*)*width);

    double* c_realVet = (double*)malloc(sizeof(double)*width);
    double* c_imagVet = (double*)malloc(sizeof(double)*height); 

    //#pragma omp parallel for schedule(static)
    for(int i = 0; i < width; i++) {
        cont[i] = &matriz[i * height];
    }

    // 2. Preenchimento do vetor real
    for(int i = 0; i < width; i++) {
        c_realVet[i] = re_min + ((double)i / (width - 1)) * (re_max - re_min);
    }

    // 3. Preenchimento do vetor imaginário
    for(int j = 0; j < height; j++) {
        c_imagVet[j] = im_min + ((double)j / (height - 1)) * (im_max - im_min);
    }

    int simetria_valida = usar_simetria && pode_usar_simetria(im_min, im_max);
    int altura_calculo = simetria_valida ? (height + 1) / 2 : height;

    double tempo_inicial = omp_get_wtime();

    { 
    int k = 0;
    double c_real, c_imag, z_tempReal, z_real, z_imag;


    for(int i = 0; i < width; i++){
        c_real = c_realVet[i];
        for(int j = 0; j < altura_calculo; j++){
            c_imag = c_imagVet[j];
            z_real = 0;
            z_imag = 0;
            for(k = 0; k < max_iter; k++){
                z_tempReal = z_real*z_real - z_imag * z_imag + c_real;
                z_imag = 2*z_real*z_imag + c_imag;
                z_real = z_tempReal;

                if(z_real*z_real + z_imag*z_imag > 4)
                    break;
            }
            cont[i][j] = k;
            if(simetria_valida){
                int j_espelho = height - 1 - j;
                if(j_espelho != j){
                    cont[i][j_espelho] = k;
                }
            }
        }
    }
    double tempo_final = omp_get_wtime();
}

    
    free(c_realVet);
    free(c_imagVet);
    return cont;
}

void free_matriz(int** cont) {
    if (cont != NULL) {
        free(cont[0]); // Libera o bloco contíguo
        free(cont);    // Libera os ponteiros das linhas
    }
}

void salvar_ppm(const char *nome_arquivo, int **matriz, int width, int height, int max_iter) {
    // Abre em "wb" (write binary), essencial para o formato P6
    FILE *arquivo = fopen(nome_arquivo, "wb");
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo para escrita!\n");
        return;
    }

    // 1. Escreve o cabeçalho PPM (P6)
    // 255 é o limite padrão para os canais RGB
    fprintf(arquivo, "P6\n");
    fprintf(arquivo, "%d %d\n", width, height);
    fprintf(arquivo, "255\n");

    // 2. Escreve os valores da matriz
    for (int j = 0; j < height; j++) {         // Linha (Y)
        for (int i = 0; i < width; i++) {     // Coluna (X)
            unsigned char r, g, b;
            
            // Mapeia o número de iterações do pixel atual para a cor RGB
            mapa_de_cor(matriz[i][j], max_iter, &r, &g, &b);
            
            // Escreve os 3 bytes (R, G, B) diretamente no arquivo
            fputc(r, arquivo);
            fputc(g, arquivo);
            fputc(b, arquivo);
        }
    }

    // 3. Fecha o arquivo
    fclose(arquivo);
    printf("Imagem salva com sucesso em '%s'!\n", nome_arquivo);
}

void salvar_csv(MandelbrotParams params, EstatisticasTempo est, ComparacaoResultado comp) {
    FILE *arquivo_csv = fopen("benchmark_resultados.csv", "a");
    if (arquivo_csv == NULL) {
        printf("Erro ao criar/abrir arquivo CSV.\n");
        return;
    }

    // Grava o cabeçalho alinhado (larguras fixas)
    fseek(arquivo_csv, 0, SEEK_END);
    if (ftell(arquivo_csv) == 0) {
        fprintf(arquivo_csv, "%-10s, %-16s, %-7s, %-13s, %-5s, %-10s, %-7s, %-5s, %-12s, %-12s, %-12s, %-12s, %-10s, %-10s, %-10s, %-8s, %-12s, %-12s, %-12s, %-10s, %-10s, %-12s, %-10s, %s\n",
                "Modo", "Cenario", "Threads", "Escalonamento", "Chunk", "Resolucao", "MaxIter", "Vezes",
                "T_Med_Glob", "T_Min_Glob", "T_Max_Glob", "T_Med_Seq", "Speedup", "Eficiencia", "FatorBal", "Simetria",
                "T_Min_Thr", "T_Max_Thr", "T_Med_Thr",
                "Acuracia_%", "Diferentes", "Dif_Alem_Tol", "Aprovado", "Medias_Threads");    }

    // Configuração de strings
    const char* modo_str = (params.is_parallel == 1) ? "Paralelo" : "Sequencial";
    const char* esc_str = "N/A";
    if (params.is_parallel == 1) {
        if (params.escalonamento == omp_sched_static) esc_str = "Static";
        else if (params.escalonamento == omp_sched_dynamic) esc_str = "Dynamic";
        else if (params.escalonamento == omp_sched_guided) esc_str = "Guided";
    }
    int simetria_efetiva = params.usar_simetria && pode_usar_simetria(params.im_min, params.im_max);
    const char* simetria_str = simetria_efetiva ? "Aplicada" : "Nao";
    const char* aprovado_str = params.comparar ? (comp.aprovado ? "Sim" : "Nao") : "N/A";
    
    char res_str[32];
    snprintf(res_str, sizeof(res_str), "%dx%d", params.width, params.height);

    // Formata o vetor de médias de threads em uma única string, separados por |
    char buffer_medias[4096] = "";
    int offset = 0;
    for (int i = 0; i < est.num_threads; i++) {
        int written = snprintf(buffer_medias + offset, sizeof(buffer_medias) - offset, "%.6lf|", est.medias_por_thread[i]);
        if (written > 0 && (size_t)(offset + written) < sizeof(buffer_medias)) offset += written;
    }
    if (offset > 0) buffer_medias[offset - 1] = '\0'; // Remove o último pipe

    // Grava a linha formatada alinhada exatamente com o cabeçalho
fprintf(arquivo_csv, "%-10s, %-16s, %-7d, %-13s, %-5d, %-10s, %-7d, %-5d, %-12.6lf, %-12.6lf, %-12.6lf, %-12.6lf, %-10.4lf, %-10.4lf, %-10.6lf, %-8s, %-12.6lf, %-12.6lf, %-12.6lf, %-10.4lf, %-10lld, %-12lld, %-10s, %s\n",            
            modo_str, 
            params.cenario,
            est.num_threads, 
            esc_str, 
            params.C, 
            res_str, 
            params.max_iter, 
            params.run_count, 
            est.tempo_medio_exec, est.tempo_min_exec, est.tempo_max_exec, 
            est.media_sequencial, est.speedup, est.eficiencia, 
            est.fator_balanceamento, simetria_str,
            est.tempo_min_thread, est.tempo_max_thread, est.tempo_medio_thread,
            params.comparar ? comp.acuracia : 0.0, 
            params.comparar ? comp.diferentes : 0LL, 
            params.comparar ? comp.diferentes_alem_tolerancia : 0LL, 
            aprovado_str, 
            buffer_medias);
    
    fclose(arquivo_csv);
    printf("Resultados exportados para 'benchmark_resultados.csv' com sucesso!\n\n");
}

static void mapa_de_cor(int iter, int max_iter, unsigned char *r, unsigned char *g, unsigned char *b) {
    // Cor de dentro do fractal (quando atinge o limite de iterações)
    if (iter >= max_iter) {
        *r = 11; *g = 29; *b = 58; // Um tom de azul escuro
        return;
    }
    
    // Normalização cores fundo
    double t = log(1.0 + iter) / log(1.0 + max_iter);
    
    // double t = (double)iter / max_iter; // Opção linear comentada pelo seu colega
    
    double um_menos_t = 1.0 - t;
    
    // Cálculo dos canais RGB usando polinômios
    double rd = 9.0 * um_menos_t * t * t * t;
    double gd = 15.0 * um_menos_t * um_menos_t * t * t;
    double bd = 8.5 * um_menos_t * um_menos_t * um_menos_t * t;
    
    *r = (unsigned char)(255.0 * rd);
    *g = (unsigned char)(255.0 * gd);
    *b = (unsigned char)(255.0 * bd);
}

void salvar_binario(const char *nome_arquivo, int **matriz, int width, int height) {
    FILE *arquivo = fopen(nome_arquivo, "wb");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo binario para escrita: %s\n", nome_arquivo);
        return;
    }

    /*
     * O arquivo deve conter inteiros de 32 bits em ordem row-major:
     *
     * linha 0: matriz[0][0], matriz[1][0], ..., matriz[width-1][0]
     * linha 1: matriz[0][1], matriz[1][1], ..., matriz[width-1][1]
     * ...
     *
     * A matriz em memoria usa matriz[x][y], por isso a ordem dos lacos
     * abaixo e y (linha) depois x (coluna).
     */
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int32_t valor = (int32_t)matriz[i][j];

            if (fwrite(&valor, sizeof(int32_t), 1, arquivo) != 1) {
                printf("Erro ao escrever o arquivo binario.\n");
                fclose(arquivo);
                return;
            }
        }
    }

    fclose(arquivo);

    printf("Matriz binaria salva com sucesso em '%s' (%d x %d, row-major).\n",
           nome_arquivo, width, height);
}

ComparacaoResultado comparar_matrizes(int **paralelo, int **sequencial, int width, int height) {
    long long total = (long long)width * height;
    ComparacaoResultado res = {0.0, 0, 0, 0, 0};

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int a = paralelo[i][j];
            int b = sequencial[i][j];
            
            if (a == b) {
                res.iguais++;
            } else {
                res.diferentes++;
                if (abs(a - b) > 1) {
                    res.diferentes_alem_tolerancia++;
                }
            }
        }
    }

    res.acuracia = (total > 0) ? ((double)res.iguais / (double)total) * 100.0 : 0.0;
    double pct_diferentes = (total > 0) ? ((double)res.diferentes / (double)total) * 100.0 : 0.0;

    res.aprovado = (pct_diferentes <= 0.01) && (res.diferentes_alem_tolerancia == 0);
    
    if (res.aprovado) {
        printf("Validacao de corretude: APROVADO\n");
    } else {
        printf("Validacao de corretude: REPROVADO\n");
    }

    return res;
}

EstatisticasTempo calcular_estatisticas(ResultadoTempos *resultado, int is_parallel) {
    EstatisticasTempo est = {0};
    est.media_sequencial = resultado->media_sequencial;
    est.num_threads = resultado->num_threads;
    est.medias_por_thread = (double*)calloc(est.num_threads, sizeof(double));

    // 1. Cálculos do Tempo Global de Execução
    if (resultado->run_count > 0 && resultado->tempos_execucao != NULL) {
        est.tempo_min_exec = resultado->tempos_execucao[0];
        est.tempo_max_exec = resultado->tempos_execucao[0];
        double soma_exec = 0.0;
        
        for (int i = 0; i < resultado->run_count; i++) {
            double t = resultado->tempos_execucao[i];
            if (t < est.tempo_min_exec) est.tempo_min_exec = t;
            if (t > est.tempo_max_exec) est.tempo_max_exec = t;
            soma_exec += t;
        }
        est.tempo_medio_exec = soma_exec / resultado->run_count;
    }

    // 2. Cálculos das Threads e Fator de Balanceamento
    est.fator_balanceamento = 1.0;
    
    if (is_parallel == 1 && resultado->tempos_thread != NULL && resultado->num_threads > 0) {
        est.tempo_min_thread = resultado->tempos_thread[0][0];
        est.tempo_max_thread = resultado->tempos_thread[0][0];
        
        double soma_fator = 0.0;
        double soma_global_threads = 0.0;
        
        // Loop que calcula o Fator e os Extremos
        for (int r = 0; r < resultado->run_count; r++) {
            double max_thread_rodada = resultado->tempos_thread[r][0];
            double min_thread_rodada = resultado->tempos_thread[r][0]; // Novo
            double soma_thread_rodada = 0.0;
            
            for (int t = 0; t < resultado->num_threads; t++) {
                double tempo_t = resultado->tempos_thread[r][t];
                
                if (tempo_t < est.tempo_min_thread) est.tempo_min_thread = tempo_t;
                if (tempo_t > est.tempo_max_thread) est.tempo_max_thread = tempo_t;
                
                if (tempo_t > max_thread_rodada) max_thread_rodada = tempo_t;
                if (tempo_t < min_thread_rodada) min_thread_rodada = tempo_t; // Novo
                
                soma_thread_rodada += tempo_t;
                soma_global_threads += tempo_t;
            }
            
            // Fórmula exigida pelo professor: (Tmax - Tmin) / Tmax
            double fator_rodada = (max_thread_rodada > 0.0) ? ((max_thread_rodada - min_thread_rodada) / max_thread_rodada) : 0.0;
            soma_fator += fator_rodada;
        }
        
        // Fecha as médias gerais e o fator
        est.fator_balanceamento = soma_fator / resultado->run_count;
        est.tempo_medio_thread = soma_global_threads / (resultado->run_count * resultado->num_threads);
        
        // Loop NOVO: Calcula a média INDIVIDUAL de cada thread
        for (int t = 0; t < resultado->num_threads; t++) {
            double soma_t = 0.0;
            for (int r = 0; r < resultado->run_count; r++) {
                soma_t += resultado->tempos_thread[r][t];
            }
            est.medias_por_thread[t] = soma_t / resultado->run_count;
        }
        if (resultado->media_sequencial > 0.0 && est.tempo_medio_exec > 0.0) {
            est.speedup = resultado->media_sequencial / est.tempo_medio_exec;
            est.eficiencia = est.speedup / est.num_threads;
        }
        else {
            est.speedup = -1.0; // Indica que não foi calculado
            est.eficiencia = -1.0; // Indica que não foi calculado
        }
        
    } else if (is_parallel != 1) {
        est.tempo_min_thread = est.tempo_min_exec;
        est.tempo_max_thread = est.tempo_max_exec;
        est.tempo_medio_thread = est.tempo_medio_exec;
        est.medias_por_thread[0] = est.tempo_medio_exec;
    }

    return est;
}