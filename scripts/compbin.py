import subprocess
import filecmp
import os

# ==========================================
# 1. Configurações (ajuste se necessário)
# ==========================================
# Comando para rodar o seu programa em C. 
# Se você tiver executáveis separados, adicione ambos.
COMANDO_EXECUCAO = "./mandelbrot.exe"
# Caminho das saídas geradas pelo seu código C
ARQUIVO_SEQ = "saida/mandelbrot_sequencial.ppm" # ou mude para .bin se preferir comparar o binário puro
ARQUIVO_PAR = "saida/mandelbrot_paralelo.ppm"

# ==========================================
# 2. Execução
# ==========================================
print("Rodando o programa em C...")
try:
    # Executa o seu código compilado (ele vai gerar/sobrescrever as imagens na pasta saida)
    subprocess.run(COMANDO_EXECUCAO, shell=True, check=True)
except subprocess.CalledProcessError:
    print("❌ Erro durante a execução do programa em C.")
    exit(1)

# ==========================================
# 3. Comparação Binária
# ==========================================
if not os.path.exists(ARQUIVO_SEQ) or not os.path.exists(ARQUIVO_PAR):
    print(f"❌ Erro: Os arquivos não foram encontrados na pasta 'saida/'.")
    exit(1)

print("\nComparando as saídas byte a byte...")

# O shallow=False garante que o Python abra os arquivos e compare o conteúdo binário real,
# em vez de olhar apenas os metadados (como tamanho e data de modificação).
sao_iguais = filecmp.cmp(ARQUIVO_SEQ, ARQUIVO_PAR, shallow=False)

if sao_iguais:
    print("✅ SUCESSO: As saídas são idênticas! A sua paralelização está correta.")
else:
    print("🚨 ERRO: Os arquivos são diferentes!")
    print("   Isso geralmente indica uma condição de corrida (race condition) no OpenMP.")
    print("   Verifique se não há variáveis sendo compartilhadas indevidamente entre as threads (use private/firstprivate se necessário).")