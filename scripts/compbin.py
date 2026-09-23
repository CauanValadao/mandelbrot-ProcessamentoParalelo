import subprocess
import filecmp
import os

# ==========================================
# 1. Configurações 
# ==========================================
COMANDO_EXECUCAO = "./mandelbrot.exe"
ARQUIVO_SEQ = "saida/mandelbrot_sequencial.bin" 
ARQUIVO_PAR = "saida/mandelbrot_paralelo.bin"

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


sao_iguais = filecmp.cmp(ARQUIVO_SEQ, ARQUIVO_PAR, shallow=False)

if sao_iguais:
    print("✅ SUCESSO: As saídas são idênticas! A sua paralelização está correta.")
else:
    print("🚨 ERRO: Os arquivos são diferentes!")
    print("   Isso geralmente indica uma condição de corrida (race condition) no OpenMP.")
    print("   Verifique se não há variáveis sendo compartilhadas indevidamente entre as threads (use private/firstprivate se necessário).")