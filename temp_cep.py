import pandas as pd
import requests

# Dicionário para guardar os resultados da API e evitar consultas repetidas
cache = {}

def get_address(cep):
    """
    Consulta a API ViaCEP para obter bairro e cidade a partir do CEP informado.
    Retorna uma tupla (bairro, cidade) ou (None, None) se houver erro.
    """
    # Converte o CEP para string e remove quaisquer espaços e hífens
    cep = str(cep).replace("-", "").strip()
    
    # Se já consultou esse CEP, retorna do cache
    if cep in cache:
        return cache[cep]
    
    # Monta a URL da API
    url = f"https://viacep.com.br/ws/{cep}/json/"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "erro" not in data:
                # Guarda os dados desejados: bairro e cidade (localidade)
                result = (data.get("bairro"), data.get("localidade"))
            else:
                result = (None, None)
        else:
            result = (None, None)
    except Exception as e:
        print(f"Erro ao consultar o CEP {cep}: {e}")
        result = (None, None)
    
    # Armazena o resultado no cache
    cache[cep] = result
    return result

# Lê a planilha CSV informando que a coluna 'CEP' deve ser interpretada como string
df = pd.read_excel('Clientes Gerais.xlsx', dtype={'Código postal (CEP) de envio': str})

# Garante que os CEPs estão em formato de string, removendo espaços extra se houver
df["Código postal (CEP) de envio"] = df["Código postal (CEP) de envio"].apply(lambda x: str(x).strip())

# Cria as novas colunas para Bairro e Cidade aplicando a função get_address em cada CEP
df["Bairro"], df["Cidade"] = zip(*df["Código postal (CEP) de envio"].apply(get_address))

# Exporta o DataFrame final para um arquivo Excel (.xlsx)
df.to_excel("Relatorio_de_vendas_com_endereco.xlsx", index=False)

print("Processamento finalizado. Planilha atualizada salva como 'Relatorio_de_vendas_com_endereco.xlsx'.")
