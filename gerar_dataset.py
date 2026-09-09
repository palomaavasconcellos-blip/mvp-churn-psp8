"""
Script utilizado para GERAR a base de dados sintética e propositalmente "suja"
usada neste MVP.

Contexto simulado: um restaurante (ou rede de restaurantes) com programa de
fidelidade/cadastro de clientes — muito comum hoje em dia via app próprio,
plataforma de delivery ou cartão de pontos — mantém uma base de clientes
exportada diretamente do sistema de CRM/PDV (ponto de venda). Esse tipo de
exportação, na prática profissional, quase sempre chega com problemas de
qualidade: valores ausentes, categorias digitadas de formas diferentes, tipos
de dado errados, datas em formatos inconsistentes, duplicidades e valores
fora do intervalo esperado (outliers).

Este script existe para dar transparência total sobre como os dados foram
produzidos (nenhuma "caixa-preta"): toda imperfeição inserida é documentada
abaixo, com sua respectiva probabilidade de ocorrência. Isso permite que, no
notebook, cada etapa de limpeza seja justificada apontando exatamente qual
problema está sendo corrigido.

Execução: python3 gerar_dataset.py
Saída: ../dados/clientes_restaurante_brutos.csv
"""

import numpy as np
import pandas as pd

# Semente fixa para reprodutibilidade: qualquer pessoa que rodar este script
# obtém exatamente a mesma base "suja".
RNG = np.random.default_rng(seed=42)
N = 2200  # número de registros (clientes cadastrados no programa de fidelidade)


def gerar_base_limpa(n: int) -> pd.DataFrame:
    """Gera, primeiro, uma base coerente e limpa (o 'mundo real' por trás dos
    dados). É sobre essa base coerente que depois aplicamos as sujeiras —
    assim garantimos que existe, de fato, um padrão real que o modelo de
    machine learning poderá aprender, e não apenas ruído.
    """
    idade = RNG.integers(18, 80, size=n)
    tempo_relacionamento_meses = RNG.integers(1, 72, size=n)
    categoria_fidelidade = RNG.choice(
        ["Bronze", "Prata", "Ouro"], size=n, p=[0.45, 0.35, 0.20]
    )

    ticket_base = {"Bronze": 45.00, "Prata": 90.00, "Ouro": 160.00}
    ticket_medio_mensal = np.array([ticket_base[c] for c in categoria_fidelidade]) + RNG.normal(0, 8, size=n)
    ticket_medio_mensal = np.round(np.clip(ticket_medio_mensal, 10, None), 2)

    forma_pagamento = RNG.choice(
        ["Cartão de Crédito", "Pix", "Boleto"], size=n, p=[0.4, 0.4, 0.2]
    )
    reclamacoes_registradas = RNG.poisson(1.2, size=n)
    pedidos_no_show = np.clip(RNG.exponential(2, size=n).astype(int), 0, 15)
    nota_satisfacao = np.clip(RNG.normal(3.4, 1.0, size=n).round(), 1, 5).astype(int)
    genero = RNG.choice(["Feminino", "Masculino"], size=n, p=[0.52, 0.48])

    datas_cadastro = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        RNG.integers(0, 1400, size=n), unit="D"
    )

    # --- Regra "real" de inatividade do cliente ------------------------
    # Construída para refletir causas plausíveis de um cliente parar de
    # pedir/visitar um restaurante: clientes muito novos no cadastro (ainda
    # não criaram o hábito), muitas reclamações, muitos pedidos/reservas com
    # no-show, baixa satisfação declarada, categoria de fidelidade mais baixa
    # (menor engajamento) e pagamento por boleto (menos integrado ao
    # app/fluxo de recompra rápida) aumentam a chance de o cliente ficar
    # inativo (não fazer nenhum novo pedido/visita em um período recente).
    logit = (
        -2.6
        + 2.6 * (tempo_relacionamento_meses < 6)
        + 0.65 * reclamacoes_registradas
        + 0.20 * pedidos_no_show
        - 1.3 * (nota_satisfacao - 3)
        + 1.0 * (categoria_fidelidade == "Bronze")
        + 0.4 * (forma_pagamento == "Boleto")
    )
    prob_inatividade = 1 / (1 + np.exp(-logit))
    inativo = RNG.binomial(1, prob_inatividade)
    inativo_str = np.where(inativo == 1, "Sim", "Não")

    df = pd.DataFrame(
        {
            "cliente_id": np.arange(1, n + 1),
            "idade": idade,
            "genero": genero,
            "tempo_relacionamento_meses": tempo_relacionamento_meses,
            "categoria_fidelidade": categoria_fidelidade,
            "ticket_medio_mensal": ticket_medio_mensal,
            "forma_pagamento": forma_pagamento,
            "reclamacoes_registradas": reclamacoes_registradas,
            "pedidos_no_show": pedidos_no_show,
            "nota_satisfacao": nota_satisfacao,
            "data_cadastro": datas_cadastro,
            "cliente_inativo": inativo_str,
        }
    )
    return df


def sujar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica, de forma controlada e documentada, os problemas de qualidade
    que qualquer cientista de dados encontra ao pegar uma base "do mundo
    real". Cada bloco abaixo corresponde a UM tipo de problema, para que o
    notebook possa citar exatamente qual célula de código o resolve.
    """
    df = df.copy()
    n = len(df)

    # 1) Valores ausentes (missing values) em várias colunas, com taxas
    #    diferentes — como acontece em sistemas reais, onde alguns campos são
    #    obrigatórios no cadastro e outros não.
    for coluna, taxa in {
        "idade": 0.04,
        "genero": 0.03,
        "ticket_medio_mensal": 0.05,
        "nota_satisfacao": 0.10,
        "forma_pagamento": 0.02,
    }.items():
        idx = RNG.choice(n, size=int(n * taxa), replace=False)
        df.loc[idx, coluna] = np.nan

    # 2) Inconsistência de categorias: a mesma informação escrita de formas
    #    diferentes (maiúsculas/minúsculas, abreviações, idioma).
    mapa_genero_sujo = {"Feminino": ["Feminino", "feminino", "F", "fem"],
                        "Masculino": ["Masculino", "masculino", "M", "masc"]}
    def sujar_genero(valor):
        if pd.isna(valor):
            return valor
        return RNG.choice(mapa_genero_sujo[valor])
    df["genero"] = df["genero"].apply(sujar_genero)

    mapa_fidelidade_sujo = {
        "Bronze": ["Bronze", "bronze", "BRONZE"],
        "Prata": ["Prata", "prata", "PRATA"],
        "Ouro": ["Ouro", "ouro", "OURO"],
    }
    df["categoria_fidelidade"] = df["categoria_fidelidade"].apply(lambda v: RNG.choice(mapa_fidelidade_sujo[v]))

    mapa_inativo_sujo = {"Sim": ["Sim", "sim", "S", "Yes", "1"],
                        "Não": ["Não", "não", "N", "No", "0"]}
    df["cliente_inativo"] = df["cliente_inativo"].apply(lambda v: RNG.choice(mapa_inativo_sujo[v]))

    # 3) Coluna numérica armazenada como texto, com símbolo de moeda —
    #    problema clássico de exportação de sistemas financeiros/PDV.
    idx_moeda = RNG.choice(n, size=int(n * 0.25), replace=False)
    df["ticket_medio_mensal"] = df["ticket_medio_mensal"].astype(object)
    for i in idx_moeda:
        valor = df.loc[i, "ticket_medio_mensal"]
        if pd.notna(valor):
            df.loc[i, "ticket_medio_mensal"] = f"R$ {valor:.2f}".replace(".", ",")

    # 4) Datas em múltiplos formatos dentro da MESMA coluna.
    def formatar_data_suja(data):
        if RNG.random() < 0.5:
            return data.strftime("%d/%m/%Y")
        return data.strftime("%Y-%m-%d")
    df["data_cadastro"] = df["data_cadastro"].apply(formatar_data_suja)

    # 5) Outliers plausíveis de erro de digitação (idade improvável).
    idx_outlier = RNG.choice(n, size=8, replace=False)
    df.loc[idx_outlier, "idade"] = RNG.integers(120, 200, size=8)

    # 6) Linhas duplicadas (mesmo cliente exportado duas vezes).
    duplicatas = df.sample(n=25, random_state=42)
    df = pd.concat([df, duplicatas], ignore_index=True)

    # 7) Espaços em branco indevidos em colunas de texto (comuns em exports
    #    de planilhas/CRM/PDV).
    df["forma_pagamento"] = df["forma_pagamento"].apply(
        lambda v: f"  {v} " if pd.notna(v) and RNG.random() < 0.15 else v
    )

    # Embaralha a ordem final das linhas para não deixar pistas óbvias
    # (ex.: duplicatas não ficam mais uma do lado da outra).
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


if __name__ == "__main__":
    base_limpa = gerar_base_limpa(N)
    base_suja = sujar_base(base_limpa)
    caminho_saida = "../dados/clientes_restaurante_brutos.csv"
    base_suja.to_csv(caminho_saida, index=False)
    print(f"Base gerada com {len(base_suja)} linhas em {caminho_saida}")
