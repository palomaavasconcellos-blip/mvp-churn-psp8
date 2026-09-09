# -*- coding: utf-8 -*-
"""Monta o notebook mvp_churn_analise.ipynb célula a célula, usando nbformat.
Rodar com: python3 build_notebook.py
Gera: ../notebook/mvp_churn_analise.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ------------------------------------------------------------------
md(r"""# MVP — Previsão de Inatividade de Clientes de um Restaurante
### Disciplina PSP8 — Ciência de Dados | Prof. Dr. André Serrano
### Universidade de Brasília (UnB)

**Autor(a):** Paloma Vasconcellos

---

## Sobre este notebook

Este notebook funciona como **relatório completo do MVP**: além do código, cada
etapa é narrada em blocos de texto, explicando o raciocínio, as decisões
tomadas e a interpretação dos resultados — como pede o enunciado do trabalho.

**Como executar:** basta rodar as células de cima para baixo (`Ambiente de
execução > Executar tudo`, no Colab). O dataset é carregado diretamente de uma
URL pública no início do notebook, então não é necessário fazer upload manual
de arquivo nenhum.

> **Importante para quem for reproduzir este trabalho:** a variável
> `URL_DADOS`, logo na célula de carregamento dos dados, aponta para o arquivo
> `clientes_restaurante_brutos.csv` hospedado no repositório público do
> GitHub deste projeto. Se você mover o arquivo de lugar, basta atualizar essa
> variável.
""")

# ------------------------------------------------------------------
md(r"""## 1. Definição do problema

**Contexto.** Este trabalho parte de um problema real e atual que acompanho
no meu trabalho com clientes do setor de restaurantes. Um cliente meu — uma
franquia que opera no modelo rodízio (valor fixo, consumo à vontade) — está
em processo de reposicionamento comercial: identificou que um item específico
do cardápio é um gargalo de custo desproporcional dentro do rodízio e está
testando cobrar por ele separadamente, monitorando de perto como essas
mudanças afetam o consumo e o desperdício. Nesse cenário de transição, a
franquia também quer acompanhar mais de perto a fidelidade dos seus
clientes — hoje medida, na prática, pelo histórico de consumo (frequência de
visitas, ticket médio, reclamações, satisfação) — para identificar
antecipadamente clientes que estão reduzindo a frequência ou a caminho de
ficar inativos, já que uma mudança de precificação como essa tende a afetar
primeiro os clientes mais sensíveis a preço. Este MVP endereça exatamente
essa necessidade: prever, a partir do cadastro e do histórico de consumo, quais
clientes têm maior risco de ficar inativos, permitindo uma ação de retenção
antes de perdê-los de vez.

A base de dados usada aqui reproduz o formato e os problemas de qualidade
típicos de uma exportação real de sistema de CRM/PDV (ponto de venda) de um
cliente deste tipo — **os valores foram gerados sinteticamente, para
preservar a confidencialidade dos dados do cliente real em conformidade com
a LGPD (Lei Geral de Proteção de Dados)**, mas a estrutura, as variáveis e a
lógica do problema refletem fielmente a situação encontrada na prática.

**Descrição do problema.** A pergunta de negócio é:

> **É possível prever, a partir do cadastro e do histórico de consumo de um
> cliente, se ele vai ficar inativo (parar de fazer pedidos/visitas) no
> curto prazo?**

Definimos **cliente inativo** como aquele que não realiza nenhum novo
pedido, reserva ou visita dentro de uma janela recente de observação —
definição operacional comum em análises de churn no varejo/food service,
que não depende de o cliente ter formalmente "cancelado" nada (a maioria dos
restaurantes não tem contrato de assinatura; o cliente simplesmente para de
voltar). Trata-se de um **problema de classificação binária** (aprendizado
supervisionado): a variável-alvo `cliente_inativo` assume dois valores
possíveis (ficou inativo / continua ativo).

**Premissas e hipóteses.** Partimos da hipótese de que a inatividade está
relacionada a um conjunto de fatores observáveis no cadastro e no histórico
de consumo: tempo de relacionamento (clientes muito recentes ainda não
criaram o hábito de voltar e abandonam com mais facilidade), volume de
reclamações registradas, número de pedidos/reservas com no-show, satisfação
declarada (via pesquisa pós-atendimento ou avaliação no app), categoria de
fidelidade e forma de pagamento preferida. Assumimos também que esse
comportamento passado é suficiente para explicar boa parte da inatividade
futura — uma hipótese razoável, mas que não esgota todas as causas possíveis
(ex.: mudança de bairro do cliente, abertura de um concorrente, motivos que
não estão registrados no cadastro, ou — no cenário específico do cliente que
motivou este trabalho — a reação de clientes mais sensíveis a preço à
mudança de precificação de itens do cardápio, efeito que este dataset, por
ser anterior a essa mudança, ainda não captura diretamente).

**Restrições/condições impostas na seleção dos dados.** Por não ser possível
compartilhar a base real de um cliente sem quebrar a confidencialidade dos
dados dele, optou-se por **gerar uma base sintética, porém realista**, que
reproduz deliberadamente os problemas de qualidade típicos de uma exportação
real de CRM/PDV de restaurante (valores ausentes, inconsistência de
categorias, tipos de dado incorretos, datas em formatos diferentes, outliers
e duplicidades). O script usado para gerar essa base — com toda a lógica de
geração e de "sujeira" documentada — está em `scripts/gerar_dataset.py`, no
mesmo repositório deste notebook, garantindo total transparência sobre a
origem dos dados.

**Descrição do dataset.** O dataset simulado contém informações cadastrais e
de consumo de 2200 clientes de um restaurante (mais algumas duplicidades
propositais), descritos pelos seguintes atributos:

| Atributo | Descrição | Tipo esperado |
|---|---|---|
| `cliente_id` | Identificador do cliente | numérico (id) |
| `idade` | Idade do cliente, em anos | numérico |
| `genero` | Gênero do cliente | categórico |
| `tempo_relacionamento_meses` | Tempo desde o primeiro cadastro/pedido, em meses | numérico |
| `categoria_fidelidade` | Nível no programa de fidelidade (Bronze, Prata, Ouro) | categórico |
| `ticket_medio_mensal` | Valor médio gasto por mês | numérico (moeda) |
| `forma_pagamento` | Forma de pagamento preferida (Boleto, Cartão de Crédito, Pix) | categórico |
| `reclamacoes_registradas` | Nº de reclamações registradas (atendimento, prato, entrega) | numérico |
| `pedidos_no_show` | Nº de pedidos/reservas não concluídos (cliente não apareceu/cancelou em cima da hora) | numérico |
| `nota_satisfacao` | Nota de satisfação declarada (1 a 5), de pesquisa pós-atendimento | numérico/ordinal |
| `data_cadastro` | Data de entrada do cliente no programa de fidelidade | data |
| `cliente_inativo` (**alvo**) | Se o cliente ficou inativo na janela de observação (Sim/Não) | categórico binário |
""")

# ------------------------------------------------------------------
md(r"""## 2. Bibliotecas utilizadas

Antes de carregar os dados, importamos e explicamos **cada biblioteca**
utilizada neste notebook e o papel que ela cumpre:

- **pandas** → estrutura de dados (`DataFrame`) e todas as operações de
  leitura, limpeza e transformação de dados tabulares.
- **numpy** → operações numéricas vetorizadas (ex.: `NaN`, arredondamentos,
  máscaras booleanas usadas na limpeza).
- **matplotlib** e **seaborn** → construção dos gráficos exploratórios e das
  matrizes de confusão; seaborn é usado por gerar gráficos estatísticos mais
  legíveis com menos código.
- **scikit-learn (sklearn)** → biblioteca de machine learning "clássico"
  usada para: divisão treino/teste (`train_test_split`), pré-processamento
  (`OneHotEncoder`, `StandardScaler`, `ColumnTransformer`), construção de
  pipelines (`Pipeline`), os três algoritmos de classificação
  (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`),
  otimização de hiperparâmetros (`GridSearchCV`), validação cruzada
  (`cross_val_score`) e todas as métricas de avaliação (matriz de confusão,
  acurácia, precisão, revocação, F1 e AUC-ROC).
""")

code(r"""# Manipulação e análise de dados tabulares
import pandas as pd
import numpy as np

# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento e construção de pipelines de machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Algoritmos de classificação que serão comparados
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Métricas de avaliação de modelos de classificação
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

# Configurações visuais padrão para deixar os gráficos consistentes
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# Semente global de aleatoriedade: garante que qualquer pessoa que rodar este
# notebook obtenha exatamente os mesmos resultados (reprodutibilidade).
SEED = 42
""")

# ------------------------------------------------------------------
md(r"""## 3. Carregamento dos dados

Conforme exigido no enunciado, o dataset é carregado **diretamente por uma
URL pública**, dentro do próprio notebook — permitindo que qualquer pessoa
execute o notebook do início ao fim sem nenhuma configuração adicional.
""")

code(r"""# URL pública (raw) do arquivo CSV hospedado no repositório GitHub deste MVP.
# >>> SUBSTITUA esta URL pela URL "raw" do seu próprio repositório após o push <<<
URL_DADOS = "https://raw.githubusercontent.com/SEU-USUARIO/SEU-REPOSITORIO/main/dados/clientes_restaurante_brutos.csv"

# pd.read_csv aceita tanto um caminho local quanto uma URL: o pandas faz o
# download do conteúdo e monta o DataFrame diretamente a partir dele.
df_bruto = pd.read_csv(URL_DADOS)

# Sempre inspecionamos as primeiras linhas logo após o carregamento, para uma
# primeira checagem visual de que os dados vieram como o esperado.
df_bruto.head(10)
""")

# ------------------------------------------------------------------
md(r"""## 4. Diagnóstico de qualidade dos dados

Antes de qualquer modelagem, é preciso **entender o estado real da base**.
Vamos inspecionar tipos de dado, valores ausentes e valores únicos de cada
coluna categórica para mapear, de forma concreta, todos os problemas de
qualidade presentes.
""")

code(r"""# .info() mostra o tipo de dado (dtype) inferido pelo pandas para cada coluna
# e a quantidade de valores não nulos — primeiro sinal de colunas com
# problema de tipagem (ex.: ticket_medio_mensal como texto) ou com dados
# faltantes.
df_bruto.info()
""")

code(r"""# Quantidade e percentual de valores ausentes por coluna, ordenado do maior
# para o menor problema — ajuda a priorizar o que tratar primeiro.
ausentes = df_bruto.isna().sum().sort_values(ascending=False)
percentual_ausentes = (ausentes / len(df_bruto) * 100).round(2)
pd.DataFrame({"qtd_ausentes": ausentes, "percentual (%)": percentual_ausentes})
""")

code(r"""# Para cada coluna categórica, listamos os valores únicos: é aqui que fica
# evidente a inconsistência de categorias (ex.: "Sim", "sim", "S", "Yes"
# representando a mesma informação).
colunas_categoricas_brutas = ["genero", "categoria_fidelidade", "forma_pagamento", "cliente_inativo"]
for coluna in colunas_categoricas_brutas:
    print(f"--- {coluna} ---")
    print(sorted(df_bruto[coluna].dropna().unique().tolist()))
    print()
""")

code(r"""# Quantidade de linhas totalmente duplicadas (mesmo cliente exportado mais
# de uma vez do CRM).
qtd_duplicadas = df_bruto.duplicated(subset=[c for c in df_bruto.columns if c != "cliente_id"]).sum()
print(f"Linhas duplicadas (ignorando o id): {qtd_duplicadas}")

# describe() nas colunas numéricas ajuda a identificar outliers plausíveis,
# como uma idade de 187 anos.
df_bruto[["idade", "tempo_relacionamento_meses", "reclamacoes_registradas",
          "pedidos_no_show", "nota_satisfacao"]].describe()
""")

md(r"""**Achados do diagnóstico** (resumo dos problemas encontrados, que serão
corrigidos nas próximas seções):

1. Valores ausentes em `idade`, `genero`, `ticket_medio_mensal`,
   `nota_satisfacao` e `forma_pagamento`.
2. Categorias inconsistentes em `genero` (ex.: "Feminino"/"feminino"/"F"),
   `categoria_fidelidade` (ex.: "Bronze"/"BRONZE") e `cliente_inativo` (ex.:
   "Sim"/"S"/"Yes"/"1").
3. `ticket_medio_mensal` misturando números puros com texto no formato
   `"R$ 79,90"` — coluna inteira lida pelo pandas como `object` (texto),
   quando deveria ser numérica.
4. `data_cadastro` com dois formatos de data diferentes na mesma coluna
   (`dd/mm/aaaa` e `aaaa-mm-dd`).
5. Outliers de digitação em `idade` (valores acima de 120 anos).
6. Linhas duplicadas (mesmo cliente aparecendo mais de uma vez).
7. Espaços em branco indevidos em `forma_pagamento` (ex.: `" Pix "`).

Cada um desses pontos é tratado, em sua própria célula, na seção seguinte.
""")

# ------------------------------------------------------------------
md(r"""## 5. Limpeza e pré-processamento dos dados

Cada etapa abaixo resolve **um** problema específico identificado no
diagnóstico, nesta ordem: duplicidades → padronização de texto → tipos de
dado → datas → outliers → valores ausentes.
""")

code(r"""# Trabalhamos sempre sobre uma cópia, preservando o df_bruto original para
# consulta e para o relatório (boa prática: nunca perder o dado de origem).
df = df_bruto.copy()

# 5.1 — Remoção de duplicidades: mantemos apenas a primeira ocorrência de
# cada cliente (ignorando o id, que muda a cada exportação simulada).
linhas_antes = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "cliente_id"], keep="first")
print(f"Linhas removidas por duplicidade: {linhas_antes - len(df)}")
""")

code(r"""# 5.2 — Padronização de colunas de texto: removemos espaços nas pontas e
# colocamos tudo em minúsculas antes de mapear para as categorias corretas
# — isso evita ter que prever manualmente toda combinação de maiúsculas.
def padronizar_texto(serie):
    return serie.astype(str).str.strip().str.lower().replace("nan", np.nan)

for coluna in ["genero", "categoria_fidelidade", "forma_pagamento", "cliente_inativo"]:
    df[coluna] = padronizar_texto(df[coluna])

# Mapas de de-para: cada variação conhecida aponta para um único valor canônico.
mapa_genero = {"feminino": "Feminino", "fem": "Feminino", "f": "Feminino",
               "masculino": "Masculino", "masc": "Masculino", "m": "Masculino"}
mapa_fidelidade = {"bronze": "Bronze", "prata": "Prata", "ouro": "Ouro"}
mapa_inativo = {"sim": "Sim", "s": "Sim", "yes": "Sim", "1": "Sim",
                "não": "Não", "nao": "Não", "n": "Não", "no": "Não", "0": "Não"}

df["genero"] = df["genero"].map(mapa_genero)
df["categoria_fidelidade"] = df["categoria_fidelidade"].map(mapa_fidelidade)
df["cliente_inativo"] = df["cliente_inativo"].map(mapa_inativo)
# forma_pagamento já fica pronta só com o strip + lower; mapeamos
# explicitamente os três valores esperados para exibição consistente.
mapa_pagamento = {"boleto": "Boleto", "pix": "Pix", "cartão de crédito": "Cartão de Crédito"}
df["forma_pagamento"] = df["forma_pagamento"].map(mapa_pagamento)

# Conferimos que cada coluna ficou só com os valores esperados.
for coluna in ["genero", "categoria_fidelidade", "forma_pagamento", "cliente_inativo"]:
    print(coluna, "->", df[coluna].unique())
""")

code(r"""# 5.3 — Correção do tipo da coluna ticket_medio_mensal: removemos o prefixo
# "R$" e trocamos a vírgula decimal (padrão brasileiro) por ponto (padrão do
# Python), então convertemos a coluna inteira para numérico.
df["ticket_medio_mensal"] = (
    df["ticket_medio_mensal"].astype(str)
    .str.replace("R$", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)
df["ticket_medio_mensal"] = pd.to_numeric(df["ticket_medio_mensal"], errors="coerce")
print(df["ticket_medio_mensal"].dtype)
df["ticket_medio_mensal"].describe()
""")

code(r"""# 5.4 — Padronização de datas: usamos dayfirst=True (padrão brasileiro) e
# deixamos o pandas inferir automaticamente entre os dois formatos presentes
# (dd/mm/aaaa e aaaa-mm-dd); o parser do pandas reconhece ambos.
df["data_cadastro"] = pd.to_datetime(df["data_cadastro"], dayfirst=True, errors="coerce")
print(df["data_cadastro"].dtype)
df["data_cadastro"].describe()
""")

code(r"""# 5.5 — Tratamento de outliers de idade: valores acima de 100 anos são,
# neste contexto de negócio (cliente de restaurante), fisicamente
# improváveis o suficiente para serem tratados como erro de digitação — não
# como um cliente real. Em vez de remover a linha inteira (perderíamos as
# outras informações do cliente), tratamos como valor ausente e deixamos a
# imputação da próxima etapa cuidar disso.
df.loc[df["idade"] > 100, "idade"] = np.nan
df["idade"].describe()
""")

code(r"""# 5.6 — Tratamento de valores ausentes (imputação), com estratégia
# diferente por tipo de coluna:
#   - Numéricas -> mediana (mais robusta a outliers do que a média).
#   - Categóricas -> moda (valor mais frequente).
# Essa é uma estratégia simples e defensável para um MVP; caberia, em um
# projeto mais avançado, testar imputação por KNN ou por modelo.
colunas_numericas = ["idade", "ticket_medio_mensal", "nota_satisfacao"]
for coluna in colunas_numericas:
    mediana = df[coluna].median()
    df[coluna] = df[coluna].fillna(mediana)

colunas_categoricas = ["genero", "forma_pagamento"]
for coluna in colunas_categoricas:
    moda = df[coluna].mode()[0]
    df[coluna] = df[coluna].fillna(moda)

# Linhas com data de cadastro que não pôde ser interpretada, e linhas sem o
# rótulo-alvo, são poucas e não têm imputação sensata (não daria pra "chutar"
# a data de entrada do cliente nem se ele ficou inativo) — removemos essas
# linhas.
df = df.dropna(subset=["data_cadastro", "cliente_inativo"])

print("Valores ausentes restantes por coluna:")
print(df.isna().sum())
print(f"\nFormato final da base limpa: {df.shape}")
""")

# ------------------------------------------------------------------
md(r"""## 6. Análise exploratória (após a limpeza)

Com os dados já limpos e com tipos corretos, exploramos a relação entre as
variáveis e o alvo (`cliente_inativo`) para confirmar visualmente as
hipóteses do problema e para embasar as decisões de modelagem.
""")

code(r"""# Proporção de clientes inativos na base: importante para saber se as
# classes estão balanceadas ou não (isso influencia a escolha das métricas
# de avaliação).
proporcao_inatividade = df["cliente_inativo"].value_counts(normalize=True) * 100
print(proporcao_inatividade.round(1))

fig, eixo = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="cliente_inativo", hue="cliente_inativo", palette="Blues", legend=False, ax=eixo)
eixo.set_title("Distribuição da variável-alvo (cliente_inativo)")
eixo.set_xlabel("")
eixo.set_ylabel("Quantidade de clientes")
plt.show()
""")

md(r"""A base apresenta um **desbalanceamento moderado** entre as classes
(bem mais clientes ativos do que inativos) — situação esperada em problemas
de churn/inatividade e que motiva, mais adiante, o uso de métricas como
precisão, revocação e F1 além da simples acurácia.
""")

code(r"""# Relação entre variáveis numéricas e a inatividade, usando boxplots: cada
# painel compara a distribuição da variável entre clientes ativos e inativos.
variaveis_numericas = ["tempo_relacionamento_meses", "reclamacoes_registradas",
                        "pedidos_no_show", "nota_satisfacao", "ticket_medio_mensal"]

fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
eixos = eixos.flatten()
for i, variavel in enumerate(variaveis_numericas):
    sns.boxplot(data=df, x="cliente_inativo", y=variavel, hue="cliente_inativo", palette="Blues", legend=False, ax=eixos[i])
    eixos[i].set_title(variavel)
eixos[-1].axis("off")  # último painel sobra vazio (5 variáveis em grade 2x3)
plt.tight_layout()
plt.show()
""")

code(r"""# Taxa de inatividade por categoria de fidelidade e por forma de pagamento —
# variáveis categóricas que, segundo a hipótese do problema, também
# influenciam a inatividade.
fig, eixos = plt.subplots(1, 2, figsize=(12, 4))

taxa_por_fidelidade = df.groupby("categoria_fidelidade")["cliente_inativo"].apply(lambda s: (s == "Sim").mean() * 100)
taxa_por_fidelidade.plot(kind="bar", color="#3b6fa0", ax=eixos[0])
eixos[0].set_title("Taxa de inatividade por categoria de fidelidade (%)")
eixos[0].set_ylabel("% inativos")

taxa_por_pagamento = df.groupby("forma_pagamento")["cliente_inativo"].apply(lambda s: (s == "Sim").mean() * 100)
taxa_por_pagamento.plot(kind="bar", color="#3b6fa0", ax=eixos[1])
eixos[1].set_title("Taxa de inatividade por forma de pagamento (%)")
eixos[1].set_ylabel("% inativos")

plt.tight_layout()
plt.show()
""")

md(r"""**Leitura dos gráficos.** Os clientes que ficam inativos tendem a ter
menos tempo de relacionamento, mais reclamações registradas, mais
pedidos/reservas com no-show e notas de satisfação mais baixas —
confirmando, na prática, as hipóteses levantadas na definição do problema. A
categoria Bronze e o pagamento por boleto também aparecem associados a taxas
de inatividade um pouco maiores, embora com menor intensidade do que os
fatores comportamentais.
""")

# ------------------------------------------------------------------
md(r"""## 7. Preparação para a modelagem

**Separação de atributos previsores (X) e atributo-alvo (y).** O alvo é
`cliente_inativo`; removemos também `cliente_id` (não carrega informação
preditiva, é só um identificador) e `data_cadastro` (mantemos a informação de
tempo já resumida em `tempo_relacionamento_meses`, evitando redundância).

**Separação treino/teste.** Usamos `train_test_split` com **estratificação**
pela variável-alvo (`stratify=y`), garantindo que a proporção de clientes
inativos seja a mesma no treino e no teste — importante justamente por causa
do desbalanceamento identificado na análise exploratória.

**Validação cruzada.** Faz sentido usar validação cruzada (k-fold) neste
problema: a base, mesmo após a limpeza, tem um tamanho moderado (pouco mais de
2 mil registros), e uma única divisão treino/teste poderia gerar uma
estimativa de desempenho sensível a qual sorte de linhas caiu em cada lado.
Usamos `StratifiedKFold` (que preserva a proporção de classes em cada
partição) tanto na otimização de hiperparâmetros (`GridSearchCV`) quanto na
checagem de overfitting mais adiante.

**Transformação de dados.** Atributos numéricos são padronizados
(`StandardScaler`) para que fiquem na mesma escala — importante sobretudo para
a Regressão Logística, sensível à escala das variáveis. Atributos categóricos
são convertidos por `OneHotEncoder`, criando uma coluna binária para cada
categoria (necessário porque os algoritmos usados não trabalham diretamente
com texto).

**Seleção de atributos.** Com apenas 8 atributos previsores, não há
necessidade de uma técnica formal de redução de dimensionalidade (o problema
está longe da "maldição da dimensionalidade" mencionada na aula). Ainda assim,
avaliamos a importância de cada atributo **depois** do treinamento do modelo
de Random Forest (seção 9), o que serve tanto para validar a seleção quanto
para interpretar o resultado.
""")

code(r"""# Separação entre atributos previsores (X) e o alvo (y).
colunas_removidas = ["cliente_id", "data_cadastro", "cliente_inativo"]
X = df.drop(columns=colunas_removidas)
y = (df["cliente_inativo"] == "Sim").astype(int)  # 1 = ficou inativo, 0 = continua ativo

colunas_numericas_modelo = ["idade", "tempo_relacionamento_meses", "ticket_medio_mensal",
                            "reclamacoes_registradas", "pedidos_no_show", "nota_satisfacao"]
colunas_categoricas_modelo = ["genero", "categoria_fidelidade", "forma_pagamento"]

# Divisão treino/teste estratificada (80% treino, 20% teste).
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Treino: {X_treino.shape[0]} clientes | Teste: {X_teste.shape[0]} clientes")
print(f"Proporção de inatividade — treino: {y_treino.mean():.1%} | teste: {y_teste.mean():.1%}")
""")

code(r"""# ColumnTransformer aplica uma transformação diferente para cada grupo de
# colunas: escala as numéricas e cria variáveis dummy (one-hot) para as
# categóricas. Ele é reutilizado dentro de cada pipeline de modelo abaixo,
# garantindo que a MESMA transformação (ajustada só no treino) seja aplicada
# ao teste, evitando vazamento de dados (data leakage).
pre_processador = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), colunas_numericas_modelo),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas_modelo),
    ]
)

# Validação cruzada estratificada com 5 partições (5-fold), reutilizada em
# toda otimização/avaliação de estabilidade dos modelos.
validacao_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
""")

# ------------------------------------------------------------------
md(r"""## 8. Modelagem: seleção e treinamento dos algoritmos

Escolhemos comparar **três algoritmos clássicos de classificação**,
representando abordagens diferentes — uma decisão intencional para poder
justificar qual funciona melhor **para este problema específico**, em vez de
assumir de antemão qual seria o melhor:

- **Regressão Logística** — modelo linear, simples e altamente
  interpretável (os coeficientes indicam a direção e a força de cada
  atributo); serve de referência (baseline) para os demais.
- **Random Forest** — conjunto (ensemble) de árvores de decisão; captura
  relações não lineares e interações entre atributos sem exigir muita
  preparação adicional dos dados.
- **Gradient Boosting** — outro ensemble de árvores, porém construído de
  forma sequencial (cada árvore nova corrige o erro das anteriores);
  costuma obter desempenho competitivo em bases tabulares de porte pequeno
  a médio, como esta.

Cada modelo é embutido em um `Pipeline` junto com o `pre_processador`
definido acima — isso garante que a transformação dos dados seja parte do
próprio modelo (boa prática: evita erros de vazamento de dados e deixa o
código mais organizado).
""")

code(r"""# Pipeline 1 — Regressão Logística (baseline interpretável).
pipeline_logistica = Pipeline([
    ("pre_processamento", pre_processador),
    ("modelo", LogisticRegression(max_iter=1000, random_state=SEED)),
])
pipeline_logistica.fit(X_treino, y_treino)
print("Regressão Logística treinada.")
""")

code(r"""# Pipeline 2 — Random Forest, com otimização de hiperparâmetros via
# GridSearchCV. Testamos combinações de número de árvores e profundidade
# máxima — os dois hiperparâmetros que mais afetam o equilíbrio entre
# underfitting (árvores rasas demais) e overfitting (árvores profundas
# demais, decorando o treino).
pipeline_rf = Pipeline([
    ("pre_processamento", pre_processador),
    ("modelo", RandomForestClassifier(random_state=SEED)),
])

grade_hiperparametros_rf = {
    "modelo__n_estimators": [100, 200, 400],
    "modelo__max_depth": [4, 8, None],
    "modelo__min_samples_leaf": [1, 5, 10],
}

busca_rf = GridSearchCV(
    pipeline_rf, grade_hiperparametros_rf,
    scoring="f1", cv=validacao_cruzada, n_jobs=-1
)
busca_rf.fit(X_treino, y_treino)

print("Melhores hiperparâmetros (Random Forest):", busca_rf.best_params_)
print(f"Melhor F1 médio em validação cruzada: {busca_rf.best_score_:.3f}")
pipeline_rf_otimizado = busca_rf.best_estimator_
""")

code(r"""# Pipeline 3 — Gradient Boosting, também com otimização de hiperparâmetros:
# número de estágios (n_estimators), taxa de aprendizado (learning_rate) e
# profundidade de cada árvore.
pipeline_gb = Pipeline([
    ("pre_processamento", pre_processador),
    ("modelo", GradientBoostingClassifier(random_state=SEED)),
])

grade_hiperparametros_gb = {
    "modelo__n_estimators": [100, 200],
    "modelo__learning_rate": [0.05, 0.1],
    "modelo__max_depth": [2, 3, 4],
}

busca_gb = GridSearchCV(
    pipeline_gb, grade_hiperparametros_gb,
    scoring="f1", cv=validacao_cruzada, n_jobs=-1
)
busca_gb.fit(X_treino, y_treino)

print("Melhores hiperparâmetros (Gradient Boosting):", busca_gb.best_params_)
print(f"Melhor F1 médio em validação cruzada: {busca_gb.best_score_:.3f}")
pipeline_gb_otimizado = busca_gb.best_estimator_
""")

md(r"""**Sobre a métrica usada na otimização (`scoring="f1"`).** Como a base é
desbalanceada (mais clientes ativos do que inativos), otimizar diretamente
pela acurácia poderia favorecer um modelo "preguiçoso" que quase sempre prevê
"continua ativo" e ainda assim acerta bastante — sem nenhuma utilidade
prática para o negócio. O F1-score equilibra precisão e revocação e é a
escolha mais adequada quando se quer, de fato, identificar corretamente os
clientes que vão ficar inativos.
""")

# ------------------------------------------------------------------
md(r"""## 9. Avaliação dos resultados

Avaliamos os três modelos **no conjunto de teste**, que não participou nem do
treinamento nem da otimização de hiperparâmetros — é a única forma honesta de
estimar como cada modelo se comportaria com clientes novos, nunca vistos.

**Métricas escolhidas e por que.**
- **Acurácia** — proporção total de acertos; reportada por ser a métrica mais
  intuitiva, mas **não deve ser lida sozinha** neste problema, pelo
  desbalanceamento já discutido.
- **Precisão** — dentre os clientes que o modelo previu que ficariam
  inativos, quantos de fato ficaram. Importante quando uma ação de
  reengajamento (cupom de desconto, contato do time de relacionamento) tem
  custo, e não se quer desperdiçá-la em clientes que continuariam ativos de
  qualquer forma.
- **Revocação (recall)** — dentre os clientes que realmente ficaram
  inativos, quantos o modelo conseguiu identificar. Importante porque, em
  geral, o custo de **não identificar** um cliente que vai ficar inativo (e
  perder a chance de agir a tempo) é maior do que o custo de uma ação de
  reengajamento "desperdiçada".
- **F1-score** — média harmônica entre precisão e revocação; resume as duas
  em um único número, útil para comparar modelos.
- **AUC-ROC** — mede a capacidade do modelo de separar as duas classes em
  todos os limiares possíveis de decisão, não apenas no limiar padrão de 50%;
  é uma métrica mais robusta ao desbalanceamento do que a acurácia.
""")

code(r"""def avaliar_modelo(nome, pipeline, X_teste, y_teste):
    '''Calcula e retorna, em um dicionário, todas as métricas de avaliação
    de um modelo já treinado, sobre o conjunto de teste.'''
    y_previsto = pipeline.predict(X_teste)
    y_probabilidade = pipeline.predict_proba(X_teste)[:, 1]  # prob. da classe "ficou inativo"

    return {
        "modelo": nome,
        "acuracia": accuracy_score(y_teste, y_previsto),
        "precisao": precision_score(y_teste, y_previsto),
        "revocacao": recall_score(y_teste, y_previsto),
        "f1": f1_score(y_teste, y_previsto),
        "auc_roc": roc_auc_score(y_teste, y_probabilidade),
    }

resultados = [
    avaliar_modelo("Regressão Logística", pipeline_logistica, X_teste, y_teste),
    avaliar_modelo("Random Forest", pipeline_rf_otimizado, X_teste, y_teste),
    avaliar_modelo("Gradient Boosting", pipeline_gb_otimizado, X_teste, y_teste),
]

tabela_resultados = pd.DataFrame(resultados).set_index("modelo").round(3)
tabela_resultados
""")

code(r"""# Matrizes de confusão dos três modelos, lado a lado, para comparação visual.
modelos_treinados = {
    "Regressão Logística": pipeline_logistica,
    "Random Forest": pipeline_rf_otimizado,
    "Gradient Boosting": pipeline_gb_otimizado,
}

fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5))
for eixo, (nome, pipeline) in zip(eixos, modelos_treinados.items()):
    y_previsto = pipeline.predict(X_teste)
    matriz = confusion_matrix(y_teste, y_previsto)
    ConfusionMatrixDisplay(matriz, display_labels=["Ativo", "Inativo"]).plot(
        ax=eixo, cmap="Blues", colorbar=False
    )
    eixo.set_title(nome)
plt.tight_layout()
plt.show()
""")

md(r"""### Interpretação da matriz de confusão

A matriz de confusão organiza as previsões do modelo em quatro quadrantes,
comparando o que o modelo **previu** com o que **de fato aconteceu**:

- **Verdadeiro Negativo (canto superior-esquerdo)** — o cliente continuou
  ativo e o modelo previu corretamente que continuaria ativo. É o cenário
  mais comum, dado que a maioria dos clientes permanece ativa.
- **Falso Positivo (canto superior-direito)** — o cliente continuou ativo,
  mas o modelo previu que ele ficaria inativo. Na prática, isso significa
  gastar uma ação de reengajamento (cupom, contato) com um cliente que
  continuaria voltando de qualquer forma — um custo, porém menor do que o
  próximo erro.
- **Falso Negativo (canto inferior-esquerdo)** — o cliente ficou inativo,
  mas o modelo previu que ele continuaria ativo. Este é o **erro mais caro**
  para o negócio: é exatamente o cliente que o restaurante gostaria de
  identificar a tempo de agir (com uma oferta, uma ligação, um convite), e o
  modelo deixou passar.
- **Verdadeiro Positivo (canto inferior-direito)** — o cliente ficou
  inativo e o modelo previu corretamente. É o acerto que gera valor de
  negócio: permite uma ação de reengajamento **antes** de o cliente ser
  perdido de vez.

Na execução deste notebook (resultados da tabela da seção 9, com semente fixa
para reprodutibilidade), a **Regressão Logística** foi o modelo com o melhor
equilíbrio entre os quatro quadrantes: maior precisão (0,839) e o maior
F1-score (0,584) entre os três, com uma taxa de falsos positivos baixa — ou
seja, quando ela aponta que um cliente vai ficar inativo, acerta na grande
maioria das vezes. O Random Forest chegou a um AUC-ROC levemente mais alto
(0,794 contra 0,777 da Regressão Logística), mas isso não conta toda a
história: ele erra mais no quadrante de falso negativo (revocação de apenas
0,379, contra 0,448 da Regressão Logística), deixando passar uma proporção
maior de clientes que realmente ficariam inativos. A razão para essa
aparente contradição — AUC mais alto, porém pior no quadrante que mais
importa — fica clara na checagem de overfitting da próxima seção.
""")

code(r"""# ROC (Receiver Operating Characteristic): compara a taxa de verdadeiros
# positivos com a taxa de falsos positivos para cada limiar de decisão
# possível — quanto mais a curva se aproxima do canto superior-esquerdo,
# melhor o modelo separa as duas classes.
fig, eixo = plt.subplots(figsize=(6, 5))
for nome, pipeline in modelos_treinados.items():
    y_probabilidade = pipeline.predict_proba(X_teste)[:, 1]
    fpr, tpr, _ = roc_curve(y_teste, y_probabilidade)
    auc = roc_auc_score(y_teste, y_probabilidade)
    eixo.plot(fpr, tpr, label=f"{nome} (AUC={auc:.3f})")

eixo.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Modelo aleatório")
eixo.set_xlabel("Taxa de Falsos Positivos")
eixo.set_ylabel("Taxa de Verdadeiros Positivos (Revocação)")
eixo.set_title("Curvas ROC — comparação dos modelos")
eixo.legend()
plt.show()
""")

# ------------------------------------------------------------------
md(r"""## 10. Checagem de overfitting e underfitting

Comparamos o desempenho de cada modelo no **treino** contra o desempenho na
**validação cruzada** (dados que o modelo não viu durante o próprio ajuste
dentro de cada partição). Uma diferença muito grande entre os dois indica
overfitting (o modelo decorou o treino, mas não generaliza); desempenho baixo
em ambos indicaria underfitting (o modelo é simples demais para captar o
padrão).
""")

code(r"""for nome, pipeline in modelos_treinados.items():
    score_treino = f1_score(y_treino, pipeline.predict(X_treino))
    scores_cv = cross_val_score(pipeline, X_treino, y_treino, cv=validacao_cruzada, scoring="f1")
    print(f"{nome}:")
    print(f"  F1 no treino completo........ {score_treino:.3f}")
    print(f"  F1 médio em validação cruzada. {scores_cv.mean():.3f} (+/- {scores_cv.std():.3f})")
    print()
""")

md(r"""**Leitura dos números obtidos nesta execução:** a Regressão Logística tem F1
de treino (0,519) muito próximo do F1 médio em validação cruzada (0,517) —
ou seja, generaliza bem, sem overfitting relevante: o desempenho que ela
mostra no teste é o desempenho real que se deve esperar dela em produção.
Já o Random Forest chega a memorizar o treino quase perfeitamente (F1 de
treino = 1,000) mas cai para 0,495 na validação cruzada, uma queda enorme que
caracteriza **overfitting** mesmo depois de otimizado por `GridSearchCV` —
sinal de que, para este dataset, um modelo de árvores tende a se ajustar
demais a particularidades do conjunto de treino, e seu AUC-ROC aparentemente
melhor no teste (seção 9) é, em parte, sorte da divisão treino/teste
específica desta execução, não uma vantagem real e estável. O Gradient
Boosting apresenta o mesmo padrão, em grau menor (0,730 no treino contra
0,551 na validação cruzada). Esse resultado é uma boa ilustração prática de
que **um modelo mais sofisticado não é automaticamente um modelo melhor**, e
de que **uma única métrica de teste pode enganar**: aqui, o modelo mais
simples (linear) é o único cujo desempenho é confiável de fato, o que
reforça a importância de comparar sempre desempenho de treino contra
desempenho de validação cruzada, e não apenas olhar a métrica de teste
isoladamente.
""")

code(r"""# Importância dos atributos segundo o Random Forest — usada aqui como uma
# forma de checar, a posteriori, se a seleção de atributos feita (usar todos
# os 8 disponíveis) foi razoável, e para interpretar quais fatores mais
# pesam na decisão do modelo.
nomes_colunas_transformadas = (
    colunas_numericas_modelo
    + list(pipeline_rf_otimizado.named_steps["pre_processamento"]
           .named_transformers_["categoricas"]
           .get_feature_names_out(colunas_categoricas_modelo))
)
importancias = pipeline_rf_otimizado.named_steps["modelo"].feature_importances_

importancia_ordenada = pd.Series(importancias, index=nomes_colunas_transformadas).sort_values(ascending=True)

fig, eixo = plt.subplots(figsize=(7, 6))
importancia_ordenada.plot(kind="barh", color="#3b6fa0", ax=eixo)
eixo.set_title("Importância dos atributos (Random Forest)")
plt.tight_layout()
plt.show()
""")

# ------------------------------------------------------------------
md(r"""## 11. Análise dos resultados e pontos de atenção

**Principais achados.** Os três modelos conseguiram separar razoavelmente bem
os clientes que ficam inativos dos que permanecem ativos (AUC-ROC entre 0,77
e 0,79 no teste), confirmando que os atributos disponíveis carregam sinal
preditivo real (não são ruído). A **Regressão Logística** entregou o melhor
F1-score no conjunto de teste (0,584, com precisão de 0,839) e foi também a
única a não apresentar overfitting relevante — seu desempenho em validação
cruzada (F1 = 0,517) praticamente repete o desempenho de treino (F1 = 0,519),
o que a torna a candidata mais sólida para uma primeira versão em produção.
Isso reforça a boa prática de sempre incluir um modelo simples como
baseline, em vez de assumir que um modelo mais complexo (Random Forest,
Gradient Boosting) será superior por padrão: aqui, ambos os ensembles de
árvores memorizaram fortemente o conjunto de treino (F1 de treino de 1,000 e
0,730, respectivamente) sem sustentar esse desempenho em validação cruzada
(0,495 e 0,551). Os fatores com maior peso na decisão dos modelos foram, de forma
consistente com a análise exploratória da seção 6 e com a importância de
atributos da seção 10, o tempo de relacionamento, o número de reclamações
registradas e os pedidos com no-show — variáveis diretamente ligadas à
experiência do cliente com o restaurante, e não apenas ao seu perfil
cadastral (idade, gênero).

**Pontos de atenção.**
- A base é **sintética**: os padrões aprendidos refletem a lógica usada para
  gerá-la, não o comportamento real de clientes de um restaurante
  específico. Em uma aplicação real, o próximo passo seria repetir
  exatamente este pipeline sobre uma base real do restaurante/cliente
  (anonimizada), o que já deixa o notebook pronto para fazer.
- O desbalanceamento entre as classes, mesmo moderado, exige atenção
  contínua: qualquer mudança no negócio que altere a proporção de clientes
  inativos (ex.: sazonalidade, abertura de nova unidade) deve vir
  acompanhada de um re-treinamento do modelo.
- A imputação de valores ausentes pela mediana/moda é uma escolha simples;
  em uma segunda iteração do MVP, valeria testar imputação mais sofisticada
  (ex.: `KNNImputer`) e comparar o impacto no desempenho final.
- O modelo escolhido para produção deve equilibrar desempenho (poucos falsos
  negativos) com interpretabilidade exigida pela área de negócio — nem
  sempre o modelo com a métrica ligeiramente melhor é o mais indicado se for
  uma "caixa-preta" difícil de explicar a quem vai usá-lo no dia a dia.

**Melhor solução encontrada.** Entre os três modelos avaliados, a
**Regressão Logística** é a solução recomendada para este problema: obteve o
melhor desempenho no conjunto de teste em praticamente todas as métricas,
generalizou bem (sem sinal de overfitting) e, como modelo linear, tem a
vantagem adicional de ser facilmente interpretável pela área de negócio —
seus coeficientes indicam diretamente o quanto cada atributo aumenta ou
reduz a chance de o cliente ficar inativo, o que facilita tanto a explicação
do resultado quanto o desenho de ações de reengajamento direcionadas (por
exemplo, priorizar contato com clientes recém-cadastrados que já
registraram reclamações).

## 12. Conclusão

Este MVP percorreu o ciclo completo de um projeto de ciência de dados
aplicado a um problema de negócio real e recorrente no setor de restaurantes
— a previsão de inatividade de clientes cadastrados no programa de
fidelidade — partindo de uma base deliberadamente "suja" (para simular
fielmente o cotidiano de quem trabalha com dados de produção de um
CRM/PDV), até a comparação criteriosa de três modelos de classificação, com
métricas e interpretação de resultados alinhadas ao impacto de negócio de
cada tipo de erro. O pipeline construído (limpeza → preparação → modelagem →
avaliação) é diretamente reaproveitável: bastaria substituir a URL de origem
dos dados por uma base real de um restaurante/cliente (respeitando a
confidencialidade dos dados) para aplicar a mesma análise a um cenário de
produção.
""")

# ------------------------------------------------------------------
md(r"""## 13. Checklist do MVP (resumo consolidado)

| Item do checklist | Onde foi respondido |
|---|---|
| Descrição do problema, premissas e restrições | Seção 1 |
| Descrição do dataset | Seção 1 |
| Separação treino/teste e validação cruzada | Seção 7 |
| Transformação de dados (padronização, one-hot) | Seção 7 |
| Seleção de atributos | Seção 7 e Seção 10 (importância dos atributos) |
| Escolha e justificativa dos algoritmos | Seção 8 |
| Ajuste inicial e otimização de hiperparâmetros | Seção 8 (GridSearchCV) |
| Underfitting / overfitting | Seção 10 |
| Métricas de avaliação e justificativa | Seção 9 |
| Resultados fazem sentido / comparação de modelos | Seção 9 e Seção 11 |
| Melhor solução encontrada, justificada | Seção 11 |
| Análise de resultados e conclusão | Seções 11 e 12 |

---
*Notebook produzido para a disciplina PSP8 (Introdução à Ciência de Dados),
Prof. Dr. André Serrano — Universidade de Brasília (UnB).*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("../notebook/mvp_churn_analise.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook criado com", len(cells), "células.")
