# MVP — Previsão de Inatividade de Clientes de um Restaurante — PSP8

Trabalho de MVP da disciplina **PSP8 (Introdução à Ciência de Dados)**,
Prof. Dr. André Serrano — Universidade de Brasília (UnB).

## O que este projeto faz

Simula a situação de um restaurante com programa de fidelidade que precisa
prever, a partir do cadastro e do histórico de consumo de cada cliente, se
ele vai **ficar inativo** (parar de fazer pedidos/visitas). É um problema de
**classificação binária** (aprendizado supervisionado), resolvido com três
algoritmos de machine learning clássico (Regressão Logística, Random Forest
e Gradient Boosting), comparados por métricas de avaliação e matriz de
confusão.

A base de dados usada é **sintética, mas realista**: foi gerada
propositalmente com os mesmos problemas de qualidade que aparecem em uma
exportação real de CRM/PDV de restaurante (valores ausentes, categorias
inconsistentes, tipos de dado errados, datas em formatos diferentes,
outliers e duplicidades), para que o notebook demonstre também o trabalho de
limpeza e preparação de dados — e não apenas o treinamento do modelo. Os
valores foram simulados para preservar a confidencialidade de dados reais de
cliente, mas a estrutura e a lógica do problema refletem uma situação real
do dia a dia de restaurantes.

## Estrutura do repositório

```
mvp-churn-psp8/
├── README.md                          # este arquivo
├── notebook/
│   └── mvp_churn_analise.ipynb        # ENTREGA PRINCIPAL: notebook completo (relatório + código)
├── dados/
│   └── clientes_restaurante_brutos.csv  # base de dados "suja", usada como entrada do notebook
└── scripts/
    ├── gerar_dataset.py                # script que gerou a base (documenta cada "sujeira" inserida)
    └── build_notebook.py               # script que monta o notebook célula a célula (uso interno)
```

## Como publicar este trabalho (passo a passo)

O enunciado exige que a entrega final seja um **notebook público no Google
Colab**, com o dataset carregado por URL — e o professor pediu, à parte,
para o repositório também ficar salvo no **GitHub**. Siga esta ordem:

### 1. Criar o repositório no GitHub

1. Acesse [github.com/new](https://github.com/new) e crie um repositório
   **público** (ex.: `mvp-churn-psp8`).
2. No seu computador (ou em qualquer ambiente com `git`), suba os arquivos
   desta pasta:

```bash
cd mvp-churn-psp8
git init
git add .
git commit -m "MVP PSP8 - previsão de inatividade de clientes de restaurante"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/mvp-churn-psp8.git
git push -u origin main
```

### 2. Pegar a URL "raw" do CSV

Depois do push, abra o arquivo `dados/clientes_restaurante_brutos.csv` no
GitHub e clique em **"Raw"** — copie a URL da barra de endereço. Ela terá
este formato:

```
https://raw.githubusercontent.com/SEU-USUARIO/mvp-churn-psp8/main/dados/clientes_restaurante_brutos.csv
```

### 3. Atualizar o notebook com essa URL

Abra `notebook/mvp_churn_analise.ipynb`, encontre a célula da **Seção 3 —
Carregamento dos dados** e troque o valor de `URL_DADOS` pela URL que você
copiou no passo anterior.

### 4. Abrir no Google Colab e publicar

1. Vá em [colab.research.google.com](https://colab.research.google.com),
   clique em **Arquivo > Abrir notebook > GitHub**, cole a URL do seu
   repositório e selecione `notebook/mvp_churn_analise.ipynb` — isso abre o
   notebook já conectado ao GitHub.
2. Rode **Ambiente de execução > Executar tudo** para confirmar que tudo
   executa do início ao fim sem erros (é um dos critérios de nota).
3. Clique em **Compartilhar** (canto superior direito) e mude o acesso para
   **"Qualquer pessoa com o link"** — a entrega final precisa ser um notebook
   **público**.
4. (Opcional, mas recomendado) No Colab, use **Arquivo > Salvar uma cópia no
   GitHub** para gravar essa versão já executada de volta no seu repositório,
   mantendo os dois lugares sincronizados.

### 5. O que entregar ao professor

O link do notebook público no Colab (e, se ele pedir, também o link do
repositório no GitHub).

## Como explicar a origem dos dados ao professor

O próprio enunciado do MVP diz: *"Caso você prefira usar um dataset que
reflita um problema real da sua empresa (cuidado apenas com a
confidencialidade dos dados), será muito bem-vindo."* A forma honesta e
correta de apresentar este trabalho é exatamente essa: o **problema** (prever
inatividade de clientes de um restaurante com programa de fidelidade) é
inspirado em uma situação real que você acompanha no seu trabalho com
clientes de restaurante, mas os **valores da base** foram simulados/gerados
por script (não são um export literal de um cliente real), justamente para
não expor dados confidenciais de terceiros. Isso é uma prática comum e
aceita em ciência de dados — não é o mesmo que apresentar dados fabricados
como se fossem uma extração real.

## Regenerar a base de dados (opcional)

Se quiser gerar uma nova versão da base (por exemplo, para testar com outra
semente aleatória), rode:

```bash
cd scripts
python3 gerar_dataset.py
```

Isso sobrescreve `dados/clientes_restaurante_brutos.csv`. Lembre-se de
repetir o push para o GitHub depois — e, se os números mudarem bastante, vale
reler os parágrafos do notebook que citam valores específicos (seções 9, 10
e 11) para garantir que continuam batendo com a nova execução.

## Adaptando para uma base real

O pipeline de limpeza, modelagem e avaliação foi construído para ser
reaproveitável: para aplicar a uma base real de um restaurante/cliente
(mantendo o cuidado com a confidencialidade dos dados, como pede o
enunciado), basta substituir o arquivo
`dados/clientes_restaurante_brutos.csv` por uma base real com colunas
equivalentes (ou ajustar os nomes das colunas nas primeiras células da Seção
5 do notebook) e repetir os passos acima.
