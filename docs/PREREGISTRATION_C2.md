# Pré-registro C2 — revisões 3 e 4

**Base:** `docs/PREREGISTRATION_C2.md` v1.0 rev. 2 (selado, sem resultado)
**Status desta revisão:** escrita **antes** de qualquer simulação de C2.
**Revisão 4:** acrescenta `R = 4` repetições por padrão, por causa escrita abaixo.
**Motivo da revisão 3:** a rev. 2 não fixava **como** o circuito entra na tarefa, e
o agente que a executou precisava escolher. Escolher depois de ver resultado é
exatamente o que o registro existe para impedir, então a escolha é feita agora,
por escrito.

---

## 1. O que muda em relação à rev. 2

A rev. 2 fixa condições, controles e critérios de seleção do circuito. Todas as
escolhas de substrato continuam válidas e foram aplicadas:

- circuito de **2 saltos a partir de `sugar_grn`**, 10 201 neurônios,
  966 858 arestas, ponte de 239,0 MB, contendo `mn9`, com 572 `descending` e
  83 `motor`. weakestmente conexo, zero self-loops, zero pares duplicados;
- base v783, filtro de validade `cell_type` não vazio **e** `status` vazio
  (136 543 de 138 639, 98,49 %);
- condição B por troca dupla com restrição de sinal, sem par duplicado.

O que esta revisão acrescenta é o **desenho do experimento neural**, que a rev. 2
deixava aberto. E acrescenta uma resposta a uma objeção séria.

## 2. A objeção que obriga o desenho

Existe uma circularidade óbvia na forma ingênua de C2: se o readout é treinado
na condição A e testado na condição A, A ganha por construção, e "A ganha" não é
descoberta nenhuma. Se o readout é treinado em A e testado em B, a assimetria
entre A e B também é parcialmente artificial.

A correção adotada aqui é declarativa e verificada:

> **Nenhuma condição é usada para escolher, ajustar ou validar o readout. O mesmo
> decoder fixo é aplicado às três condições, e o mesmo a todas as sementes. A
> única coisa que é calibrada é a dificuldade da condição A, e essa calibração
> acontece antes de B e C serem simuladas.**

A consequência é precisa e vale dizer: **esta experiência não é uma competição
entre condições.** A é a referência por definição. O que se mede é **quanto B e C
se afastam de A**, e portanto **se os controles são controles de verdade**. Um
controle que reproduz A não serve para nada, e é isso que a experiência testa.

## 3. Desenho de C2a — o substrato neural sozinho

C2a é a **medida de representacional**, sem a tarefa social por cima. Existe
porque é a base de que C2b precisa, e porque é rápida o bastante para rodar em
minutos e ser **auditada** antes de ser construída em cima.

**Pergunta.** Com a mesma arquitetura, o mesmo vocabulário de entrada e o mesmo
número de passos, quanto da informação do pistas de entrada sobrevive até a
população de saída no circuito **real**, e quanto sobra no circuito embaralhado
por grau e no aleatório?

### 3.1 Condições

| | grafo |
|---|---|
| **A — real** | o circuito selecionado do FlyWire v783 |
| **B — grau preservado** | A, reescrito por troca dupla com restrição de sinal, semente 20261002 |
| **C — aleatório** | A, com alvos sorteados sem restrições de grau; é o controle fraco |

O circuit B e C herdam **exatamente** os mesmos nós, os mesmos pesos por aresta e
o mesmo sinal. Muda só para onde cada aresta aponta. A condição C existe para
mostrar que um gradiente de "dá errado" não é artefato de um controle ruim.

### 3.2 Estímulo e leitura

- **Pistas:** `K = 8` padrões binários sobre os 20 neurônios de `sugar_grn`,
  gerados por `default_rng(semente_do_padrao).integers(0, 2, (8, 20))`, com
  **pelo menos 3 neurônios ativos por padrão** (re-amostra até satisfazer). A
  vocabulary é fixado antes de qualquer simulação.
- **Resposta:** taxa de disparo de cada neurônio da população de saída
  (os 655 `descending` + `motor` do circuito) na janela de `W` passos após o
  início do padrão. Vetor de 655 dimensões por padrão.
- **Decoder:** classificador de centroide mais próximo, leave-one-out, sem
  hyperparâmetros livres. Escolhido por ser o mais simples que ainda usa a
  geometria da representação, e por não ter nada a ajustar.
- **Repetições:** `R = 4` simulações independentes por padrão, com jitter no
  amplitude do evento injetado, sorteado por `default_rng(semente)`. São
  necessárias porque leave-one-out é **degenerado com uma amostra por classe**:
  retirar a única medição de um padrão deixaria o classificador sem qualquer
  referência para aquele padrão. Com `R = 4` são 32 medições por conjunto de
  sementes, 8 por padrão.
- **Medida:** acurácia leave-one-out de classificar o padrão correto, por
  padrão de sementes.

### 3.3 A calibração, e por que não vaza

Um circuito ou é trivialmente legível ou é ilegível, e nos dois casos não há
experimento. Para sair do extremo:

> **Regra de calibração, fixada agora:** escolher o menor `W` tal que a
> acurácia leave-one-out da condição **A** fique dentro de `0,75 ± 0,05` no
> primeiro conjunto de sementes. Busca binária sobre `W` em
> `{60, 80, 100, 120, 140, 160, 180, 200, 250, 320, 400, 640, 1000, 2000, 4000,
> 8000, 16000}`.

> **Revisão 5, e por quê.** A grade original da rev. 4 era
> `{100, 250, 500, 1000, 2000, 4000, 8000, 16000}`. Com a leitura por neurônio
> corrigida (ver seção 7), a condição **A** atingiu acurácia **1,000 já em
> W = 250**, ou seja a calibração falhou por ser **fácil demais**, e não por
> falta de informação. A transição para o regime decodificável acontece entre
> W = 100 (0,375) e W = 250 (1,000), e a grade saltava esse intervalo inteiro.
> A rev. 5 refina **apenas a ponta baixa da grade de `W`**. O alvo
> (`0,75 ± 0,05`), a regra do "menor `W` na faixa", a definição de deixar B e C
> de fora da calibração, N, sementes e a regra de decisão **não mudaram**. A
> falha foi observada **na condição A apenas**, antes de qualquer simulação de B
> ou de C existir.

`W` é então **fixado** e aplicado às três condições. B e C **não** entram na
calibração em momento algum, e o `W` escolhido é gravado junto do resultado
antes de B e C serem simulados.

Se a condição A não puder atingir essa faixa com nenhum `W` da grade, isso é
**resultado**, e C2a é reportado como "calibração impossível" sem as condições B
e C, porque sem um regime intermediário elas não significam nada.

### 3.4 Amostra e decisão

- **N = 16 conjuntos de sementes** independentes, derivados de
  `SeedSequence(20261002).spawn(16)`. Cada conjunto gera as 8 pistas, e dá uma
  acurácia por condição. Nenhuma semente é descartada.
- **Primário:** diferença pareada por semente, `A − B`, IC 95 % por bootstrap
  percentílico, `B = 10 000`, semente `20261002`.
- **Secundário:** `A − C`; e o **desvio** `|A − B|` contra `|A − C|`, que diz
  qual controle é mais fiel.
- **POSITIVO** se o limite inferior do IC de `A − B` > 0.
- **NEGATIVO** se o intervalo de `A − B` contém 0 **e** o de `A − C` também.
- **INCONCLUSIVO** nos demais casos, ou se N < 16.
- Nenhuma condição, semente, corte ou recorte é decidido depois de ver os dados.

## 4. O que C2a **não** faz, e por que ele é honesto

- **Não é neurobiologia.** Nada aqui diz como a mosca real codifica pistas.
- **Não é aexperiência social.** C2a não tem agentes, não tem recompensa, não
  tem aprendizagem. Mede representacional, e vai ser lido assim.
- **Não usa todo o cérebro.** É um circuito de 2 saltos, 10 201 dos 138 639
  neurônios do v783, e o resultado é sobre ele.
- **Não testa plasticidade.** Nenhum peso muda em nenhuma condição.
- **A nao equivale ao connectoma completo.** Dizer que o connectoma e melhor
  que o embaralhado, a partir de um subgrafo, seria uma extrapolacao que estes
  dados nao sustentam. O que se pode dizer e sobre este circuito, e o relatorio
  vai dizer exatamente isso.
  a partir de um subgrafo seria uma extrapolação que estes dados não sustentam.
  O que se pode dizer é sobre este circuito, e o texto do relatório vai dizer
  exatamente isso.

## 5. Por que a leitura vai ser a mesma nas três condições

O decoder é **fixo e idêntico**. Não é retreinado por condição, não é
otimizado por condição, e não é escolhido comparando condições. Se B e C
divergem de A, a divergência é do grafo, não da leitura.

E há um teste de sanidade que **precisa** ser reportado, mesmo que o resultado
principal tenha ficado bonito: um decoder **treinado nos controles** e aplicado
a A. Se esse decoder também acertar A, então A e os controles são a mesma
representação e C2a é inconclusivo por construção. Esse teste é registrado aqui
como obrigatório, e não pode ser omitido porque o resultado principal ficou
bonito.

## 6. Procedência

- Seleção do circuito: `scripts/select_circuit_v783.py` → `outputs/neural_c2/circuit_card.json`
- Portão do Rust: `scripts/gate_rust_circuit.py` → `outputs/neural_c2/gate_rust.json`
- Controle: `flybrain/experiments/rewire.py`, 20 testes em `tests/test_rewire.py`
- Execução: `scripts/run_neural_c2a.py`, a escrever.
- Artefatos: `outputs/neural_c2/c2a/`, diretório novo.
- Memória nativa medida do circuito: 64 164 368 bytes; 0,0513 ms por passo.

## 7. Um erro de leitura que invalidou uma execucao inteira

A primeira execucao de C2a produziu acuracias plausiveis e **nao significava
nada**. O que foi medido como "resposta de 655 dimensoes da populacao de saida"
era **um escalar por padrao**.

Causa: `population_counts` do nucleo Rust e construido como
`vec![0; populations.len()]` - **um contador por populacao, nao por neuronio**. O
runner passava as 655 saidas como **uma** populacao e recebia um numero. O teste
de referencia que ja existia no repositorio fixava isso
(`populations=[[0], [1, 2]]` devolve `[1, 0]`, quer dizer "a populacao disparou
zero vezes entre os dois") e nao foi lido.

Como o defeito so aparece quando a resposta e observada e comparada, e nao quando
o codigo roda, ele sobreviveu a lint, a compilacao e a suite de testes.

Correcoes, ambas aplicadas **antes** de qualquer resultado valido existir:

1. `simulate()` passa uma populacao unitaria por neuronio de saida e **levanta
   excecao** se o retorno nao tiver exatamente 655 contadores. O numero nao pode
   mais mentir sobre quantas dimensoes foram medidas.
2. `tests/test_readout_contract.py` fixa o contrato nos dois sentidos: N
   neuronios numa populacao dao 1 contagem; populacoes unitarias dao N; e a soma
   por neuronio tem que bater com a populacao unica.

A execucao invalidada foi descartada e o diretorio de evidencia removido antes de
reexecutar, para o numero errado nao sobreviver ao lado do certo.

**A licao, escrita porque e a quarta vez da mesma classe neste projeto:** o padrao
e meu, e e o de *inventar o contrato em vez de le-lo*. Aconteceu com a ordem de
entrada do nucleo Rust, com o protocolo de estimulo, com a coluna `status` das
anotacoes e agora com a granularidade do readout. A pergunta que falta antes de
apresentar qualquer numero e "o que eu medi de fato?", nao "o que da para
contar?".

---

*Se qualquer numero aqui mudar depois de ver dados, esta revisao permanece e a
mudanca vira rev. 6 datada, com o motivo - como ja aconteceu com a rev. 2 de
`PREREGISTRATION_C1.md` e com as rev. 2 a 5 de `PREREGISTRATION_C2.md`.*
