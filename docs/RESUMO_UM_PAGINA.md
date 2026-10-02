# Três estudos, uma pergunta — resumo para reunião

**Data:** 2 de outubro de 2026
**Repositório:** `github.com/AkashaCorporation/FlyBrain` (branch local `research/first-cycle-20260929`, ainda não enviado)
**Substrato:** conectoma real da mosca — FlyWire v783, 10 201 neurônios, 966 858 arestas
**Todos os três pré-registrados e selados por hash antes de executar**
**Todos os três: veredito POSITIVO pela regra pré-registrada**

---

## A pergunta em três passos

| | pergunta | resposta |
|---|---|---|
| **C1** | um agente tabular aprende a usar um canal de comunicação? | **sim**, e rápido |
| **C2a** | quanto da pista de entrada sobrevive até a saída **no circuito real**? | **mais** que em qualquer embaralhamento |
| **C2b** | a fiação real **carrega** a mensagem da tarefa social? | **sim**, e só ela carrega |

A pergunta que abre a IC é da IC: *o que a organização de um circuito neural
acrescenta à aprendizagem de comunicação?* Nenhum dos três a responde sozinho. Juntos
dão a resposta em três partes.

---

## C1 — o canal é aprendível

Agente tabular, 2 indivíduos, 3 símbolos, recompensa ambiental. **Nenhum substrato
neural.**

| | |
|---|---|
| N | **64 / 64** sementes, nenhuma descartada |
| mudança pareada pós − pré | **média 0,4881**, mediana 0,4883, dp 0,0026 |
| **IC 95 % da média** | **[0,4874, 0,4887]** |
| leave-one-out | [0,4880, 0,4882] |
| bloqueado / reamostrado | **exatamente 0,5000 nas 64** |
| controles | manual 1,00 · sem informação 0,50 · aleatório 0,50 |

O receptor acerta 0,9881 com o canal íntegro e **exatamente 0,5000** quando o canal
é bloqueado ou substituído. O que ele aprendeu foi o canal, não a situação.

> **A ressalva que um revisor cético vai usar:** dp de 0,0026 é limpo demais para um
> problema de aprendizagem. A tarefa é **fácil**, e qualquer agente tabular com a mesma
> tabela chega lá. C1 é **andaime, não achado** — e está escrito assim no relatório.

---

## C2a — quanto a anatomia acrescenta

Medida representacional pura: **sem agentes, sem recompensa, sem aprendizagem.**
A pista de entrada é dirigida ao circuito real e a acurácia de classificar 8 padrões é
medida na saída.

| condição | acurácia de decodificação |
|---|---|
| **A — conectoma real** | **0,7285** ± 0,1184 |
| **B — grau preservado** | 0,4590 ± 0,0746 |
| **C — alvos aleatórios** | 0,2500 ± 0,0000 |

**A − B = +0,2695**, IC 95 % **[0,2129, 0,3223]**.

A decomposição é o achado:

```
0,250   C, aleatório        informação destruída
  ↓     +0,209               devolvem os ESTATÍSTICOS: grau, pesos, sinal
0,459   B, grau preservado
  ↓     +0,270               a topologia real acrescenta o resto
0,729   A, conectoma real
```

Preservar grau, peso e sinal recupera **quase metade** do que o aleatório destrói.
A topologia verdadeira acrescenta um ganho comparável por cima — e é esse segundo
degrau, o que a estatística de grafo **não** captura, que é o objeto do estudo.

Sanidade obrigatória (decoder dos controles aplicado a A): **0,3574** contra 0,7285.
Os controles são mesmo representações diferentes.

---

## C2b — a fiação carrega a mensagem

O circuito é posto **no lugar onde a mensagem é produzida**. O receptor é o mesmo
agente de C1, com o mesmo orçamento.

```
pista privada → circuito real → símbolo → receptor tabular
```

| condição | fidelidade do canal | intacto | bloqueado | reamostrado | ganho |
|---|---|---|---|---|---|
| **A — real** | **1,0000** | **0,9941** | 0,5000 | 0,5000 | **+0,4941** |
| **B — grau preservado** | 0,5000 | 0,5000 | 0,5000 | 0,5000 | 0,0000 |
| **C — aleatório** | 0,5000 | 0,5000 | 0,5000 | 0,5000 | 0,0000 |

**A − B = +0,4941**, IC 95 % **[0,4938, 0,4944]**.

Sanidade reversa (moldes dos controles aplicados a A): **0,5000**.

---

## As duas medidas neurais não se substituem

| | A | B | C | |
|---|---|---|---|---|
| **C2a** — 8 padrões | 0,7285 | 0,4590 | 0,2500 | **gradiente**: separa B de C |
| **C2b** — 2 pistas | 0,9941 | 0,5000 | 0,5000 | **degrau**: perfeito ou nada |

**C2b** é mais direta: mostra que a mensagem *depende* da anatomia.
**C2a** é mais informativa: mostra *quanto* a anatomia acrescenta, porque a tarefa
de 8 padrões exige muito mais da estrutura. Se só uma couber, C2a é a mais forte.
Se couberem as duas, elas se completam: **necessária e insuficiente**.

---

## O rigor, e o que ele custou

- **Pré-registro selado por SHA-256 antes de cada execução.** O script **recusa
  rodar** se o documento não bater com o selo, e confere de novo depois. Cinco
  revisões de registro, todas antes de qualquer dado, cada uma com o motivo.
- **Controles auditados antes de qualquer decodificação:** graus idênticos,
  divergência de Jensen-Shannon 0,0000 em grau e peso, 966 858 trocas aceitas,
  marginais de sinal preservadas. O controle é restrito a arestas de mesmo sinal e
  rejeita par duplicado.
- **Testes de sanidade obrigatórios** em C2a e C2b, registrados como não removíveis
  mesmo que o resultado principal fique bonito. Nos dois, os controles **não**
  leem a condição A.
- **242 testes automatizados, 0 falhas.**

### Seis resultados foram descartados por serem medidos errado

Este é o ponto que mais fala a favor da seriedade do trabalho. Em seis ocasiões o
código rodou, os testes passaram, e o número saiu errado porque **o que se mediu não
era o que se achava que se media**:

1. Coluna `status` das anotações está vazia — um filtro casou com **zero** de 138 639
   neurônios e o pipeline aceitou mesmo assim.
2. Entrada do kernel Rust não ordenada — 0 spikes, parecia falha de kernel.
3. Protocolo de estímulo inventado em vez do da v0 — circuito **morto**, 26 spikes.
4. Arestas com sinal desconhecido ficavam paradas no embaralhamento.
5. Condição C construída com `permutation`, que **preserva** os graus — não era
   controle nenhum.
6. **`population_counts` é por população, não por neurônio** — uma "representação de
   655 dimensões" era **um escalar**, e a primeira C2a foi invalidada.
7. `evaluate()` reemitia a mensagem com a política do remetente enquanto o treino
   usava o circuito — o receptor aprendia certo e era avaliado noutro canal.

Cada um ganhou correção **e teste**. O padrão está escrito nos pré-registros, porque
é o meu e precisa ficar visível.

---

## O que não é

Isto precisa estar na primeira página de qualquer conversa, e não na última:

- **Não é neurobiologia.** Nenhuma afirmação sobre comportamento de mosca real.
- **Não é o cérebro inteiro.** 10 201 dos 138 639 neurônios do v783, dois saltos de
  uma âncora gustativa. Dizer "o conectoma é melhor" a partir de um subgrafo seria
  extrapolação que estes dados não sustentam.
- **O circuito não é plástico** e **não aprende** em nenhum dos três estudos.
- **O remetente não tem agência** em C2b: a mensagem vem do circuito.
- **A leitura é externa** e, em C2a/C2b, os moldes saem da condição A — o que
  favorece A por construção. O contrapeso é o teste de sanidade reverso.
- **A tarefa de C1 é fácil**, e o desvio padrão de 0,0026 denuncia isso.

---

## Custos e o que torna isto viável

| | |
|---|---|
| kernel Rust | **0,0513 ms/passo** em 10 201 neurônios / 966 858 arestas |
| memória nativa | 64 164 368 bytes (a estimativa conservative diz 4× isso) |
| validação do kernel | estado **bit a bit idêntico** ao NumPy em 200 passos |
| C1 | 424,9 s · C2a 26,8 s · C2b 154,4 s |
| máquina | i7-7700K, 15,9 GB RAM, sem GPU |

O Rust não é detalhe: sem o kernel event-driven, 20 arestas por passo em vez de
14,7 milhões, a simulação seria inviável.

---

## O próximo passo óbvio

Os três estudos usam um circuito **fixo** e leitura **tabular**. Os dois passos que
têm mais valor para uma IC, nesta ordem:

1. **Retenção (S1).** A tarefa de C1 termina numa escolha; **não há o que reter**.
   Exposição, retirada do parceiro e teste solo são o protocolo já especificado e
   nunca executado. É a parte que o professor de psicologia vai perguntar primeiro.
2. **Plasticidade.** Nenhum peso muda em nenhum estudo. Ligar a aprendizagem
   social aos pesos de um circuito com anotações de tipo celular é a pergunta que
   fecha o arco.

---

## Onde está tudo

| | |
|---|---|
| relatórios | `docs/CONFIRMATORY_C1_REPORT.md` · `docs/NEURAL_C2A_REPORT.md` · `docs/NEURAL_C2B_REPORT.md` |
| pré-registros | `docs/PREREGISTRATION_C1.md` · `PREREGISTRATION_C2.md` · `PREREGISTRATION_C2B.md` (+ selos) |
| figuras | `docs/figures/c2a/` (5) · `docs/figures/c2b/` (3) |
| artefatos | `outputs/` — fora do git por decisão de projeto; a proveniência versionada está em `docs/` |
| commits | `ad280e0` `5160467` `84b9265` `623fb5b` `43c8fac` — **ainda não enviados** |

`docs/PROPOSTA_PESQUISA.md` está **fora** do índice do git, por decisão explícita.