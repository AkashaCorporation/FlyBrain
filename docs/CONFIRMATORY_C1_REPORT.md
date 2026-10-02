# Relatório do ciclo confirmatório C1

**Data:** 2 de outubro de 2026
**Estudo:** C1, uso funcional de canal de comunicação
**Pré-registro:** `docs/PREREGISTRATION_C1.md` v1.0 rev. 2, selado antes de qualquer dado
**Execução:** `scripts/run_confirmatory_c1.py`
**Artefatos:** `outputs/confirmatory_c1/`
**Veredito, pela regra pré-registrada:** **POSITIVO**

---

## 1. Integridade do pré-registro

| | |
|---|---|
| sha256 selado | `bacbb3262389b5c3895f1f912706859bad79d88e30042aa6d5ecea64d9a731be` |
| sha256 do arquivo hoje | `bacbb3262389b5c3895f1f912706859bad79d88e30042aa6d5ecea64d9a731be` |
| **inalterado depois dos dados** | **sim** |

O script de execução **recusa rodar** se o hash do documento não bater com o selo.
A lista das 64 sementes foi gravada em disco antes do primeiro treino, em
`outputs/confirmatory_c1/preregistered_seeds.json`, derivada de
`SeedSequence(20261002).spawn(64)`.

**Histórico de revisões, ambas antes de qualquer dado:**

| rev | momento | motivo |
|---|---|---|
| 1 | 2026-10-02T05:13:23Z | primeira gravação |
| 2 | 2026-10-02T05:12:xx (mesma sessão) | correção de descrição: a avaliação usa `repeats=64` condições por tipo sobre produto cartesiano exato, não 512 episódios |

Nenhuma hipótese, limiar, N, semente ou regra de decisão mudou entre as revisões.

## 2. Resultado

64 de 64 sementes concluíram o treino. **Nenhuma semente foi descartada.**

| estatística | valor |
|---|---|
| N analisado | **64 / 64** |
| média da diferença pareada `d` | **0,4881** |
| mediana de `d` | 0,4883 |
| desvio padrão | 0,002628 |
| IQR | [0,4867, 0,4900] |
| mínimo / máximo | 0,4811 / 0,4923 |
| **IC 95 % da média** (bootstrap percentílico, B = 10 000) | **[0,4874, 0,4887]** |
| IC 95 % da mediana | [0,4876, 0,4890] |
| leave-one-out (min, max) | [0,4880, 0,4882] |

**Regra de decisão, aplicada literalmente:** POSITIVO se o limite inferior do IC 95 %
da média excede 0,40. O limite inferior foi **0,4874**. Logo **POSITIVO**.

Testes de suporte (nenhum dos dois decide sozinho):

| teste | p | método |
|---|---|---|
| sign-flip pareado | 9,999e-05 | aleatorização por troca de sinal |
| Wilcoxon signed-rank | 3,525e-12 | `scipy.stats.wilcoxon` |

> **Sobre o p = 9,999e-05:** esse é o **piso** do teste, não um número mais preciso.
> Com B = 10 000 e suavização de +1, o menor p reportável é 1/10 001. O valor
> significa que o efeito observado é ao menos tão extremo quanto 10 000 das 10 001
> permutações nulas. Ele **não** permite afirmar que o p real é 10⁻¹²; isso só
> mentira sobre a resolução do teste. O Wilcoxon, que é exato neste tamanho de
> amostra, dá 3,5e-12.

## 3. Condições de avaliação

Retorno esperado com canal intacto, bloqueado e reamostrado, após treino, nas 64
sementes:

| condição | média | mín | máx |
|---|---|---|---|
| intacto | 0,9881 | 0,9811 | 0,9924 |
| bloqueado | **0,5000** | 0,5000 | 0,5000 |
| reamostrado | **0,5000** | 0,5000 | 0,5000 |
| ganho sobre reamostragem | 0,4881 | 0,4813 | 0,4924 |
| `tv_uniform_pairs` | 0,4381 | 0,4365 | 0,4393 |

Bloqueado e reamostrado são **exatamente** 0,5000 nas 64 sementes, sem exceção.
Ou seja: o que o receptor aprendeu é o **canal**, não a situação. Com o canal
cortado ou substituído por uma mensagem independente, o desempenho cai ao acaso
exato. É esse número — e não o 0,988 — que sustenta a afirmação de uso funcional
do canal.

## 4. Hipóteses secundárias

| | alvo | observado | cumpriu |
|---|---|---|---|
| **S1** fração que atinge o critério operacional | ≥ 0,95 | **1,000** (64/64) | sim |
| **S2** razão pós/holdout (mediana) | — | **1,0003** | sim, não degradou |
| **S3** mediana de episódios até 0,75 | ≤ 7 200 | **3 000** (64/64 cruzaram) | sim |

Nenhuma hipótese secundária foi promovida a primária depois do fato.

## 5. Controles — condição de entrada

O gate roda **antes** de qualquer treino. Se falhasse, nenhum treino aconteceu.

| controle | esperado | observado |
|---|---|---|
| `ManualChannelControl` | intacto 1,00 · bloqueado 0,50 · reamostrado 0,50 | 1,0 · 0,5 · 0,5 |
| `NoInformationPolicy` | intacto 0,50 | 0,50 |
| `RandomPolicy` | intacto 0,50 | 0,50 |

`classification: environment_controls_passed`. O gate está no código, não é
convenção: `run_tabular_pilot.py` e `run_confirmatory_c1.py` chamam
`control_gate()` e abortam com `SystemExit` se não passar.

## 6. O que este resultado **não** mostra

Isto é o que um professor com doutorado vai perguntar primeiro, e a resposta
honesta é: **C1 não mostra nada sobre a mosca.**

1. **Substrato neural: `False`.** Nenhum conectoma, nenhuma sinapse, nenhum
   neurônio. 32 parâmetros tabulares por indivíduo.
2. **Retenção: `False`.** A tarefa termina numa escolha. Não há o que reter.
   Exposição, retirada do parceiro e teste solo são protocolo futuro.
3. **Plasticidade sináptica: inexistente.** A aprendizagem está numa tabela.
4. **Não é neurobiologia.** Nenhuma afirmação sobre comportamento de mosca real.
5. **"Comunicação" é no sentido da tarefa C1**, não_signal, não JEV, não
   linguagem. Ver `docs/SCIENTIFIC_BOUNDARIES.md`.

## 7. A objeção que este resultado suscita

O desvio padrão de `d` entre sementes é **0,0026**, e bloqueado/reamostrado são
0,5000 exatos nas 64. Isso é **suspeitamente limpo** para um problema de
aprendizagem. Não é indício de erro — os controles e o portão de causalidade
passam, e o ganho sobre reamostragem é causal por construção — mas significa
que **a tarefa é fácil demais para um padrão limpo de política**.

A leitura honesta: C1 mostrou que um agente tabular **consegue** aprender a
convenção. Não mostrou que aprender a convenção seja *interessante*, nem que o
processo se pareça com qualquer coisa que um animal faça. Um agente tabular
qualquer, com a mesma tabela e o mesmo orçamento, chega lá.

É por isso que **C1 é o andaime, não o achado.** A pergunta que o conectoma faz
sentido é a C2, e ela está pré-registrada e **não executada**
(`docs/PREREGISTRATION_C2.md`, sha256 `c1402e5b…`).

O que o resultado de C1 entrega, com honestidade:

- uma plataforma com portão de causalidade, controles e avaliação exata;
- um protocolo estatístico funcionando de ponta a ponta, com registro selado;
- a **pergunta** que C2 vai atacar, agora formulada de modo falsificável.

## 8. Estado do substrato neural (não usado por C1, preparado para C2)

- Kernel Rust compilado, clippy `-D warnings` aprovado, extensão PyO3 instalada.
- **Validado só em rede pequena**: 128 neurônios, 1024 arestas, 15 testes
  diferenciais.
- **Whole-brain recusado, não rebaixado**: o adaptador por tuplas estima
  3 798 647 168 bytes para o v630 contra um teto de 256 MB. Um circuito de
  milhares de neurônios ainda **não** foi validado e precisa passar pelo mesmo
  teste diferencial antes de C2 rodar.
- Controle por grau preservado **pronto e testado**:
  `flybrain/experiments/rewire.py` + `tests/test_rewire.py`, 14 testes. A troca
  dupla é restrita a arestas de mesmo sinal e rejeita par duplicado, porque o
  FlyWire guarda uma linha por par ordenado.

## 9. Suíte de testes

**222 testes, 221 passaram, 1 pulado, 0 falhas.** O pulado é pré-existente e
condicional (`tests/test_doc_numbers.py`, diretório que não contém mais um run
whole-brain comparável). Os 14 testes do rewire são novos e estão incluídos.

## 10. Custo

| | |
|---|---|
| tempo de relógio | **424,9 s** (7 min 5 s) para 64 sementes |
| por semente | ~6,6 s (3 avaliações + treino de 12 000 episódios) |

O tempo de relógio é registrado por rastreabilidade e **não é o desfecho** do
estudo. Nenhuma análise dele entrou na regra de decisão.

## 11. Procedência

- Pré-registro: `docs/PREREGISTRATION_C1.md` (selo em `docs/PREREGISTRATION_C1_SEAL.txt`)
- Execução: `scripts/run_confirmatory_c1.py`
- Sementes: `outputs/confirmatory_c1/preregistered_seeds.json`
- Por semente: `outputs/confirmatory_c1/seed_<id>.json` + `seed_<id>_weights.npz` (com SHA-256)
- Gate: `outputs/confirmatory_c1/control_gate.json`
- Análise: `outputs/confirmatory_c1/analysis.json`
- Condições por semente: `outputs/confirmatory_c1/per_seed.csv.json`
- Ambiente: i7-7700K 4c/8t, 15,9 GB RAM, Windows 11, Python 3.11.7, NumPy 2.4.6, SciPy 1.17.1. Sem GPU.

`outputs/` é ignorado pelo git, por decisão de projeto: os binários ficam fora do
repositório e a proveniência versionada vai em `docs/`.

---

*Este relatório cita apenas números lidos de `outputs/confirmatory_c1/`. O
pré-registro que o governa foi verificado por hash **depois** da execução e não
havia sido alterado.*
