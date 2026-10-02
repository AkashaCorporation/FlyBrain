# Relatório do C2b — a fiação real dentro da tarefa de comunicação

**Data:** 2 de outubro de 2026
**Estudo:** C2b, circuito do conectoma como portador da mensagem
**Pré-registro:** `docs/PREREGISTRATION_C2B.md` rev. 1, selado antes da execução
**Execução:** `scripts/run_neural_c2b.py`
**Artefatos:** `outputs/neural_c2/c2b/`
**Veredito, pela regra pré-registrada:** **POSITIVO**

---

## 1. O que C2b é

Em C2a a fiação foi medida sozinha: quanto da pista sobrevive até a saída. Em
C2b ela é posta **no lugar onde a mensagem é produzida**.

```
pista privada do remetente  →  resposta de 655 dims do circuito  →  símbolo
      (cue ∈ {0,1})             (16 ensaios por pista)              (∈ {0,1})
                                                          ↓
                                          receptor tabular idêntico ao de C1
                                                          ↓
                                          escolha certa quando o canal é íntegro
```

O receptor é o **mesmo agente** de C1, com o mesmo orçamento de 12 000 episódios.
A única coisa que muda entre condições é de onde vem a mensagem.

## 2. O resultado

16 conjuntos de sementes, `W = 160` herdado da calibração de C2a, 16 ensaios por pista.

| condição | fidelidade do canal | intacto | bloqueado | reamostrado | ganho sobre reamostragem |
|---|---|---|---|---|---|
| **A — real** | **1,000** | **0,9941** | 0,5000 | 0,5000 | **+0,4941** |
| **B — grau preservado** | 0,500 | 0,5000 | 0,5000 | 0,5000 | +0,0000 |
| **C — alvos aleatórios** | 0,500 | 0,5000 | 0,5000 | 0,5000 | +0,0000 |

| contraste | média | IC 95 % |
|---|---|---|
| **ganho, A − B** (primário) | **+0,4941** | **[0,4938, 0,4944]** |
| ganho, A − C | +0,4941 | [0,4938, 0,4944] |
| intacto, A − B | +0,4941 | [0,4938, 0,4944] |

**POSITIVO** pela regra: o limite inferior do IC do ganho sobre reamostragem é
0,4938, e a regra exige que exceda 0.

Duas leituras importam:

1. **A limpeza do resultado está no bloqueado e no reamostrado, não no intacto.**
   Em A o receptor acerta 0,9941 quando o canal está íntegro e cai para
   **exatamente 0,5000** quando o canal é bloqueado ou substituído por uma
   mensagem independente. O que ele aprendeu foi o canal, não a situação.
2. **B e C são exatamente iguais em 0,5000**, e isso é um resultado, não
   falta de poder. A tarefa binária exige só uma distinção grosseira entre duas
   pistas, e **qualquer** embaralhamento a destrói.

## 3. C2b e C2a medem coisas diferentes, e C2a é a mais sensível

| | A | B | C |
|---|---|---|---|
| **C2a** — 8 padrões, acurácia de decodificação | 0,7285 | 0,4590 | 0,2500 |
| **C2b** — 2 pistas, canal binário | 0,9941 | 0,5000 | 0,5000 |

Em C2a há um **gradiente** de três degraus, e B se separa de C. Em C2b o
resultado é um **degrau**: perfeito ou nada. Por quê: a tarefa de C2b pede ao
circuito apenas "qual das duas pistas?", uma distinção grosseira, que a topologia
real resolve por completo e que qualquer embaralhamento apaga. A tarefa de C2a
pede "qual de oito?", que exige muito mais da estrutura.

**Portanto C2a é a medida mais informativa, e C2b é a mais direta.** Elas não se
substituem: C2a mostra *quanto* a anatomia acrescenta, C2b mostra *que* a
mensagem depende dela. Se só uma puder ser mostrada, é decisão de quem apresenta
— e o relatório diz qual número vem de qual.

## 4. Teste de sanidade obrigatório passou

Molde construído a partir de **B e C**, aplicado a **A**:

```
fidelidade do canal com molde dos controles, em A : 0,5000
```

Exatamente acaso. As representações de A e dos controles não são
intercambiáveis, que é o que torna os controles controles.

## 5. O que este resultado **não** diz

1. **Não é neurobiologia.** Nada diz sobre como a mosca real comunica.
2. **Não é o cérebro inteiro.** 10 201 dos 138 639 neurônios do v783, dois
   saltos de uma âncora gustativa.
3. **O circuito não é plástico.** Nenhum peso muda. O substrato é fixo.
4. **O remetente não tem agência.** Ele recebe a pista e o circuito responde;
   por isso ele é congelado e não aprende. Só o receptor aprende. Isto está
   registrado no pré-registro e travado em `tests/test_external_message.py`.
5. **A leitura é enviesada a favor de A por construção**: os moldes saem da
   condição A. O contrapeso é a direção reversa da seção 4, que dá 0,5000.
6. **O estímulo é grosseiro**: cada pista ativa exatamente 10 dos 20 neurônios
   `sugar_grn`, em blocos disjuntos e com o mesmo número de ativos. É um
   estímulo pareado, não uma reprodução do estímulo gustativo.
7. **B e C não se separam aqui.** Com duas pistas, o experimento não tem poder
   para graduar. C2a separa, e é por isso que ele é a medida principal do
   conjunto.

## 6. Uma execução anterior foi invalidada, e está registrada

A primeira execução de C2b deu **POSITIVO com ganho de +0,011** — um número que
satisfezia a regra e não significava nada. O canal estava perfeito em A
(fidelidade 1,000) e o receptor, mesmo assim, "não aprendia".

A bisseção mostrou o motivo, e ela é instructive:

```
sem message_fn (caminho do C1)   intact=0,9900   curva de treino 0,478 → 0,988
message_fn = identidade (cue)    intact=0,4594   curva de treino 0,791 → 0,990
```

A curva de treino chegava a 0,990 e a avaliação dava 0,459. **O receptor tinha
aprendido a coisa certa e estava sendo avaliado num canal que nunca vira.**

Causa: `train()` aceitava uma mensagem externa, mas `evaluate()` não. O replay
reemitia a mensagem com a **política do remetente** — que, num estudo em que a
fiação produz a mensagem, está congelado e sem treino. O treino e a avaliação
usavam canais diferentes.

Correções:

1. `evaluate()` aceita `message_fn` e usa a mesma fonte de mensagem do treino,
   com validação de símbolo e sem mutar as políticas.
2. `tests/test_channel_coherence.py`, 5 testes: o receptor que aprende com
   `message_fn` tem de ser avaliado **alto** com `message_fn` e **perto do
   acaso** sem ele. O segundo teste é o que garante que os dois caminhos
   realmente diferem — se eles um dia convergissem, o teste pararia de proteger
   alguma coisa e falharia.
3. A execução invalidada foi descartada e o diretório removido antes de
   reexecutar.

Esta é a **sexta** instância da mesma classe neste projeto: *o fluxo executa,
os testes passam, e o número sai errado porque o que se mediu não é o que se
acha que se mediu*. O padrão é meu e está escrito no pré-registro.

## 7. Custos

| | |
|---|---|
| 16 sementes × 3 condições × 12 000 episódios | few minutos de relógio |
| 96 simulações de circuito por semente (2 pistas × 16 ensaios × 3 condições) | 160 passos cada |
| tempo de relógio | registrado por rastreabilidade; **não é o desfecho** |

## 8. Procedência

- Pré-registro: `docs/PREREGISTRATION_C2B.md` rev. 1, sha256 `175d2ebf03ddb241…`
- `W` e calibração: herdados de C2a, `outputs/neural_c2/c2a/calibration.json`
- `train(message_fn=...)`: `flybrain/learning/tabular.py`, 5 testes em
  `tests/test_external_message.py`
- `evaluate(message_fn=...)`: `flybrain/social/evaluation.py`, 5 testes em
  `tests/test_channel_coherence.py`
- Controle: `flybrain/experiments/rewire.py`, 20 testes
- Execução: `scripts/run_neural_c2b.py`
- Artefatos: `outputs/neural_c2/c2b/{analysis,calibration*,control_audit,preregistered_seeds,per_seed_set}.json`
- Ambiente: i7-7700K 4c/8t, 15,9 GB RAM, Windows 11, Python 3.11.7, NumPy 2.4.6. Sem GPU.

---

*Todos os números foram lidos de `outputs/neural_c2/c2b/`. O pré-registro foi
selado por hash antes da execução e verificado depois.*
