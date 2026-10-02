# Pré-registro C2b — a fiação dentro da tarefa social

**Base:** `docs/PREREGISTRATION_C2.md` rev. 5
**Status:** escrito **antes** de qualquer execução de C2b.
**Predecessor:** C2a mediu representacional e deu POSITIVO. C2b é a versão que
responde à pergunta que C2a **não** responde.

---

## 1. A lacuna que C2b fecha

C2a mostrou, em circuito real do FlyWire v783, que a topologia verdadeira carrega
informação de pista que um embaralhamento por grau perde: acurácia de decodificação
0,7285 (A) contra 0,4590 (B) e 0,2500 (C), com `A − B = +0,2695` e IC 95 %
[0,2129, 0,3223].

Isso é uma medida **representacional**. Não há agentes, não há recompensa, não há
aprendizagem. A pergunta da IC — *o que a organização do circuito acrescenta à
aprendizagem de comunicação?* — **continua sem resposta**.

C2b coloca o circuito **dentro** da tarefa, no lugar onde a mensagem é produzida.

## 2. O que muda, e o que não muda

** muda:** a emissão do remetente deixa de ser uma tabela e passa a ser a resposta
do circuito.

**Não muda:** o mundo, os papéis, a recompensa, o observador receptor, a política
do receptor, o orçamento de treino, a semente, e as três condições de grafo com o
mesmo circuito, o mesmo controle por grau preservado e o mesmo controle aleatório.

A comparação com C1 continua válida porque o receptor é **exatamente** o mesmo
agente tabular treinado com o mesmo orçamento. A única variável manipulada é de
onde vem a mensagem.

## 3. O emissor é o circuito, e isso tem consequências que precisam ser declaradas

Na fase de sinal do mundo, a mensagem entra em `{0, 1, 2}`. Em C2b ela é
determinada por:

```
pista privada do remetente  ->  resposta de 655 dims do circuito  ->  símbolo
   (cue ∈ {0,1})                 (uma por ensaio)                     (∈ {0,1})
```

Três consequências, todas deliberadas:

1. **O remetente não tem agência.** Ele recebe a pista e o circuito responde. Por
   isso o remetente **não aprende** em C2b: não há nada a aprender, e mantê-lo
   aprendendo seria fingir agência que não existe. `train()` recebe um
   `message_fn` e, quando ela está presente, congela o remetente.
2. **O circuito não é plástico.** Nenhum peso muda. O substrato é fixo.
3. **Só o receptor aprende.** Toda a aprendizagem de C2b está no receptor. Isso
   torna a comparação com C1 limpa, e torna o resultado **uma** afirmação: um
   receptor tabular consegue ler uma mensagem carregada pela fiação real da mosca,
   e o quanto depende da topologia.

Isto **não** é "a mosca aprende a se comunicar". É "a fiação carrega um sinal que
um receptor consegue aprender a usar, e o quanto disso depende da anatomia".

## 4. Condições

As mesmas três, com o mesmo circuito e o mesmo `W = 160` que C2a calibrou:

| | grafo | papel |
|---|---|---|
| **A — real** | FlyWire v783, 2 saltos de `sugar_grn` | a anatomia de verdade |
| **B — grau preservado** | A reescrito, troca dupla com restrição de sinal | controle primário |
| **C — alvos aleatórios** | A com alvos sorteados, graus destruídos | controle fraco |

## 5. O canal e como a mensagem é extraída

- Cada condição tem, para cada pista `cue ∈ {0,1}`, uma resposta de 655 dimensões
  por **R = 16 ensaios**. Um ensaio é uma simulação de `W = 160` passos com a
  pista dirigida sobre os 20 `sugar_grn` a 150 Hz — o protocolo da v0.
- A resposta média de cada pista vira **molde**. O símbolo emitido é o molde
  mais próximo por distância euclidiana, com a mesma leitura por neurônio que C2a
  usa e que `tests/test_readout_contract.py` trava.
- **Os moldes vêm da condição A e são aplicados sem alteração em B e C.** É uma
  escolha que favorece A por construção, e isso é declarado: é a mesma assimetria
  que C2a assume. O que C2a acrescenta é a **direção reversa obrigatória**, usada
  aqui como sanidade.
- A cada episódio é sorteado um dos R ensaios, de modo que o canal tem ruído de
  ensaio a ensaio, como um canal real. A fidelidade é o que varia entre condições.

## 6. Teste de sanidade obrigatório

Um molde construído a partir de **B e C**, aplicado a **A**. Se A continuar
facilmente legível, A e os controles são a mesma representação e C2b é
inconclusivo por construção. Registrado aqui como obrigatório e não removível
mesmo que o resultado principal fique bonito.

## 7. Medida, N e decisão

- **Primário:** diferença pareada por semente na taxa de acerto com canal
  **intacto**, condição A menos condição B, IC 95 % por bootstrap percentílico,
  `B = 10 000`, semente `20261002`.
- **Secundários:** A menos C; e, em cada condição, **ganho sobre reamostragem** —
  que é o número que sustenta a afirmação de que o canal foi usado, e não a
  situação. Condição com intacto alto e ganho baixo é um resultado que precisa ser
  declarado, não escondido.
- **N = 16 conjuntos de sementes**, derivados de
  `SeedSequence(20261003).spawn(16)`. Semente distinta da de C2a de propósito: a
  amostra de sementes de C2a já foi usada para escolher `W` na calibração, e
  reutilizá-la aqui seria ligar dois estudos com uma amostra já vista.
- **POSITIVO** se o limite inferior do IC de `A − B` do ganho sobre reamostragem
  exceder 0.
- **NEGATIVO** se o intervalo do ganho contiver 0 **e** o de `A − C` também.
- **INCONCLUSIVO** nos demais casos, ou se N < 16, ou se a sanidade falhar.

O ganho sobre reamostragem, e não o retorno intacto isolado, é o desfecho
primário. Isso é mais exigente que C1, e é deliberado.

## 8. Limites, declarados agora

- **Não é neurobiologia.** Nada diz sobre comportamento de mosca real.
- **Não é o cérebro inteiro.** 10 201 dos 138 639 neurônios do v783, dois saltos
  de uma âncora gustativa.
- **O circuito não aprende.** É substrato fixo.
- **O remetente não tem agência.** Só o receptor aprende.
- **A leitura é externa e paramétrica em um ponto só**: os moldes saem da
  condição A. A direção reversa da seção 6 é o contrapeso.
- **Um W só, herdado da calibração de C2a**, e não recalibrado aqui. Recalibrar
  em C2b abriria espaço para escolher a dificuldade olhando as três condições.
- **Se B e C também forem legíveis**, o resultado é "a topologia não importa para
  esta medida neste circuito", que é um resultado legítimo e será reportado como
  tal.

## 9. Procedência

- Controle: `flybrain/experiments/rewire.py`, 20 testes
- Contrato de leitura: `tests/test_readout_contract.py`, 4 testes
- Calibração e W: C2a, `outputs/neural_c2/c2a/calibration.json`
- Execução: `scripts/run_neural_c2b.py`, a escrever
- Artefatos: `outputs/neural_c2/c2b/`, diretório novo

---

*Se qualquer número aqui mudar depois de ver dados, esta revisão permanece e a
mudança vira rev. 2 datada, com o motivo.*