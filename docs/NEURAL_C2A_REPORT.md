# Relatório do C2a — a fiação real carrega a informação

**Data:** 2 de outubro de 2026
**Estudo:** C2a, medida representacional em circuito do conectoma
**Pré-registro:** `docs/PREREGISTRATION_C2.md` rev. 5, selado antes de qualquer dado válido
**Execução:** `scripts/run_neural_c2a.py`
**Artefatos:** `outputs/neural_c2/c2a/`
**Veredito, pela regra pré-registrada:** **POSITIVO**

---

## 1. A pergunta, em uma linha

Com a mesma arquitetura, o mesmo estímulo e o mesmo número de passos: **quanto da
informação da pista de entrada sobrevive até a população de saída no circuito
real, e quanto sobra no mesmo circuito embaralhado?**

## 2. O substrato

Circuito escolhido por regra determinística, registrada antes de ver qualquer
resultado (`scripts/select_circuit_v783.py`):

| | |
|---|---|
| base | FlyWire **v783**, filtro `cell_type` presente e `status` vazio (136 543 de 138 639, 98,49 %) |
| construção | 2 saltos a partir de `sugar_grn`, o maior que satisfaz âncora de saída, filtro e orçamento |
| neurônios | **10 201** |
| arestas | **966 858** (606 566 excitatórias, 360 292 inibitórias) |
| sinapses | 1 135 948 |
| entrada | 20 neurônios `sugar_grn` |
| saída | 655 neurônios (572 `descending` + 83 `motor`), incluindo a âncora `mn9` |
| integridade | zero self-loops, zero pares duplicados, fracamente conexo (10 201/10 201) |
| 3 saltos | 95 136 neurônios, 2 799,6 MB — **fora** do orçamento registrado de 256 MB, portanto inelegível |

O kernel Rust foi validado **neste** circuito antes de qualquer uso: estado
**bit a bit idêntico** ao NumPy em 200 passos, todos os campos, 44 909 spikes
correspondentes, 0,0513 ms por passo, 64 164 368 bytes de memória nativa.

## 3. O resultado

16 conjuntos de sementes, 8 padrões de entrada, 4 repetições cada
(32 medições por semente), `W = 160` passos, acaso = 1/8 = 0,125.

| condição | média | dp | mín | máx |
|---|---|---|---|---|
| **A — real** | **0,7285** | 0,1184 | 0,531 | 0,938 |
| **B — grau preservado** | **0,4590** | 0,0746 | 0,250 | 0,500 |
| **C — alvos aleatórios** | **0,2500** | 0,0000 | 0,250 | 0,250 |

| contraste | média | IC 95 % (bootstrap, B = 10 000) |
|---|---|---|
| **A − B** (primário) | **+0,2695** | **[0,2129, 0,3223]** |
| A − C (secundário) | +0,4785 | [0,4219, 0,5332] |

**POSITIVO**: o limite inferior do IC de `A − B` é 0,2129, e a regra exige que
exceda 0. Nenhuma semente foi descartada; as 16 concluíram.

## 4. A parte mais interessante: a decomposição

Os três números contam uma história em três degraus, e ela é mais informativa
que o contraste principal:

```
0,250   C, grafo aleatório          a informação é destruída
  ↓     +0,209   restituí-lo           devolvem os ESTATÍSTICOS: grau de cada
                                       neurônio, distribuição de pesos, sinal
0,459   B, grau preservado           por neurônio. A estrutura estatística
                                       já carrega quase metade da informação.
  ↓     +0,270   a topologia real      só então a fiação específica acrescenta
                                       o resto.
0,729   A, conectoma real
```

Ou seja: **preservar grau, peso e sinal recupera a maior parte do que o grafo
aleatório destrói, e a topologia verdadeira acrescenta um ganho comparável por
cima.** É esse segundo degrau — o que a estatística de grafo **não** captura —
que é o objeto do estudo.

E o **C com desvio padrão exatamente zero** é um sinal limpo, não um número morto:
um grafo aleatório produz uma resposta praticamente **independente da pista**, a
ponto de a acurácia de classificação ficar travada em 2/8 nas 16 sementes. É o
que se esperaria de um controle que perdeu a organização.

## 5. Os controles foram auditados antes de qualquer decodificação

A auditoria roda **antes** de o primeiro padrão ser decodificado, e está em
`outputs/neural_c2/c2a/control_audit.json`:

| | graus idênticos | KL grau de saída | KL peso | trocas aceitas | marginais de sinal |
|---|---|---|---|---|---|
| A vs B | **sim** | 0,0000 | 0,0000 | 966 858 | preservadas |
| A vs C | **não** (é o controle fraco) | 0,0000 | 0,0000 | — | — |

A condição B preserva **exatamente** o grau de entrada e de saída de cada
neurônio, a distribuição de pesos e as marginais de excitação/inibição por alvo.
Isso não é sorte: a troca dupla é restrita a arestas de mesmo sinal e rejeita
qualquer troca que criasse par duplicado, porque o FlyWire guarda uma linha por
par ordenado.

## 6. O teste de sanidade obrigatório passou

O registro obriga um teste que não pode ser omitido **mesmo que o resultado
principal fique bonito**: um decoder treinado **nos controles** e aplicado a A.

```
decoder dos controles em A : 0,3574    (regra: inconclusivo se >= 0,50)
A                          : 0,7285
```

O controle acerta A em 0,3574 contra 0,7285 da própria A. Os controles **são**
representações diferentes. Se este número tivesse saído alto, os controles não
seriam controles e o resultado principal não valeria nada — está assim porque a
restrição de sinal e a preservação de grau funcionam, e não por sorte.

## 7. A calibração, e por que B e C não entraram nela

Um circuito ou é trivialmente legível ou é ilegível, e nos dois casos não há
experimento. Para sair do extremo, a regra pré-registrada escolhe **o menor `W`
que leve a condição A a 0,75 ± 0,05**, e o faz **usando só A**:

```
W:      60   80  100  120  140  160  180  200  250  320 ...
acc:  0,125 0,125 0,375 0,594 0,625 0,781 0,875 0,938 1,000 1,000
                                 ^^^^  escolhido (menor W na faixa)
spikes: 0,0  0,0  0,2  0,9  2,0  3,8  5,5  7,7 17,7 40,1
```

Uma curva sigmóide limpa. `W = 160` dá 0,781 e produz 3,8 spikes acumulados na
saída por padrão — pouco o bastante para não saturar, bastante para carregar a
pista. **B e C não entram em nenhum momento dessa escolha**, e o `W` foi gravado
antes de eles serem simulados.

## 8. O que este resultado **não** diz

Isto é a seção que um professor com doutorado vai ler primeiro.

1. **Não é neurobiologia.** Nenhuma afirmação sobre como a mosca real processa
   açúcar. É um modelo LIF com parâmetros de um paper, sobre um circuito real.
2. **Não é o cérebro inteiro.** São 10 201 dos 138 639 neurônios do v783, dois
   saltos a partir de uma âncora gustativa. Dizer "o conectoma é melhor que o
   embaralhado" a partir de um subgrafo seria extrapolação que estes dados não
   sustentam. O que se sustenta é sobre **este circuito**.
3. **Não é a tarefa social.** C2a mede representacional: não há agentes, não há
   recompensa, não há aprendizagem. A pergunta "a fiação importa para a
   comunicação" continua **sem resposta** e é a C2b.
4. **Não testa plasticidade.** Nenhum peso muda em nenhuma condição.
5. **A leitura é externa.** Um classificador centroide mais próximo, leave-one-out,
   fixo e idêntico nas três condições. Ele não foi escolhido comparando condições
   e não tem parâmetro livre.
6. **Uma leitura só.** Uma taxa de disparo por neurônio de saída, numa janela
   única. Nada aqui diz que a informação está onde medimos, e não em outro
   momento ou em outra estatística.

## 9. Uma execução anterior foi invalidada, e está registrada

A primeira execução de C2a produziu acurácias plausíveis e **não significava
nada**: o que era descrito como resposta de 655 dimensões era **um escalar por
padrão**. `population_counts` do núcleo Rust é um contador **por população**, não
por neurônio, e o runner passava as 655 saídas como uma população só.

Como o defeito só aparece quando a resposta é observada e comparada, ele passou
por lint, compilação e suíte de testes. A execução foi descartada e o diretório de
evidência removido antes de reexecutar. `tests/test_readout_contract.py` fixa o
contrato nos dois sentidos, e `simulate()` agora **levanta exceção** se o retorno
não tiver exatamente 655 contagens.

Está escrito na seção 7 do pré-registro, não escondido aqui.

## 10. Custos

| | |
|---|---|
| fase principal | **26,8 s** para 16 sementes × 3 condições |
| calibração | 17 valores de `W`, condição A apenas |
| tempo de relógio | registrado por rastreabilidade; **não é o desfecho** |

O circuito é barato porque o kernel Rust roda a 0,0513 ms por passo e because a
calibração encontrou `W = 160`, não `W = 4000`.

## 11. Procedência

- Pré-registro: `docs/PREREGISTRATION_C2.md` rev. 5, sha256 `16604abadf66ea24…`,
  selo em `docs/PREREGISTRATION_C2_SEAL.txt`
- Seleção do circuito: `scripts/select_circuit_v783.py` →
  `outputs/neural_c2/circuit_card.json`
- Portão do Rust: `scripts/gate_rust_circuit.py` → `outputs/neural_c2/gate_rust.json`
- Controle: `flybrain/experiments/rewire.py`, 20 testes em `tests/test_rewire.py`
- Contrato de leitura: `tests/test_readout_contract.py`, 4 testes
- Execução: `scripts/run_neural_c2a.py`
- Artefatos: `outputs/neural_c2/c2a/{analysis,calibration,control_audit,per_seed_set}.json`
- Figuras: `docs/figures/c2a/` (5 PNG)
- Ambiente: i7-7700K 4c/8t, 15,9 GB RAM, Windows 11, Python 3.11.7, NumPy 2.4.6. Sem GPU.
- Suíte: 232 testes, 1 pulado (pré-existente), 0 falhas.

---

*Todos os números deste relatório foram lidos de `outputs/neural_c2/c2a/`. O
pré-registro que o governa foi selado por hash antes da execução e verificado
depois.*
