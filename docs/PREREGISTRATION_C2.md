# Pré-registro do estudo C2 — o substrato neural

**Versão:** 1.0
**Status:** **PREPARADO E SELADO. NÃO EXECUTADO.**
**Data do selo:** ver `docs/PREREGISTRATION_C2_SEAL.txt`.
**Escopo:** substitui a política tabular de C1 por um circuito real do conectoma,
e pergunta se a **topologia** faz diferença.

> Este documento existe para que o experimento neural seja decidido **antes** de
> ver qualquer resultado dele. Nenhum dado de C2 existe. Nenhum número deste
> documento veio de uma execução. O código do controle está escrito e testado
> (`flybrain/experiments/rewire.py`, `tests/test_rewire.py`, 14 testes passando);
> a seleção de circuito e a execução **não** foram feitas.

---

## 1. A pergunta

C1 mostrou, com 64 sementes pré-registradas, que um agente tabular aprende a usar
um canal de comunicação de forma funcional. C1 **não** usa conectoma: são 32
parâmetros tabulares por indivíduo, e `neural_substrate_used` é `False`.

C2 faz uma pergunta que C1 não faz:

> Com tudo mais idêntico, um circuito cujas conexões são as **reais** da mosca
> produz o mesmo resultado que um circuito com a **mesma distribuição de grau,
> sinal e peso**, mas com a fiação sorteada?

Se a resposta for "não", a anatomia carrega informação que nenhuma estatística de
grafo captura. Se for "sim", o que apareceu em C1 foi uma propriedade do problema
de aprendizagem, não da mosca.

## 2. Por que esta pergunta e não outra

Um levantamento da lista curada da comunidade (`cobanov/awesome-fly`, 110
entradas) encontrou **60 projetos** de simulação da mosca. deles, apenas três
afirmam rodar o grafo retido inteiro, e a maioria é subset de circuito. O
controle por grau preservado já é publicado por ao menos quatro projetos
independentes. Portanto:

- "fazer a mosca se mexer" **não** é novidade;
- "embaralhar preservando grau" **não** é novidade, é técnica padrão;
- o que **não** apareceu em nenhum dos 60 é o controle por grau dentro de um
  **paradigma de comunicação** com pergunta psicológica. Esse cruzamento é a
  contribuição possível, e é o que C2 testa.

Se C2 der negativo, isso é resultado, não fracasso: a proposta já tem uma seção
inteira sobre o que fazer quando nada emerge.

## 3. Condições

Tudo o mais é idêntico entre as condições: mesmo mundo, mesmos dois agentes, mesmo
orçamento de treino, mesma semente, mesmo readout.

| condição | grafo | papel |
|---|---|---|
| **A — real** | circuito selecionado do FlyWire v783 | a anatomia de verdade |
| **B — grau preservado** | A reescrito por troca dupla com_restrição de sinal | controle primário |
| **C — aleatório** | A com alvos sorteados, graus destruídos | controle fraco, quase decorativo |
| **D — tabular** | sem grafo; C1 como está | linha de base já medida |

A condição **B** é a que carrega o argumento. A **C** existe para mostrar que
gradientes de "dá errado" não são artifatos de um controle ruim.

## 4. Por que "preserva grau" não basta, e o que mais o controle preserva

Esta é a parte técnica que decide se o resultado significa alguma coisa. Um
embaralhamento ingênuo em grafo assinado e ponderado destrói, junto com a
topologia:

1. **os graus** de entrada e saída de cada neurônio;
2. **as marginais de sinal por neurônio** — quanto cada alvo recebe de excitação
   e de inibição;
3. **a distribuição de pesos** associada a um dado grau;
4. **o equilíbrio excitatório/inibitório** da rede inteira.

Qualquer um desses muda a taxa de disparo e é **indistinguível** de um efeito
topológico. A troca dupla (Maslov–Sneppen; mesma ideia de `networkx`) preserva
(1) por construção, mas não (2)-(4) sozinha.

`flybrain/experiments/rewire.py` implementa a troca dupla com **duas**
restrições adicionais, e ambas são testadas:

- **só troca entre arestas de mesmo sinal** → (2) e (4) preservados por
  construção, e isso é **medido** no relatório, não assumido;
- **rejeita troca que criaria par duplicado** → o FlyWire guarda uma linha por
  par ordenado, com a contagem de sinapses como peso, e a validação do v630
  reporta zero pares duplicados; um controle com arestas paralelas deixaria de
  ser o mesmo tipo de objeto.

`compare_marginals` compara as distribuições de grau e de peso entre A e B e
reporta a divergência de Jensen-Shannon, para que se possa **auditar** a
adequação do controle em vez de confiar nela. O relatório de C2 publicará essas
marginais mesmo se o resultado for POSitivo.

## 5. Seleção do circuito — critérios registrados, não escolhidos depois

O circuito é escolhido por um procedimento determinístico, definido aqui antes de
rodar, e o procedimento é gravado junto com o resultado.

**Base: v783, não v630.** As anotações de tipo celular cobrem 138 625 de 138 639
neurônios do v783 (99,99 %) contra 106 214 de 127 400 no v630 (83,37 %). Sem
anotação, não há critério de seleção, e a escolha fica vulnerável a
pós-racionalização.

Critérios, na ordem:

1. âncora de entrada: uma população sensorial declarada
   (`sugar_grn`, `bitter_grn`, ou equivalente do registro de populações);
2. âncora de saída: uma população motora ou de comportamento declarada;
3. BFS a partir da âncora de entrada, com **no máximo 3 saltos** — limite que
   garante que o circuito é delimitado e auditável, e não um subconjunto
   arbitrário de Conveniência;
4. dentro do alcance, manter apenas neurônios **anotados** (`status` rastreado,
   `super_class` preenchida);
5. exigir que o subgrafo seja conexo e contenha ambas as âncoras;
6. o grafo resultante é gravado com SHA-256, contagem de neurônios e arestas, e
   o grau de entrada/saída de cada nó.

Se nenhum circuito satisfaz (1)-(5), isso é **reportado como resultado**, e a
seleção de outra âncora vira uma nova versão deste registro. Não se escolhe a
âncora depois de ver qual dá resultado.

## 6. Limites conhecidos, declarados agora

- **O kernel Rust foi validado em rede pequena** (128 neurônios, 1024 arestas,
  15 testes diferenciais). Um circuito de milhares de neurônios **não** foi
  validado. Antes de C2, o circuito tem de passar pelo mesmo teste diferencial
  contra o NumPy, com tolerância declarada.
- **O Rust recusa whole-brain** e isso não é um bug: o adaptador por tuplas estima
  3 798 647 168 bytes para o v630 contra um teto de 256 MB. C2 é um estudo de
  **circuito**, e o registro diz isso para que ninguém leia o resultado como
  "cérebro inteiro".
- **O readout é externo.** A leitura da atividade neural para escolha é um passo
  explícito, não um readout aprendido dentro do circuito. Isso é uma limitação de
  delineamento, não um detalhe de implementação, e precisa aparecer no relatório
  como tal.
- **Não é neurobiologia.** Nenhuma afirmação sobre sinal, comportamento ou
  cognição de mosca real. Ver `docs/SCIENTIFIC_BOUNDARIES.md`.

## 7. Métricas e regra de decisão

Fica especificado agora, com o mesmo formato de C1:

- **Primário:** diferença pareada por semente na taxa de acerto com canal
  intacto, condição A menos condição B, IC 95 % por bootstrap percentílico
  (B = 10 000, semente fixada).
- **Secundário:** A menos C; A menos D; e as marginais do relatório (4).
- **POSITIVO** se o limite inferior do IC 95 % de (A − B) > 0.
- **NEGATIVO** se o intervalo inteiro contém 0.
- **INCONCLUSIVO** nos demais casos, ou se N analisado < N declarado.

N fixado em **32 sementes** por condição, declarado aqui. Se o custo por semente
mostrar que 32 é inviável no prazo, isso vira **nova versão deste registro**, com
o motivo — não um ajuste silencioso depois de olhar o efeito.

## 8. Procedência

- Controle: `flybrain/experiments/rewire.py`, testado em
  `tests/test_rewire.py` (14 testes).
- Seleção: procedimento da seção 5, a implementar como
  `scripts/select_circuit_v783.py`.
- Execução: `scripts/run_neural_c2.py`, **ainda não escrito**.
- Artefatos: `outputs/neural_c2/`, diretório novo.
- Base de anotações: `E:\HipoCampo\stack\third_party\flywire_annotations\supplemental_files\Supplemental_file1_neuron_annotations.tsv`.

## 9. O que este registro não autoriza

- Não autoriza rodar C2 e depois escolher a âncora, o número de saltos ou o
  critério de anotação que deem o resultado desejado.
- Não autoriza usar C para justificar uma conclusão, se B já não sustentou.
- Não autoriza relatar C2 como resultado sobre o cérebro da mosca, e sim sobre um
  circuito delimitado com critério registrado.

---

*Se qualquer número aqui mudar depois de ver dados, esta versão permanece e a
mudança vira uma revisão datada, com o motivo — como já aconteceu com a rev. 2 de
`PREREGISTRATION_C1.md`.*
