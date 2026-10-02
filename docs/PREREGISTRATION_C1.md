# Pré-registro do estudo confirmatório C1

**Versão:** 1.0
**Escopo:** confirmação do uso funcional de canal na tarefa C1, com 64 sementes
independentes.
**Status:** escrito **antes** de executar qualquer semente deste estudo.
**Estudo anterior:** piloto exploratório de 3 sementes (`outputs/first_cycle/tabular_pilot/`,
29/09/2026). Este registro **não** substitui nem reconfirma aquele resultado; ele o
trata como origem dos números de projeto, não como evidência.

---

## 1. Por que este estudo existe

O piloto exploratório de 29/09/2026 usou 3 sementes e atingiu escores próximos do
teto (0,986 a 0,988), mas o próprio script o classificou como
`exploratory_three_seeds_not_confirmation`. Três sementes não permitem intervalo
de confiança, não permitem estatística pareada e não distinguem "o efeito é real
e robusto" de "três sorteios favoráveis".

Este estudo existe para substituir evidência exploratória por evidência
confirmatória, **antes** de qualquer escrita do relatório. Nada aqui foi escolhido
depois de ver o resultado de 50 sementes, porque este documento é anterior a elas.

## 2. Pergunta e hipótese

**Pergunta.** Em um tarefa de escolha oculta com canal de três símbolos, um
indivíduo receptor treinado usa a mensagem do parceiro de modo **funcional** —
isto é, a recepção altera causalmente a escolha, e não apenas acompanha a situação?

**Hipótese primária (H1).** Após treino, o retorno esperado do receptor com o canal
intacto excede o acaso (0,50) por pelo menos **Δ = 0,40**, medido por diferença
pareada entre pós-teste e pré-teste dentro de cada semente.

**Hipótese nula (H0).** A diferença pareada média é ≤ 0,30, ou o ganho sobre
reamostragem é ≤ 0,10.

**Critério de decisão, fixado agora:**

| resultado | condição |
|---|---|
| **POSITIVO** | limite inferior do IC 95 % da média pareada (pós − pré) **> 0,40** |
| **NEGATIVO** | limite superior do IC 95 % da média pareada **≤ 0,30** |
| **INCONCLUSIVO** | qualquer outro caso; publicado como inconclusivo, sem novo teste |

A faixa entre 0,30 e 0,40 é **inconclusiva por construção**. Isso é deliberado:
reclassificar essa faixa depois de ver os dados seria exatamente o que este registro
existe para impedir.

## 3. Hipóteses secundárias (pré-especificadas, não decisivas)

- **S1.** A classificação por critério operacional (`intact ≥ 0,75`,
  `gain_vs_resampled ≥ 0,10`, `tv_uniform_pairs ≥ 0,10`) é atingida em **≥ 95 %**
  das sementes.
- **S2.** O resultado persiste sob uma semente de avaliação independente, com
  pesos congelados e mundos novos. Reportado como razão
  `intact_reavaliacao / intact_pos`.
- **S3.** A curva de treino atinge o limiar de 0,75 de retorno esperado em
  **≤ 60 %** dos episódios de treino. Reportado como mediana e IQR, não como
  média.

Nenhuma hipótese secundária pode ser promovida a primária depois do fato.

## 4. Desenho

- **Tarefa:** C1, duas opções equiprováveis, um alvo binário IID, dois
  indivíduos com tabela de 22 logits e 10 baselines cada, papéis sorteados por
  episódio, canal de três símbolos (0, 1, 2 = ausência de emissão).
- **Recompensa:** ambiental e cooperativa. Não existe tabela
  mensagem→resposta em lugar nenhum do código.
- **Treino:** 12 000 episódios por semente, limite de 30 s por semente.
  Score-function/REINFORCE com baseline por observação local; taxa 0,06;
  EMA 0,05; sem bônus de mensagem, informação, influência ou entropia.
- **Avaliação:** antes com semente `seed+10000`, depois com `seed+20000`, e uma
  reavaliação de retenção imediata com `seed+30000`. Cada avaliação usa
  `repeats=64` condições por tipo de condição, sobre o produto cartesiano
  balanceado (alvo × papel × ordem), que é percorrido **exatamente**. Por isso o
  retorno esperado é uma **expectativa exata sobre a grade balanceada**, e não
  uma amostra ruidosa: a única fonte de variação é a estocasticidade da política.
  A reavaliação usa pesos congelados — nenhum aprendizado acontece entre a
  avaliação posterior e a de retenção.
- **Amostra:** **N = 64 sementes** pré-especificadas, derivadas de
  `np.random.SeedSequence(20261002).spawn(64)`. A lista é gravada em disco
  **antes** do treino e não muda.

### 4.1 Justificativa de N = 64

O risco aqui não é falta de poder — com efeito esperado de ~0,49 e desvio
por semente pequeno, meia dúzia de sementes já daria poder alto. O risco é
**sensibilidade a um outlier** e **confiança em caudas**. N = 64 permite:

- intervalo de confiança por **bootstrap percentílico** (não pressupõe
  normalidade, que não foi verificada e não deve ser assumida);
- estimativa de mediana e IQR reportados junto, que são robustos a outlier;
- leave-one-out para quantificar a influência de cada semente.

Custo medido: o job de 3 sementes levou 16,20 s. N = 64 é ~6 min. O custo não
limita o desenho; portanto **não há justificativa estatística para N menor**, e
não haverá justificativa de conveniência para reduzir N depois do fato.

## 5. Controles — condição de entrada, não resultado

Os controles são executados **antes** de qualquer treino. Se qualquer um falhar,
**nenhum treino acontece** e o estudo é reportado como "não executado por falha de
controle". Isso está no código (`run_tabular_pilot.py`, `control_gate`), não é
convenção.

| Controle | Esperado |
|---|---|
| `ManualChannelControl` — canal perfeito, sem treino | `intact` = 1,00 · `blocked` = 0,50 · `resampled` = 0,50 · `tv_0_1` = 1 |
| `NoInformationPolicy` — receptor cego ao canal | `intact` = 0,50 · `tv_0_1` = 0 |
| `RandomPolicy` — ação uniforme | `intact` = 0,50 · `tv_0_1` = 0 |

Se os controles passarem e o resultado do treino for ~0,50, isso é um **negativo
real** sobre a política tabular, não um defeito do experimento. Os controles são o
que torna essa leitura possível.

## 6. Análise

1. **Estimador primário:** por semente, `d_i = intact_pos,i − intact_pre,i`, onde
   `intact` é o retorno **esperado** exato sobre o produto cartesiano balanceado
   (alvo × papel × ordem), conforme `flybrain/social/evaluation.py`.2. **Inferência:** IC 95 % por bootstrap percentílico sobre `d_i`, B = 10 000
   resamples, RNG com semente fixa gravada. A semente do bootstrap é fixada
   **neste documento** e não muda.
3. **Teste desupporte:** Wilcoxon signed-rank pareado sobre `d_i` (normalidade não
   assumida). Reportado como suporte, não como decisão — a decisão é o CI.
4. **Robustez:** média, mediana, IQR, min, max, leave-one-out, e fração de
   sementes que satisfazem cada critério operacional.
5. **Nenhuma semente é descartada.** Semente que falha, estoura tempo ou viola
   limite é **reportada** com o motivo, e o N efetivamente analisado é declarado.
   Se N analisado < 64, o critério de decisão da seção 2 **não é aplicado** e o
   resultado é INCONCLUSIVO por número de amostras.
6. **Nenhum subgrupo, recorte ou reparametrização** é decidido após ver os dados.

## 7. Limites declarados

Este estudo **não** testa, e não será relatado como se testasse:

- **Retenção / aprendizagem social retida (S1).** `retention_tested` permanece
  `False`. C1 termina em uma escolha; não há o que reter. A exposição, retirada
  do parceiro e teste solo são protocolo futuro (`docs/SOCIAL_EXPERIMENTS.md`).
- **Substrato neural / conectoma.** `neural_substrate_used` permanece `False`.
  Este estudo é inteiramente tabular. A pergunta "a fiação real acrescenta algo"
  é outro estudo, não este.
- **Plasticidade sináptica.** Nenhum peso de conexão muda. A aprendizagem está em
  32 parâmetros tabulares por indivíduo.
- **Biologia.** Nenhuma afirmação sobre comportamento de mosca, sinal ou cognição
  real. C1 é um artefato computacional.
- **Comunicação em sentido humano.** Ver `docs/SCIENTIFIC_BOUNDARIES.md` e a
  seção 5 de `docs/PROPOSTA_PESQUISA.md`.

## 8. O que invalidaria este estudo

Registradas agora, para não serem invocadas depois:

- Controle falha → estudo não executado.
- Média com IC que atravessa a faixa inconclusiva → INCONCLUSIVO.
- Efeito presente mas desaparecendo sob semente de avaliação independente (S2) →
  frágil, reportado como tal.
- Resultado que dependa de reparametrização não pré-especificada → inválido.
- Qualquer relato de "comunicação" que não traga o ganho sobre **reamostragem** →
  inválido, por decisão da seção 2.

## 9. Procedência

- Código: `flybrain/social/`, `flybrain/learning/tabular.py`,
  `flybrain/model/network.py`.
- Execução: `scripts/run_confirmatory_c1.py` (novo, escrito para este registro).
- Artefatos: `outputs/confirmatory_c1/`, diretório novo — os artefatos de
  `outputs/first_cycle/` **não** são sobrescritos.
- Semente do bootstrap: `20261002`.
- Sementes de treino: `SeedSequence(20261002).spawn(64)`.
- Ambiente: i7-7700K 4c/8t, 15,9 GB RAM, Windows 11, Python 3.11.7, NumPy 2.4.6.
  Sem GPU. Tempo de relógio é registrado, mas **não** é o desfecho.

---

*Este documento é anterior aos dados. Se qualquer número aqui precisar mudar após
ver os resultados, o registro original permanece e a mudança vira uma nova versão
datada, com o motivo.*
