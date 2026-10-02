# Primeiro ciclo MelanoGraph — entrega verificada em 29/09/2026

O primeiro ciclo A–C foi implementado e validado, com piloto tabular D adicional.
A referência permanece disponível, o núcleo Rust foi compilado e comparado com
NumPy e o avaliador diferencia comunicação funcional de controles sem informação.
O piloto encontrou comunicação aprendida no jogo discreto. Não demonstrou
aprendizagem social retida, plasticidade do conectoma ou comportamento de moscas.

Branch local: `research/first-cycle-20260929`. HEAD/base:
`30ad237735698c2097dfeaf2f8411b75a33e8d49`. Alterações permanecem sem commit e sem
push. `docs/PROPOSTA_PESQUISA.md`, que já existia sem rastreamento, foi preservado.
O arquivo histórico `ESTADO_ATUAL.md` deve ser lido junto desta atualização.

## Evidência principal

Todos os caminhos abaixo são relativos à raiz MelanoGraph.

| Evidência | Resultado atual | Artefato |
|---|---|---|
| Suíte original antes de editar | 158 aprovados, 1 pulado; 63,16 s | `outputs/first_cycle/baseline_pytest.xml` e `.json/.log` |
| Suíte final | **207 aprovados, 1 pulado, 0 falhas/erros**; 55,80 s | `outputs/first_cycle/final_pytest.xml` e `.json/.log` |
| Testes Rust | **5 aprovados**, 0 falhas | `outputs/first_cycle/rust_final_checks.log` |
| Clippy | aprovado com `-D warnings` | `outputs/first_cycle/rust_clippy_retry.log` |
| Ruff | aprovado, regras F/E9 | `outputs/first_cycle/final_python_lint.log` |
| Replay da v0 arquivada | identidade de bytes em fixtures estática e com trocas no modo legado; v1 mantém estática e altera intencionalmente trocas | `outputs/first_cycle/archived_replay.json` |
| NumPy ↔ Rust | 15 testes diferenciais aprovados, incluindo 12 combinações × 300 passos com estado inspecionado a cada passo | `tests/test_rust_differential.py`, `outputs/first_cycle/rust_differential_retry.xml` |
| Preservação | 48 arquivos de referência/dados/WIP conferidos e inalterados | `outputs/first_cycle/preservation_audit.json` |

O teste pulado é `tests/test_doc_numbers.py:191`: o primeiro diretório histórico
não contém mais um run whole-brain comparável. A condição já existia no baseline;
não foi alterada para obter aprovação. A extensão Rust estava instalada durante
a suíte final, portanto seus testes diferenciais **não** foram pulados.

## Preservação e correção de proveniência

`outputs/first_cycle/initial_manifest.json` contém estado Git, versões, hardware,
hashes e tamanhos antes das alterações na implementação. A fonte original está
arquivada em `v0-30ad237.zip`, com hash em `reference_archive.json`.

O relatório antigo chamava o digest combinado das fontes de SHA do processado.
A conferência dos bytes e do algoritmo `combined_digest` resolve a distinção:

| Identidade v630 | Valor |
|---|---|
| Digest combinado das fontes/metadata | `67de1c152ff8d47b8460f1c980052abfe3ed40c6e98870ff8af073a9e12dcf4a` |
| SHA-256 real de `connectome.npz`, igual antes/depois | `06d88f3a00ae8700ff0ae99e370cd5c6d4f4c53782de4234ea2e6b15fad21139` |
| Digest dos spikes históricos, recalculado do Parquet salvo | `d28feb6b82d1108056718dd246480cf4061c2c83bc14241779213941541f3760` |

Recalcular o digest de um arquivo salvo não é repetir a simulação whole-brain.
O protocolo de 1.000 ms não foi reexecutado. Não foi alterado nenhum golden file,
dataset, parâmetro padrão ou alegação de concordância parcial com o artigo.

## Contratos e núcleo

O padrão continua `legacy_v0`, inclusive a troca histórica que mantém antigos
alvos com refratariedade zero. `contract_version="v1"` reconstrói os defaults em
cada troca. `InputChannel` usa identidade explícita experimento/indivíduo/canal,
mantém o RNG ao mudar taxas e aplica entrada sensorial em condutância. A rotina
legada de Poisson mantém seu hash histórico, dependente de janela/taxa/rótulo.

[INPUT_STREAM_CONTRACT.md](INPUT_STREAM_CONTRACT.md) especifica reset, ordenação,
taxas vetoriais e limites de identidade de fluxo. [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md)
especifica limiar estrito, congelamento de v/g, ordem de entrada, atraso,
silenciamento de saída e redução float64→float32. `dt=0,5/1 ms` continua rejeitado.

O Rust usa uma thread, integração densa, CSR de fontes ativas, redução mista,
opção de ordem original de arestas e `advance` em blocos. O binding libera o GIL.
Observações padrão são contagens pequenas; estado completo requer chamada explícita.
[RUST_KERNEL.md](RUST_KERNEL.md) contém API, build e limites da ponte inicial.

### Benchmark medido, sintético

128 neurônios, 1.024 arestas, dt=0,1 ms; aquecimento separado de 100 passos e cinco
blocos de 2.000 passos por regime. Entradas gravadas idênticas; comparação dos
estados finais de cada bloco e contagens aprovada. Os tempos incluem a chamada
Python/Rust, mas excluem construção do grafo, preparação de entradas e build.

| Regime | NumPy, mediana ms/passo | Rust, mediana ms/passo | Spikes/bloco | Arestas ativas/bloco |
|---|---:|---:|---:|---:|
| Fraco | 0,03788 | 0,000645 | 240 | 1.896 |
| Moderado | 0,04813 | 0,001274 | 6.606 | 52.344 |
| Forte | 0,04790 | 0,002372 | 11.391 | 90.112 |

Tempos por bloco, p95, fluxos, hashes das entradas, memória e configuração:
`outputs/first_cycle/synthetic_benchmark.json`. Pico amostrado da árvore de
processos: 99,43 MB. Buffers nativos estimados no caso: 94.640 bytes.
Estes números não sustentam uma promessa de ms/passo para cérebro inteiro.

O adaptador por tuplas estima **3.798.647.168 bytes** para carregar v630, acima
do teto configurado de 256 MB. O benchmark de referência foi portanto omitido,
com decisão registrada no audit; nenhum subset foi executado como whole-brain.

## Causalidade e aprendizagem

Controles executados em 512 episódios balanceados:

| Controle | Retorno esperado íntegro | Bloqueado | Reamostrado | TV média dos pares |
|---|---:|---:|---:|---:|
| Manual, apenas validação | 1,00 | 0,50 | 0,50 | 0,4444 |
| Sem informação | 0,50 | 0,50 | 0,50 | 0 |
| Aleatório | 0,50 | 0,50 | 0,50 | 0 |

O receptor recebe apenas observação local. O snapshot antecede a mensagem;
intervenções reexecutam recepção/ação e preservam o RNG pareado. Reamostragem usa
a marginal global, nunca grupos que preservem o alvo oculto. Há testes de fases,
ordem, roles, índices, seeds, reset, reward, mutabilidade, sinais ignorados,
sensibilidade sem informação e invariância às seis permutações dos símbolos.

O piloto rodou uma vez por semente, sem seleção de vencedor ou busca de
hiperparâmetros: 12.000 episódios, dois indivíduos com tabelas e RNGs próprios,
recompensa cooperativa ambiental, nenhum dicionário ou supervisão do controle
manual. Os três runs terminaram dentro do teto. O job inteiro levou 16,20 s e
56,40 MB de RSS de pico amostrado, incluindo controles e avaliações.

| Semente | Antes, íntegro | Depois, íntegro | Depois, reamostrado | Nova avaliação dos pesos salvos, íntegro |
|---|---:|---:|---:|---:|
| 11 | 0,49991 | 0,98810 | 0,50000 | 0,98634 |
| 23 | 0,49998 | 0,98567 | 0,50000 | 0,99098 |
| 37 | 0,49978 | 0,98681 | 0,50000 | 0,98789 |

Na nova avaliação, bloqueio e reamostragem ficaram em 0,50 nas três sementes.
A variação do íntegro decorre de ações do emissor amostradas; o retorno do receptor
é integrado sobre sua distribuição de ações. Retornos amostrados também constam
dos JSONs. Três duplas são piloto, não confirmação; episódios não são novas
réplicas de aprendizagem. O alvo aprendido é comunicação funcional em C1.

Artefatos: `outputs/first_cycle/social_controls.json`,
`tabular_pilot/seed_{11,23,37}.json`, pesos `.npz` com hashes, curvas e
`tabular_pilot/frozen_reevaluation.json`. A classificação final usa TV média de
todos os pares, evitando privilegiar nomes de símbolos. O primeiro piloto também
passava o critério mais restrito anterior de TV 0↔1; ambas as evidências foram mantidas.

PPO não foi executado: `torch`, `stable_baselines3` e `sb3_contrib` estão ausentes
na `.venv`; não foi instalada uma stack de RL. A opção tabular foi realizada.
[SOCIAL_EXPERIMENTS.md](SOCIAL_EXPERIMENTS.md) prepara o protocolo S1 de exposição,
retirada do parceiro e avaliação solo. `learning/interfaces.py` prepara a
interface local de actor-critic/elegibilidade e máscara plástica. A tarefa
individual de recompensa atrasada continua sendo gate obrigatório antes da
futura frente social mecanística. Não há alegação de retenção ou plasticidade interna.

## Ambiente, limites e ocorrências preservadas

Inspeção atual: i7-7700K, 4 núcleos/8 threads, 17,11 GB físicos decimais (~15,94 GiB),
GTX 1650 de 4 GB. Python 3.11.7, NumPy 2.4.6, Rust/Cargo 1.98.0 MSVC e Visual Studio
2022 Community já estavam instalados. Rust e Python não estavam no PATH usado pelo
primeiro build. Foi usado o caminho absoluto; não houve instalação administrativa.
Instalados apenas maturin 1.10.2 (wheel ~8,9 MB) e a extensão local na `.venv`.
As licenças dos manifests e arquivos de licença do maturin estão no audit.

Cada job tem JSON de comando, versões, commit, hashes de fonte, limites, modo
solicitado/efetivo, status e log completo. O supervisor amostra RSS da árvore
a cada 100 ms, usa limite de log de 10 MB e recusa reserva acima de 60% da RAM
disponível. O pico da suíte final foi 1,515 GB sob teto de 2,2 GB.

Ocorrências que permanecem nos artefatos, sem reclassificação como sucesso:

- Regressões novas inicialmente falharam porque a API versionada ainda não existia.
- Uma reserva de 3,5 GB para a suíte foi recusada pela guarda; a mesma suíte rodou
  com teto de 2,2 GB. O baseline anterior havia rodado com 3,5 GB quando havia mais RAM.
- Primeiro `cargo check` do binding falhou por ausência de Python no PATH.
- O primeiro build release produziu a wheel corretamente; seu supervisor falhou
  depois ao imprimir um emoji em CP1252. Corrigida saída UTF-8; logs/JSON preservados.
- Primeiros 15 testes diferenciais falharam na conversão de `Vec<u8>`/bytes do PyO3;
  corrigido o adaptador, sem mudar resultados esperados.
- Um teste causal falhou ao serializar classe local; o fingerprint passou a
  serializar identidade da classe e estado. Clippy apontou tipo excessivamente
  complexo; criado alias. Ruff apontou import sem uso; removido.
- O linker MSVC imprime mensagem informativa de criação de biblioteca, tratada
  pelo Rust como warning de saída do linker; os builds terminaram com código zero.

## Auditoria requisito por requisito

| Requisitos do goal | Evidência/decisão |
|---|---|
| A1 | Manifesto inicial, archive do HEAD real, inspeção de pacote/backend/kernel, baseline e suíte final. |
| A2 | Audit dos 48 arquivos, fontes e digest de spikes salvos; identidade de cache distinguida de digest de origem. |
| A3 | Casos A→B, união/interseção, clear, reset e alternância em `test_input_contracts.py`; legado preservado por versão e replay. |
| A4 | Taxas/samplers/reset separados, hash legado verificado, fluxos explícitos e testes de independência. |
| A5–A6 | Contrato numérico escrito; testes de atraso, congelamento, limiar, silenciamento e cancelamento misto. |
| B1–B4 | Dois crates, PyO3/maturin, uma thread, CSR, buffers, modos de ordem, advance e observações pequenas. |
| B5–B6 | Cinco testes Rust e 15 diferenciais sintéticos; estados/eventos por passo, inputs comuns e primeiro ponto divergente em falha. E0/E1/E2 separados. |
| B7 | Benchmark sintético real com aquecimento/medianas/p95/memória/spikes/arestas; whole-brain recusado pelo orçamento da ponte. |
| B8 | Ramo de ausência de toolchain não aplicável: compilador e linker foram usados com sucesso. |
| C1–C3 | Mundo binário equiprovável, fases, pista privada, dados locais, RNGs/pesos individuais e recompensa por resultado. |
| C4–C7 | Três controles, bateria de vazamento, snapshots/replay/intervenções, métricas causais e gate antes do piloto. |
| D1–D3 | Três pilotos tabulares completos, hiperparâmetros/curvas/pesos/seeds/limites, avaliação congelada e nova avaliação. PPO opcional não executado por dependências ausentes. |
| D4–D5 | Protocolo de retenção documentado e interfaces preparadas; execução S1/eligibilidade/plasticidade fica no roadmap. |
| Artefatos finais | Este relatório, contratos, scripts, logs/JSON/XML, wheels, hashes e fontes no workspace. |

## Próximos marcos priorizados — não executados neste ciclo

1. Trocar a ponte de tuplas por buffers com propriedade explícita; medir circuito
   real pequeno e protocolo v630 com orçamento próprio, antes de alegar desempenho whole-brain.
2. Implementar tarefa individual de recompensa atrasada e readout com elegibilidade;
   comprovar competência e identificar exatamente onde os pesos mudam.
3. Construir S1 com observações solo suficientes e controles pareados, sem feedback
   de ensino no teste. Só depois estudar plasticidade interna e redução de mecanismos.
4. Planejar circuitos v783 anotados com máscaras registradas. Manter os 385 de sinal
   desconhecido do v630; não inferir sensores por grau zero. Comparar membership,
   status Traced e superclasses MaleCNS v1.0 separadamente, sem reutilizar totais v0.9
   nem chamar todo endpoint excluído de glia. BANC fica registrado como alternativa
   feminina cérebro+cordão. Corpo 3D, Flybody/FlyGym e migração de datasets não iniciados.

## Comandos efetivamente executados

Os argumentos completos e saídas estão nos JSONs/logs do supervisor em
`outputs/first_cycle`. Entradas principais:

```powershell
git switch -c research/first-cycle-20260929
& .\.venv\Scripts\python.exe scripts/cycle_evidence.py inventory
& .\.venv\Scripts\python.exe -m pytest -q -ra --junitxml=outputs/first_cycle/final_pytest.xml
& C:\Users\Mazum\.cargo\bin\cargo.exe test --workspace --locked
& C:\Users\Mazum\.cargo\bin\cargo.exe clippy --workspace --all-targets --locked -- -D warnings
& C:\Users\Mazum\.cargo\bin\cargo.exe fmt --all -- --check
& .\.venv\Scripts\python.exe -m ruff check flybrain tests scripts --select F,E9
& .\.venv\Scripts\python.exe scripts/benchmark_rust.py
& .\.venv\Scripts\python.exe scripts/run_social_controls.py
& .\.venv\Scripts\python.exe scripts/run_tabular_pilot.py
& .\.venv\Scripts\python.exe scripts/reevaluate_tabular_pilot.py
& .\.venv\Scripts\python.exe scripts/replay_archived_v0.py
& .\.venv\Scripts\python.exe scripts/verify_first_cycle.py
```

Os scripts de evidência recusam sobrescrever resultados existentes. Para outra
campanha, escolher novo diretório/nome de run em vez de apagar estes artefatos.
Os comandos de build/install executados estão em [RUST_KERNEL.md](RUST_KERNEL.md)
e nos logs `final_wheel_build`/`final_wheel_install`.

Arquivos de implementação novos: `Cargo.toml`, `Cargo.lock`, dois crates,
`model/inputs.py`, `model/rust.py`, `social/{choice,policies,evaluation}.py`,
`learning/{tabular,interfaces}.py` e seus `__init__.py`. Alterados:
`network.py`, documentação de `stimulus.py`, `.gitignore`. Acrescentados quatro
arquivos de teste, sete scripts de evidência/execução e os cinco documentos desta
entrega. Artefatos volumosos permanecem sob `outputs/`, ignorado pelo Git.
