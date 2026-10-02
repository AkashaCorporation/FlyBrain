# Núcleo Rust experimental da primeira entrega

`crates/flybrain-core` contém apenas Rust/std, uma thread, arrays contíguos e
buffers reutilizados. Integra todos os neurônios a cada passo fixo. CSR contém
as arestas de saída; apenas as fontes com spikes atrasados contribuem. Colisões
são reduzidas em float64 e convertidas para float32 antes de somar ao estado.
`reference_order=True` ordena contribuições pelo ID original da aresta;
`False` conserva a ordem CSR. O teste de cancelamento extremo demonstra que esses
modos podem divergir; não são universalmente bit-idênticos.

A fila de atraso usa buckets de índices de spikes com capacidade previamente
reservada. Integração não usa operações fused-multiply-add. Coeficientes vêm da
mesma construção float64 da bancada e são estreitados para float32. Tempos de
último spike permanecem float32 para o contrato de referência; os contadores
Rust são u64, diferentemente dos contadores float32 históricos. A equivalência
testada não se estende a contagens acima da resolução exata do float32.

`crates/flybrain-python` usa PyO3 0.27.2; seu `pyproject.toml` fixa maturin 1.10.2.
`Cargo.lock` registra as dependências e checksums. O inventário de licenças,
versões e hashes dos manifests está em `outputs/first_cycle/preservation_audit.json`.
O código do núcleo foi escrito localmente; não incorpora código das stacks de
terceiros presentes em `stack/`. PyO3 e maturin são dependências de build.

## Uso e limites

```python
from flybrain.model.rust import RustBrain, RecordingOptions

brain = RustBrain(connectome, max_memory_bytes=256_000_000)
result = brain.advance(
    200,
    inputs=[(0, 0, 68.75, 0.0)],  # passo relativo, índice, delta_v, delta_g
    recording_options=RecordingOptions(max_spikes=100, max_seconds=10),
    populations=[[0], [1, 2]],
)
```

Inputs devem estar ordenados por passo e podem conter colisões. `dv` entra após
limiar, antes de recorrência; `dg` entra após recorrência. Não há RNG dentro do
núcleo: primeiro valida-se a aritmética com eventos gravados. O adaptador não
promete executar automaticamente os novos sensores contínuos em Rust.

A chamada libera o GIL durante o bloco, copia entradas e retorna contagens
pequenas. `state()` copia o estado completo apenas quando pedido. `completed`,
`stop_reason`, passos concluídos, spikes, arestas percorridas e truncamento de
gravação são explícitos. Um bloco interrompido não reseta o cérebro; o chamador
deve deslocar os eventos restantes para retomar. O limite temporal é verificado
entre passos, portanto pode ultrapassar o teto pelo custo de um passo.

A guarda estima buffers nativos e o custo conservador das tuplas Python antes da
conversão. Não é um alocador com limite rígido. As execuções da entrega foram
também monitoradas por `scripts/cycle_evidence.py`, incluindo filhos, limite de
RAM, duração e tamanho de log. RSS de pico é amostrado a cada 100 ms.

O adaptador inicial por tuplas não é apropriado para importar whole-brain com o
teto de 256 MB: para v630 estima aproximadamente 3,80 GB só no orçamento de
grafo/ponte. O protocolo whole-brain não foi executado nem substituído por subset.
Uma ponte por buffers será necessária antes desse benchmark. O benchmark atual
tem 128 neurônios/1.024 arestas, três regimes de entrada, aquecimento separado,
cinco blocos de 2.000 passos por regime, mediana e p95, memória, spikes e arestas.

## Build local reproduzível

Na raiz MelanoGraph, PowerShell, usando o toolchain já instalado:

```powershell
$env:PATH = 'C:\Users\Mazum\.cargo\bin;' + $env:PATH
$env:PYO3_PYTHON = 'E:\HipoCampo\MelanoGraph\.venv\Scripts\python.exe'
& .\.venv\Scripts\python.exe -m pip install maturin==1.10.2
cargo test --workspace --locked
& .\.venv\Scripts\python.exe -m maturin build --manifest-path crates/flybrain-python/Cargo.toml --interpreter $env:PYO3_PYTHON --release --locked --out outputs/local-wheels
& .\.venv\Scripts\python.exe -m pip install --no-deps --force-reinstall outputs/local-wheels/flybrain_native-0.1.0-cp311-cp311-win_amd64.whl
& .\.venv\Scripts\python.exe -m pytest tests/test_rust_differential.py -q
```

O nome da wheel acima corresponde ao Python 3.11/Windows x64 desta máquina.
Não selecionar uma wheel de outra ABI. O pacote Python existente continua com
setuptools; a extensão opcional é uma distribuição separada. Pedir Rust sem a
extensão produz erro explícito, nunca fallback NumPy.
