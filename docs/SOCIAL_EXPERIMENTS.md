# C1 e o próximo protocolo de retenção

Implementado: `flybrain/social/choice.py`, controles em `policies.py`, avaliador
em `evaluation.py` e piloto em `learning/tabular.py`. O ambiente é discreto,
com duas opções, dois indivíduos e um alfabeto de três ações de emissão. Não usa
conectoma nem corpo. A recompensa cooperativa é um pressuposto declarado.

## Fronteira de informação

O mundo sorteia um objeto relevante binário IID equiprovável. Papéis e ordem das
opções vêm de outro fluxo aleatório, independente do alvo. O emissor vê a pista
privada na fase `signal`; o receptor recebe apenas a ordem pública das opções e,
na fase `choice`, o símbolo recebido. Só o receptor escolhe a opção. O mundo
calcula recompensa pelo objeto escolhido; não contém tabela mensagem→resposta.
O símbolo 2 representa ausência de emissão, que também pode adquirir significado
num protocolo aprendido. Bloqueio experimental usa `None`, distinguível dos três
símbolos treináveis. Reamostragem preserva a marginal global dos símbolos e
complementa esse bloqueio, evitando depender apenas de uma entrada inédita.

Observações são dataclasses congeladas. Apenas o avaliador usa o construtor de
mundos controlados e a função de retorno esperado. Os atores recebem observações;
os aprendizes recebem observação própria, ação própria e recompensa ambiental.
Isso é separação de fluxo em código confiável, não um sandbox contra políticas
Python maliciosas. Estado global, sementes do mundo e rótulos do avaliador não
são entregues aos indivíduos.

O reset do mundo remove mensagem e resultado anteriores e não reinicia os pesos
dos indivíduos. Os controles e o piloto atual são políticas sem memória temporal.
Uma política recorrente futura exigirá contrato explícito de reset de memória e
uma tarefa/protocolo apropriados; os resultados atuais não validam esse caso.

## Avaliação causal

O avaliador cria o produto cartesiano balanceado de alvo, papel e ordem pública,
embaralhado em cada avaliação. O cronograma e os índices não chegam aos atores.
Antes da mensagem, copia mundo, RNGs e indivíduo receptor. Cada intervenção
reexecuta emissão, recepção e escolha com o mesmo estado aleatório do receptor.
Os indivíduos treinados ficam congelados: os testes verificam que a avaliação
não muda seus pesos, baselines ou RNGs.

São reportados retorno esperado da distribuição de ações, retorno amostrado,
ganho contra bloqueio, ganho contra reamostragem independente do alvo oculto,
TV entre símbolos 0/1 e TV média sobre todos os pares sob q uniforme nos três
símbolos. A classificação usa a última medida, sendo invariante à renomeação dos
símbolos. Reamostragem não é estratificada pela resposta correta. C1 termina após
uma escolha: não há decisão posterior para medir; isso não exclui efeitos tardios
em tarefas futuras.

Critério operacional exploratório, definido no código: retorno ≥0,75, ganho
contra reamostragem ≥0,10 e TV média ≥0,10. Não é limiar universal nem teste
confirmatório. Os relatórios mantêm todas as sementes e limitam os exemplos de
replay a 16 episódios, marcando truncamento do registro.

Os controles manual, aleatório e sem informação precisam passar antes de treinar.
Testes adicionais distinguem emissor informativo ignorado de comunicação e
receptor sensível recebendo sinais não informativos de comunicação funcional.

## Piloto tabular

Três sementes, 12.000 episódios por semente, limite de 30 s de treinamento por
semente, 512 episódios de avaliação antes e depois. Cada indivíduo tem 22 logits
e 10 baselines locais, inicialização e RNG próprios. Usa score-function/REINFORCE
com baseline de resultado por observação local: taxa 0,06, EMA 0,05, sem bônus de
mensagem, informação, influência ou entropia. Os dois papéis são sorteados a cada
episódio e cada indivíduo possui suas próprias tabelas dos dois papéis.

Não há respostas do controle manual no treinamento. Os pesos finais, hashes,
curvas e avaliações ficam em `outputs/first_cycle/tabular_pilot/`. Uma segunda
avaliação carrega esses pesos e usa novas sementes de mundo e ação, sem retreinar.
PPO recorrente permanece opcional e não executado; este piloto realiza a opção
tabular autorizada. Não representa plasticidade de conectoma nem retenção social.

## Próxima etapa preparada: S1 e competência individual

A tarefa C1 não permite avaliar competência solo do receptor quando removemos
simultaneamente a única pista e o canal: esse teste seria impossível por desenho.
S1 precisa ser outra tarefa, na qual o aprendiz tenha observações solo suficientes
para aplicar uma associação adquirida, sem receber a resposta correta.

Protocolo proposto, ainda não executado:

1. Pré-teste solo, com percepções locais de objetos e sem recompensa de ensino.
2. Exposição a parceiro competente versus parceiro não informativo/replay de outra
   sessão; igualar oportunidades de experiência própria, tempo e recompensa própria.
3. Retirar parceiro, canal e pistas ambientais persistentes; congelar aprendizagem.
4. Testar sozinho em novos episódios sem feedback que ensine a solução no teste.
5. Comparar diferenças pareadas entre indivíduos independentes, preservando pesos
   ou dissipando estados rápidos em condições separadas. Restaurar pesos iniciais
   em outra condição. Nenhum novato recebe checkpoint do demonstrador.

Antes do experimento mecanístico social, exigir competência individual em tarefa
de recompensa atrasada. `learning/interfaces.py` prepara `LocalTransition`,
`EligibilityReadout` e `PlasticityMask`. A máscara exige identidade do dataset,
IDs originais das arestas e justificativa; deltas, traços e pesos mutáveis deverão
ser individuais. O readout é um componente externo. Não há implementação nem
resultado de actor-critic com elegibilidade ou plasticidade interna nesta entrega.
