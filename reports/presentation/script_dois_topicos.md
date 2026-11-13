# Script oral — dois tópicos

Fala contínua para os slides “Quanto da nota sensorial os testes de laboratório explicam?” e “Preparação e divisão dos dados”. Os números são os da execução do projeto.

## 1. Quanto da nota sensorial os testes de laboratório explicam?

Cerca de 2 minutos.

A nota de um vinho não sai de uma fórmula. Ela sai de especialistas. Cada provador dá uma nota, e a qualidade registrada no dataset é a mediana de pelo menos três avaliações, numa escala de 0 a 10.

O laboratório, por outro lado, mede coisas objetivas: acidez, açúcar, sulfatos, álcool e outras propriedades físico-químicas. A pergunta deste trabalho é simples: quanto dessa nota sensorial esses testes conseguem explicar?

O alvo é a coluna `quality`. Juntamos tintos e brancos no mesmo modelo e tratamos o problema como regressão supervisionada. A nota permanece um número. Não viramos o problema em classificação.

O objetivo, então, é prever a qualidade e comparar dois algoritmos na mesma divisão dos dados: a regressão linear e o Random Forest. A comparação só vale porque os dois modelos veem exatamente os mesmos vinhos de treino e os mesmos vinhos de teste.

## 2. Preparação e divisão dos dados

Cerca de 3 minutos.

Antes de treinar, a base passou por uma limpeza curta e registrada.

Não havia valor ausente. Por isso não fizemos imputação.

Havia 1.177 linhas exatamente iguais, já contando o tipo do vinho. Essas duplicatas foram removidas. A primeira ocorrência ficou.

Os outliers foram medidos pelo intervalo interquartil. 1.094 registros saem desse limite em pelo menos um atributo físico-químico. Eles foram mantidos. Não apagamos vinho raro só porque a medida foge da caixa.

Os preditores são os 11 atributos de laboratório mais o tipo: branco igual a 0 e tinto igual a 1. Não houve seleção de atributos. Os dois modelos recebem o mesmo conjunto.

A base tratada tem 5.320 vinhos. Ela foi embaralhada uma única vez, com `random_state` 42: 4.256 para treino, 80%, e 1.064 para teste, 20%.

A padronização fica dentro do pipeline. O `StandardScaler` aprende a média e o desvio só no treino. O teste entra depois, com essa escala já fixa. Assim não há vazamento.

A validação cruzada também fica só no treino: cinco partições embaralhadas. O embaralhamento importa porque, na tabela combinada, os brancos vêm antes dos tintos. Sem ele, alguma partição ficaria quase só de um tipo. O conjunto de teste não entra nessa validação. Ele fica reservado para a comparação final.
