# CLAUDE.md — g1_training

## Minimização de código

Antes de adicionar qualquer coisa ao código, procure uma função que já existe e que
possa ser modificada ou expandida para atender a demanda. Adicionar é a última opção,
não a primeira.

Sempre que possível, reduza o tamanho do código. Delete função inútil e função
repetida. Uma mudança que só soma linhas precisa de justificativa explícita.

Ao propor uma mudança, mostre a contagem antes → depois.

## Desenho de recompensa

**Mais termo de recompensa ou de penalidade não traz o efeito desejado.** Um termo novo
só vale quando está muito bem pensado. Na dúvida, conserte a forma de um termo que já
existe em vez de acrescentar outro.

**Recompense antes de penalizar.** A penalização não ensina nada — ela só diz o que não
fazer, e o espaço do "não fazer" é infinito. A recompensa diz para onde ir.

**Toda ação, tarefa e atividade precisa de gradiente de recompensa** que leve o robô a
executar o que precisa ser feito. Esperar que a aleatoriedade da exploração traga o
comportamento desejado não funciona.

Consequência prática: um termo com derivada zero na região onde o robô está hoje é um
canal morto, por mais alto que seja o peso dele. Meça a derivada no ponto de operação
real antes de propor o peso.
