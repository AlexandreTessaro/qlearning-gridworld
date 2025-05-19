
# Relatório - Q-Learning no Ambiente FrozenLake (OpenAI Gym)

## Integrantes
- Alexandre Tessaro
- Edson Borges Polucena
- Leonardo Pereira Borges
- Richard Schmitz Riedo
- Wuelliton Christian Dos Santos

## Descrição do Algoritmo

O **Q-Learning** é um algoritmo de aprendizado por reforço off-policy que busca aprender a função de valor ótima (Q-Table) para cada par estado-ação. Ele se baseia na fórmula de atualização:

```
Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

Onde:
- `s`: estado atual  
- `a`: ação escolhida  
- `r`: recompensa recebida  
- `s'`: próximo estado  
- `α`: taxa de aprendizado (learning rate)  
- `γ`: fator de desconto (discount factor)

A política de ação é baseada em **ε-greedy**, permitindo que o agente explore aleatoriamente com probabilidade ε ou escolha a melhor ação conhecida com probabilidade (1 - ε). O ε é decaído ao longo do tempo.

### Hiperparâmetros Utilizados
- **α (learning rate):** 0.8  
- **γ (discount factor):** 0.95  
- **ε inicial:** 1.0  
- **ε final:** 0.01  
- **Decaimento de ε:** ε = ε * 0.995 por episódio  
- **Episódios de treinamento:** 10.000

---

## Exemplo de Entrada e Saída

### Ambiente: FrozenLake-v1 (4x4, não deslizante)
```
S  F  F  F  
F  H  F  H  
F  F  F  H  
H  F  F  G  
```
- `S`: posição inicial do agente  
- `F`: estado livre  
- `H`: buraco (poço) — termina o jogo com fracasso  
- `G`: objetivo

### Política ótima aprendida:
```
↓  →  →  ↓  
→  X  ↓  X  
→  →  ↓  X  
X  →  →  G  
```

Onde:
- Setas indicam a ação ótima aprendida em cada estado.  
- `X` representa buracos (ações não consideradas).  
- `G` é o objetivo (ação final).

---

## Resultados

### Evolução da taxa de sucesso durante o treinamento:
- **Início:** ~4% de sucesso
- **Fim (10.000 episódios):** **98.3%** de sucesso

### Avaliação da política final (greedy, sem exploração):
- **Taxa de sucesso:** 100%
- **Média de passos até o objetivo:** 6.0

### Q-Table Final (exemplo simplificado):

| Estado | Cima | Direita | Baixo | Esquerda |
|--------|------|---------|-------|----------|
| 0      | 0.00 | 0.45    | 0.00  | 0.00     |
| 1      | 0.00 | 0.54    | 0.00  | 0.33     |
| 2      | 0.00 | 0.63    | 0.00  | 0.38     |
| ...    | ...  | ...     | ...   | ...      |

---

## Dificuldades e Aprendizados

- **Ajuste de hiperparâmetros:** encontrar bons valores de ε e seu decaimento foi essencial para equilibrar exploração e convergência.
- **Convergência lenta:** sem o decaimento adequado, o agente demorava muito para aprender uma política útil.
- **Visualização da política:** interpretar a Q-table e transformá-la em direções visuais foi um exercício importante de compreensão.
- **Ambiente determinístico:** desligar o modo "slippery" tornou o aprendizado mais direto e ajudou na depuração.

---

## Conclusão

O Q-Learning se mostrou eficaz para resolver o problema do FrozenLake, especialmente com ambiente não deslizante. A política ótima foi aprendida com alto índice de sucesso após 10.000 episódios, e a avaliação final mostrou comportamento consistente e eficiente.

---
