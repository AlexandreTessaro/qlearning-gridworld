
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

### Conceito Elaborado

1. **Processo de Decisão de Markov (MDP)**  
   O ambiente FrozenLake pode ser formalizado como um MDP, definido pelo conjunto de estados S, conjunto de ações A, função de transição P(s'|s,a) e função de recompensa R(s,a). Em cada passo t:
    
    - O agente observa um estado s_t ∈ S.
    - Escolhe uma ação a_t ∈ A de acordo com sua política.
    - Recebe uma recompensa r_{t+1} = R(s_t, a_t).
    - Transita para um novo estado s_{t+1} segundo P(s_{t+1}|s_t, a_t).

2. **Função-Valor de Ação Q(s,a)**  
   A função Q(s,a) representa a recompensa acumulada esperada ao executar a ação a no estado s e, a partir daí, seguir uma política ótima. O objetivo do Q-Learning é aproximar Q*(s,a), a função-valor ótima, que satisfaz a equação de Bellman:

    Q*(s,a) = E[r_{t+1} + γ max_{a'} Q*(s_{t+1},a') | s_t=s, a_t=a]
   
   Essa equação mostra que o valor de uma ação depende da recompensa imediata mais o valor futuro do melhor caminho a partir do próximo estado. Em termos simples, a equação de Bellman ajuda o agente a pensar assim:"Se eu fizer essa ação agora, quanto ganho agora e quanto poderei ganhar depois, assumindo que tomarei as melhores decisões a partir daqui?"
   
4. **Diferença Temporal (TD) e Atualização**  
   A cada experiência (s_t, a_t, r_{t+1}, s_{t+1}), aplica-se o update de TD:

    Q(s_t, a_t) ← Q(s_t, a_t)
      + α [r_{t+1} + γ max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]

   - Erro Temporal δ = r_{t+1} + γ max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t): mede a diferença entre o valor atual e a estimativa baseada na transição.
   - Taxa de aprendizado α: controla quanto do erro temporal é incorporado em cada atualização.
   - Fator de desconto γ: pondera a importância de recompensas futuras em relação às imediatas.

   Esse processo permite ao agente aprender diretamente com a interação com o ambiente, sem precisar conhecer como ele funciona internamente (transições e recompensas), tornando o Q-Learning um algoritmo modelo-free.

5. **Política ε-Greedy**  
   Para balancear exploração e explotação, utiliza-se:

    π(a|s) =
      - ação aleatória, com probabilidade ε,
      - argmax_a Q(s,a), com probabilidade 1 - ε.

6. **Convergência e Extração da Política Ótima**  
   Sob condições adequadas de visitas a todos os pares (s,a) e escolha decrescente de α e ε, o Q-Learning converge quase certamente para Q*(s,a). A política derivada é:

    π*(s) = argmax_a Q*(s,a).

7. **Vantagens e Limitações**  
   - Vantagens:
     - Modelo-free (não requer conhecer P nem R).
     - Simples de implementar e eficaz em MDPs discretos.
   - Limitações:
     - Escalabilidade limitada em grandes espaços de estados.
     - Sensível à escolha de hiperparâmetros e à taxa de decaimento de ε.

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
