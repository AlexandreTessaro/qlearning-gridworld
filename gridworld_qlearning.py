import numpy as np
import random
import argparse
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd

class GridWorld:
    """
    Ambiente Grid World:
      - Estados: cada célula (linha, coluna).
      - Ações: 0=Cima, 1=Direita, 2=Baixo, 3=Esquerda.
      - Recompensas:
          • +1 ao chegar no objetivo.
          • -1 ao cair em um pit.
          • -0.04 a cada passo normal.
          • -0.1 para self-loops (movimento inválido).
    """
    def __init__(self, linhas=4, colunas=4, inicio=(0, 0), objetivo=(3, 3), pits=None, semente=None):
        self.linhas = linhas
        self.colunas = colunas
        self.inicio = inicio
        self.objetivo = objetivo
        self.pits = set(pits or [(1, 1), (2, 3)])
        if semente is not None:
            random.seed(semente)
            np.random.seed(semente)
        self.reiniciar()

    def reiniciar(self):
        self.estado = self.inicio
        return self.estado

    def passo(self, acao):
        r, c = self.estado
        if acao == 0:      # Cima
            nr, nc = max(r - 1, 0), c
        elif acao == 1:    # Direita
            nr, nc = r, min(c + 1, self.colunas - 1)
        elif acao == 2:    # Baixo
            nr, nc = min(r + 1, self.linhas - 1), c
        elif acao == 3:    # Esquerda
            nr, nc = r, max(c - 1, 0)
        else:
            nr, nc = r, c

        proximo_estado = (nr, nc)

        # verifica término
        if proximo_estado == self.objetivo:
            self.estado = proximo_estado
            return proximo_estado, 1.0, True
        if proximo_estado in self.pits:
            self.estado = proximo_estado
            return proximo_estado, -1.0, True

        # penalidade por self-loop ou passo normal
        if proximo_estado == self.estado:
            recompensa = -0.1
        else:
            recompensa = -0.04

        self.estado = proximo_estado
        return proximo_estado, recompensa, False

    def estado_para_indice(self, estado):
        return estado[0] * self.colunas + estado[1]

    def indice_para_estado(self, indice):
        return (indice // self.colunas, indice % self.colunas)


def treinamento_q_learning(env, episodios, alpha, gamma,
                          eps_inicial, eps_minimo, decaimento, passos_max):
    n_estados = env.linhas * env.colunas
    Q = np.zeros((n_estados, 4))
    recompensas = []
    epsilon = eps_inicial
    sucessos = 0

    for ep in trange(episodios, desc="Treino Q-Learning"):
        estado = env.reiniciar()
        s = env.estado_para_indice(estado)
        total = 0

        for _ in range(passos_max):
            # ε-greedy
            if random.random() < epsilon:
                a = random.randrange(4)
            else:
                a = int(np.argmax(Q[s]))

            prox, recompensa, terminado = env.passo(a)
            ns = env.estado_para_indice(prox)

            # atualização de Q
            Q[s, a] += alpha * (recompensa + gamma * np.max(Q[ns]) - Q[s, a])
            total += recompensa
            s = ns

            if terminado:
                if recompensa == 1.0:
                    sucessos += 1
                break

        recompensas.append(total)
        # decaimento de epsilon
        epsilon = max(eps_minimo, epsilon * decaimento)

        # exibe progresso a cada 100 episódios
        if (ep + 1) % 100 == 0:
            taxa = sucessos / (ep + 1) * 100
            print(f"Ep {ep+1}/{episodios} — Sucessos: {taxa:.1f}% — ε: {epsilon:.3f}")

    return Q, recompensas


def extrair_politica(Q, env):
    flechas = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    politica = []
    for idx in range(env.linhas * env.colunas):
        est = env.indice_para_estado(idx)
        if est == env.objetivo:
            politica.append('G')
        elif est in env.pits:
            politica.append('P')
        else:
            politica.append(flechas[int(np.argmax(Q[idx]))])
    return np.array(politica).reshape(env.linhas, env.colunas)


def avaliar_politica(Q, env, episodios, passos_max):
    cont_sucesso = 0
    lista_passos = []
    for _ in range(episodios):
        estado = env.reiniciar()
        for passo in range(1, passos_max + 1):
            s = env.estado_para_indice(estado)
            acao = int(np.argmax(Q[s]))
            estado, _, terminado = env.passo(acao)
            if terminado:
                if estado == env.objetivo:
                    cont_sucesso += 1
                    lista_passos.append(passo)
                break

    taxa = 100 * cont_sucesso / episodios
    media_passos = np.mean(lista_passos) if lista_passos else float('nan')
    return taxa, media_passos


def plotar_resultados(recompensas, politica, Q, env, janela):
    # recompensa média móvel
    ma = np.convolve(recompensas, np.ones(janela)/janela, mode='valid')
    plt.figure()
    plt.plot(ma)
    plt.title(f'Recompensa Média (janela={janela})')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa')
    plt.grid(True)
    plt.show()

    # heatmap de V(s) com a política
    V = np.max(Q, axis=1).reshape(env.linhas, env.colunas)
    plt.figure()
    plt.imshow(V, interpolation='nearest')
    plt.title('Mapa de Valores V(s) e Política')
    for i in range(env.linhas):
        for j in range(env.colunas):
            plt.text(j, i, politica[i, j], ha='center', va='center')
    plt.colorbar(label='V(s)')
    plt.show()


def exibir_tabelas(Q, politica, env):
    # Q-Table
    estados = [f"{r},{c}" for r in range(env.linhas) for c in range(env.colunas)]
    acoes = ['Cima', 'Direita', 'Baixo', 'Esquerda']
    df_q = pd.DataFrame(Q, index=estados, columns=acoes).round(3)
    print("\n=== Q-Table ===")
    print(df_q)

    # Política ótima
    df_pol = pd.DataFrame(politica,
                          index=[f"Linha {i}" for i in range(env.linhas)],
                          columns=[f"Coluna {j}" for j in range(env.colunas)])
    print("\n=== Política Ótima ===")
    print(df_pol)


def principal():
    parser = argparse.ArgumentParser(description="Q-Learning em GridWorld")
    parser.add_argument('--episodes',    type=int,   default=10000,  help='N° de episódios (treino)')
    parser.add_argument('--alpha',       type=float, default=0.5,    help='Taxa de aprendizado α')
    parser.add_argument('--gamma',       type=float, default=0.9,    help='Fator de desconto γ')
    parser.add_argument('--eps_start',   type=float, default=1.0,    help='ε inicial')
    parser.add_argument('--eps_min',     type=float, default=0.01,   help='ε mínimo')
    parser.add_argument('--decay',       type=float, default=0.995,  help='Taxa de decaimento de ε')
    parser.add_argument('--steps',       type=int,   default=100,    help='Máx. passos/episódio')
    parser.add_argument('--window',      type=int,   default=50,     help='Janela média móvel')
    parser.add_argument('--eval_eps',    type=int,   default=100,    help='Eps. para avaliação final')
    parser.add_argument('--seed',        type=int,   default=42,     help='Semente para reproducibilidade')

    args, _ = parser.parse_known_args()

    # usa o parâmetro 'semente' para inicializar o ambiente
    env = GridWorld(semente=args.seed)

    Q, recompensas = treinamento_q_learning(
        env,
        episodios=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_inicial=args.eps_start,
        eps_minimo=args.eps_min,
        decaimento=args.decay,
        passos_max=args.steps
    )

    politica = extrair_politica(Q, env)

    plotar_resultados(recompensas, politica, Q, env, args.window)
    exibir_tabelas(Q, politica, env)

    taxa, media_passos = avaliar_politica(Q, env, args.eval_eps, args.steps)
    print(f"\n=== Avaliação final (greedy) em {args.eval_eps} episódios ===")
    print(f"Taxa de sucesso: {taxa:.1f}%")
    print(f"Passos médios: {media_passos:.1f}\n")


if __name__ == '__main__':
    principal()
