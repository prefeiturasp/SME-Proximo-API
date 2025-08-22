import numpy as np
from collections import Counter
from typing import Tuple, Optional, Set

def transformar_parametros(PAR, componente):
    """
    Transforma os parâmetros dos itens (parA e parB) com base no componente.
    """
    if componente == "LP":
        # Transformação para Língua Portuguesa
        PAR[:, 0] = PAR[:, 0] * 55.093
        PAR[:, 1] = (PAR[:, 1] - 249.985) / 55.093
    elif componente == "MT":
        # Transformação para Matemática
        PAR[:, 0] = PAR[:, 0] * 55.892
        PAR[:, 1] = (PAR[:, 1] - 249.964) / 55.892
    elif componente == "CN":
        # Transformação para Ciências da Natureza
        PAR[:, 0] = PAR[:, 0] * 55.7899
        PAR[:, 1] = (PAR[:, 1] - 249.955) / 55.7899
    elif componente == "CH":
        # Transformação para Ciências Humanas (valores hipotéticos, ajustar conforme necessário)
        PAR[:, 0] = PAR[:, 0] * 55.093  # Exemplo, ajustar conforme a escala de CH
        PAR[:, 1] = (PAR[:, 1] - 249.985) / 55.093  # Exemplo, ajustar conforme a escala de CH
    else:
        raise ValueError(f"Componente desconhecido: {componente}")

    return PAR

def EAP(U, PAR, administrado):
    print("\n=== Iniciando EAP ===")

    U = np.array(U).reshape(1, -1)
    naoNA = np.where(~np.isnan(U))[1]
    It = len(naoNA)

    q = 61
    Xr = np.linspace(-6, 6, q).reshape(-1, 1)
    AXr = 1 / np.sqrt(2 * np.pi) * np.exp(-(Xr**2) / 2) * 8 / (q - 1)

    a, b, c = PAR[:, 0], PAR[:, 1], PAR[:, 2]
    ak = np.ones((q, 1)) @ a.reshape(1, -1)
    bk = np.ones((q, 1)) @ b.reshape(1, -1)
    ck = np.ones((q, 1)) @ c.reshape(1, -1)

    Xrk = np.tile(Xr, (1, len(a)))
    P = ck + (1 - ck) / (1 + np.exp(-ak * (Xrk - bk)))

    

    Pjt = np.zeros((q, 1))
    theta_est = np.zeros(1)
    for j in range(1):
        for l in range(q):
            veroj = 1
            for i in range(It):
                if not np.isnan(U[j, i]):
                    veroj *= P[l, i] ** U[j, i] * (1 - P[l, i]) ** (1 - U[j, i])
            Pjt[l] = veroj
        Pj = Pjt * AXr
        tPj = Xr * Pj
        theta_est[j] = np.sum(tPj) / np.sum(Pj)

    squad = (Xr - theta_est) ** 2
    ep_est = np.sqrt(np.sum(squad * Pj) / np.sum(Pj))


    return float(theta_est[0]), float(ep_est)

def parar_teste(theta, theta_erro, pontos_corte, valor_critico=1):
    theta_range = [theta - valor_critico * theta_erro, theta + valor_critico * theta_erro]
    ff = np.digitize(theta_range, pontos_corte)
    return 1 if len(np.unique(ff)) == 1 else 0

def criterio_parada(
    theta_est,
    theta_ep,
    parada="EP",
    EP=0.35,            # padrão do R
    n_resp=0,
    n_min=8,
    validEixo=True,
    Area="LP",
    AnoEscolar=8,
    n_Ij=45,
):
    """
    Critério de parada compatível com a API em R.
    - Parada por EP: sem exigência de n_resp >= 16.
    - Parada por intervalo (faixa de proficiência).
    - Parada por máximo de itens (32 ou n_Ij-2).
    - Só permite parada quando validEixo == True.
    """
    # Tabelas de níveis (pontos de corte) usadas no R
    niveis = {
        "LP": {
            2: [-2.722396675, -2.268618518, -1.361062204],
            3: [-2.268618518, -1.361062204, -0.45350589],
            4: [-2.087107255, -1.179550941, -0.271994627],
            5: [-1.814840361, -0.907284047, 0.000272267],
            6: [-1.542573467, -0.635017153, 0.272539161],
            7: [-1.361062204, -0.45350589, 0.454050424],
            8: [-1.179550941, 0.000272267, 0.907828581],
            9: [-0.89393831, 0.447935304, 1.342517713],
        },
        "MT": {
            2: [-2.239742, -1.343523, -0.895413],
            3: [-1.791633, -0.895413, 0.0008],
            4: [-1.522767, -0.7161691, 0.2696725],
            5: [-1.343523, -0.4473032, 0.4489164],
            6: [-1.074657, -0.1784373, 0.7177823],
            7: [-0.895413, 0.0008, 0.8970262],
            8: [-0.7161691, 0.4489164, 1.345136],
            9: [-0.4473032, 0.8970262, 1.793246],
        },
    }

    pontos_corte = niveis.get(Area, {}).get(AnoEscolar, [])
    valor_critico = 1

    if n_resp < n_min:
        return False

    # Parada por EP (sem trava de 16 itens)
    if parada == "EP" and theta_ep <= EP and validEixo:
        return True

    # Parada por intervalo de proficiência
    if pontos_corte and validEixo:
        theta_range = [theta_est - valor_critico * theta_ep, theta_est + valor_critico * theta_ep]
        import numpy as np
        ff = np.digitize(theta_range, pontos_corte)
        if len(np.unique(ff)) == 1:
            return True

    # Parada por número máximo de itens
    if n_resp == 32 or n_resp == n_Ij - 2:
        return True

    return False

def maxima_informacao_th(theta_est, PAR, D=1):
    """
    Calcula a informação de Fisher para um dado valor de proficiência (theta_est).
    """
    a, b, c = PAR[:, 0], PAR[:, 1], PAR[:, 2]

    # Calcula a probabilidade de resposta correta (P)
    P = c + (1 - c) / (1 + np.exp(-D * a * (theta_est - b)))

    # Calcula a informação de Fisher
    max_info = (D ** 2) * (a ** 2) * ((1 - P) / P) * (((P - c) / (1 - c)) ** 2)

    return max_info

def proximo_item_criterio(INFO, administrado):
    print("\n=== Iniciando proximo_item_criterio ===")

    # Zerar a informação dos itens já administrados
    INFO[administrado] = 0
    
    # Selecionar o item com a maior informação de Fisher
    pos = np.argmax(INFO)
    return int(pos)


def verificar_valid_eixo(
administrado_idx: list[int],
    id_eixo: list[int],
    n_resp: int
) -> bool:
    """
    Política de eixo compatível com o código em R.

    Retorna:
      - validEixo: bool -> se a cobertura mínima por eixo já foi cumprida
        (só então é permitido aplicar o critério de parada).

    Regras:
      1) Considera TODOS os eixos do pool de itens (id_eixo), não apenas os administrados.
      2) Se ainda não cobriu TODOS os eixos ao menos 1 vez => validEixo=False.
      3) Se já cobriu todos:
         - Se len(eixos) < 4: alvo mínimo = 3 por eixo.
         - Se len(eixos) >= 4: alvo mínimo = 2 por eixo.
         Se total aplicado >= (#eixos * alvo) => validEixo=True.
         Caso contrário => False.
    """
    eixos = list(dict.fromkeys(id_eixo))
    num_eixos = len(eixos)

    # Atalho: com 0/1 resposta, considera válido (compatível com o fluxo em R)
    if n_resp <= 1:
        return True

    # Contagem por eixo entre os itens administrados
    eixos_aplicados = [id_eixo[i] for i in administrado_idx if 0 <= i < len(id_eixo)]
    contagem = Counter(eixos_aplicados)
    vistos = set(contagem.keys())

    # Se ainda não cobriu todos os eixos, não permite parada
    if not set(eixos).issubset(vistos):
        return False

    # Já cobriu todos; checar alvo por eixo
    alvo = 3 if num_eixos < 4 else 2
    total_aplicado = sum(contagem.values())

    return total_aplicado >= num_eixos * alvo