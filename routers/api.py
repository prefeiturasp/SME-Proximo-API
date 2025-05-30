
from fastapi import APIRouter, HTTPException, Request, Body
from typing import Any, Dict
import numpy as np
from services.adaptive_testing import (
    transformar_parametros,
    EAP,
    criterio_parada,
    maxima_informacao_th,
    proximo_item_criterio,
    verificar_valid_eixo
)
from utils.helpers import parse_str_list, normalizar_componente

router = APIRouter()

# Configuração do exemplo para documentação
PROXIMO_ITEM_DOCS = {
    "summary": "Seleciona próximo item do teste adaptativo",
    "description": "Recebe respostas e parâmetros para estimar proficiência e selecionar próximo item",
    "response_description": "Array com informações do próximo item ou resultado final"
}

EXEMPLO_PAYLOAD = Body(
    ...,
    example={
        "ESTUDANTE": "Aluno1",
        "AnoEscolarEstudante": "8",
        "proficiencia": "500.0",
        "profic.inic": "500.0",
        "idItem": "ITEM1,ITEM2",
        "parA": "1.0,2.0",
        "parB": "250.0,300.0",
        "parC": "0.2,0.3",
        "administrado": "ITEM1",
        "respostas": "A",
        "gabarito": "A",
        "erropadrao": "0.5",
        "n.Ij": "45",
        "componente": "Língua portuguesa",
        "idEixo": "1,2",
        "idHabilidade": "2,3"
    }
)
@router.post("/proximo", **PROXIMO_ITEM_DOCS)
async def proximo_item(
    request: Request,
    payload: Dict[str, Any] = EXEMPLO_PAYLOAD
):
    body = await request.json()

    try:
        # Conversão dos campos do payload recebido
        ESTUDANTE = body["ESTUDANTE"]
        AnoEscolarEstudante = int(body["AnoEscolarEstudante"])
        proficiencia = float(body["proficiencia"])
        profic_inic = float(body["profic.inic"])
        idItem = body["idItem"].split(",")
        parA = parse_str_list(body["parA"])
        parB = parse_str_list(body["parB"])
        parC = parse_str_list(body["parC"])
        # Filtrar strings vazias para administrado, respostas e gabarito
        administrado = [a for a in body["administrado"].split(",") if a]
        respostas = [r for r in body["respostas"].split(",") if r]
        gabarito = [g for g in body["gabarito"].split(",") if g]
        if len(respostas) != len(gabarito):
            raise HTTPException(status_code=400, detail="respostas e gabarito devem ter o mesmo tamanho")
        erropadrao = float(body["erropadrao"])
        n_Ij = int(body["n.Ij"])
        componente = normalizar_componente(body["componente"])
        idEixo = parse_str_list(body["idEixo"], int)
        idHabilidade = parse_str_list(body["idHabilidade"], int)

        # Gabarito corrigido (0/1)
        respostas_corrigidas = np.array([1 if r == g else 0 for r, g in zip(respostas, gabarito)])

        administrado_idx = [idx for idx, item in enumerate(idItem) if item in set(administrado)]
        PAR = np.column_stack((parA, parB, parC))
        PAR = transformar_parametros(PAR, componente)

        # NOVO: calcular validEixo com base nos eixos aplicados - Corrige parada na 8 questão
        validEixo = verificar_valid_eixo(administrado_idx, idEixo)
        
        if len(respostas_corrigidas) == 0:
            # PRIMEIRA RESPOSTA
            theta_est_ep = (profic_inic - 249.985) / 55.093
            INFO = maxima_informacao_th(theta_est_ep, PAR)
            pos = proximo_item_criterio(INFO, administrado_idx)
            return [
                idItem[pos],
                "1",
                str(pos),
                str(round(PAR[pos, 0], 6)),
                str(round(PAR[pos, 1], 14)),
                str(round(parC[pos], 3)),
                str(round(profic_inic, 13)),
                "NA"
            ]
        else:
            # ESTIMA PROFICIÊNCIA
            PAR_adm = PAR[administrado_idx, :]
            theta_est, theta_ep = EAP(respostas_corrigidas, PAR_adm, administrado_idx)

            # NOVO: enviar validEixo - Corrige parada na 8 questão
            parar = criterio_parada(
                theta_est, theta_ep, Area=componente, AnoEscolar=AnoEscolarEstudante,
                n_resp=len(respostas_corrigidas), n_Ij=n_Ij, validEixo=validEixo
            )

            if not parar:
                INFO = maxima_informacao_th(theta_est, PAR)
                pos = proximo_item_criterio(INFO, administrado_idx)

                # NOVO: Aplica a escala SAEB correta conforme o componente - Corrige diferença de escala
                if componente == "LP":
                    theta_saeb = theta_est * 55.093 + 249.985
                    erro_saeb = theta_ep * 55.093
                elif componente == "MT":
                    theta_saeb = theta_est * 55.892 + 249.964
                    erro_saeb = theta_ep * 55.892
                elif componente == "CN":
                    theta_saeb = theta_est * 55.789 + 249.955
                    erro_saeb = theta_ep * 55.789
                else:  # componente CH ou outros
                    theta_saeb = theta_est * 55.093 + 249.985
                    erro_saeb = theta_ep * 55.093

                return [
                    idItem[pos],
                    str(len(respostas_corrigidas) + 1),
                    str(pos),
                    str(round(PAR[pos, 0], 6)),
                    str(round(PAR[pos, 1], 14)),
                    str(round(parC[pos], 3)),
                    str(round(theta_saeb, 12)),
                    str(round(erro_saeb, 13))
                ]
            else:
                # NOVO: Aplica a escala SAEB correta conforme o componente - Corrige diferença de escala
                if componente == "LP":
                    theta_saeb = theta_est * 55.093 + 249.985
                    erro_saeb = theta_ep * 55.093
                elif componente == "MT":
                    theta_saeb = theta_est * 55.892 + 249.964
                    erro_saeb = theta_ep * 55.892
                elif componente == "CN":
                    theta_saeb = theta_est * 55.789 + 249.955
                    erro_saeb = theta_ep * 55.789
                else:  # componente CH ou outros
                    theta_saeb = theta_est * 55.093 + 249.985
                    erro_saeb = theta_ep * 55.093

                return [
                    "-1",
                    str(len(respostas_corrigidas)),
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    str(round(theta_saeb, 12)),
                    str(round(erro_saeb, 13))
                ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.post("/pingR")
def ping():
    return {"status": "200"}    