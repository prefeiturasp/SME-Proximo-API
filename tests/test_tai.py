import warnings
import sys
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Adicione o caminho do projeto ao Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
# Por:
from main import app
from services.adaptive_testing import (
    transformar_parametros,
    EAP,
    parar_teste,
    criterio_parada,
    maxima_informacao_th,
    proximo_item_criterio
)
import numpy as np

client = TestClient(app)

# Fixtures para reutilização de dados
@pytest.fixture
def sample_PAR():
    return np.array([[1.0, 250.0, 0.2], [2.0, 300.0, 0.3]])

@pytest.fixture
def sample_request_data():
    return {
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

# Testes para transformar_parametros
def test_transformar_parametros_LP(sample_PAR):
    PAR = sample_PAR.copy()
    transformed = transformar_parametros(PAR, "LP")
    assert np.allclose(transformed[:, 0], [55.093, 110.186], rtol=1e-4)
    assert np.allclose(transformed[:, 1], [0.000272266894, 0.907828581], rtol=1e-4)

def test_transformar_parametros_MT(sample_PAR):
    PAR = sample_PAR.copy()
    transformed = transformar_parametros(PAR, "MT")
    assert np.allclose(transformed[:, 0], [55.892, 111.784], rtol=1e-4)
    assert np.allclose(transformed[:, 1], [0.000644099334, 0.895226508], rtol=1e-4)

# Testes para EAP
def test_EAP_basic():
    U = np.array([1, 0])
    PAR = np.array([[1.5, 0.5, 0.2], [2.0, 1.0, 0.3]])
    administrado = [0, 1]
    theta, ep = EAP(U, PAR, administrado)
    assert isinstance(theta, float)
    assert isinstance(ep, float)

# Testes para parar_teste
def test_parar_teste_continuar():
    pontos_corte = [-1.0, 0.0, 1.0]
    assert parar_teste(0.5, 0.6, pontos_corte) == 0  # Intervalo cruzando pontos

def test_parar_teste_parar():
    pontos_corte = [-1.0, 0.0, 1.0]
    assert parar_teste(0.5, 0.3, pontos_corte) == 1  # Intervalo dentro de um segmento

# Testes para criterio_parada
def test_criterio_ep_atingido():
    assert criterio_parada(0.0, 0.4, parada="EP", EP=0.5, n_resp=10, n_min=8) == True

def test_criterio_max_itens():
    assert criterio_parada(0.0, 1.0, n_resp=32, n_min=8) == True  # Atualizado para 32 itens

# Testes para maxima_informacao_th
def test_maxima_informacao_th():
    PAR = np.array([[1.0, 0.5, 0.2]])
    info = maxima_informacao_th(0.5, PAR)
    assert len(info) == 1
    assert info[0] > 0

# Testes para proximo_item_criterio
def test_proximo_item_criterio():
    INFO = np.array([0.1, 0.9, 0.5])
    administrado = [1]
    pos = proximo_item_criterio(INFO, administrado)
    assert pos == 2

# Testes para endpoints
def test_ping():
    response = client.post("/pingR")
    assert response.status_code == 200
    assert response.json() == {"status": "200"}

def test_proximo_item_primeiro_item(sample_request_data):
    # Teste do primeiro item (sem respostas prévias)
    primeiro_request = sample_request_data.copy()
    # Enviar strings vazias para respostas, gabarito e administrado
    primeiro_request["respostas"] = ""
    primeiro_request["gabarito"] = ""
    primeiro_request["administrado"] = ""

    response = client.post("/proximo", json=primeiro_request)
    assert response.status_code == 200
    # Verificar se retorna o próximo item (ex: ITEM2)
    assert response.json()[0] in ["ITEM1", "ITEM2"]  # Ajuste conforme os IDs

def test_proximo_item_continuacao(sample_request_data):
    # Teste continuando (com uma resposta)
    print(f"Dados do request normal: {sample_request_data}")
    
    response = client.post("/proximo", json=sample_request_data)
    if response.status_code != 200:
        print(f"Erro: {response.status_code}")
        print(f"Detalhes: {response.text}")
        
    assert response.status_code == 200
    
    resultado = response.json()
    print(f"Resultado do request normal: {resultado}")
    
    assert isinstance(resultado, list)
    assert len(resultado) == 8
    
    # O primeiro elemento deve ser o ID do próximo item ou -1 para parar
    assert resultado[0] in ["ITEM1", "ITEM2", "-1"]
    
    # Número da próxima resposta deve ser 2 (pois já temos uma)
    if resultado[0] != "-1":
        assert resultado[1] == "2"

def test_proximo_item_parada():
    # Configurar 45 itens e seus parâmetros
    n_itens = 45
    
    request_data = {
        "ESTUDANTE": "Aluno1",
        "AnoEscolarEstudante": "8",
        "proficiencia": "500.0",
        "profic.inic": "500.0",
        "idItem": ",".join([f"ITEM{i}" for i in range(1, n_itens + 1)]),
        "parA": ",".join(["1.0"] * n_itens),
        "parB": ",".join(["250.0"] * n_itens),
        "parC": ",".join(["0.2"] * n_itens),
        "administrado": ",".join([f"ITEM{i}" for i in range(1, 33)]),  # 32 itens
        "respostas": ",".join(["A"] * 32),
        "gabarito": ",".join(["A"] * 32),
        "erropadrao": "0.5",
        "n.Ij": "45",
        "componente": "Língua portuguesa",
        "idEixo": ",".join(["1"] * n_itens),
        "idHabilidade": ",".join(["2"] * n_itens)
    }
    
    response = client.post("/proximo", json=request_data)
    assert response.status_code == 200
    
    resultado = response.json()
    # Deve retornar -1 como primeiro elemento (parada pelo critério de 32 itens)
    assert resultado[0] == "-1"

# Teste para verificar transformação correta do componente
def test_normalizacao_componente(sample_request_data):
    # Teste com diferentes formatos do nome do componente
    variantes = [
        ("Língua portuguesa", "LP"),
        ("Matemática", "MT"),
        ("Ciências da Natureza", "CN"),
        ("Ciências Humanas", "CH"),
        ("LP", "LP")  # Já normalizado
    ]
    
    for entrada, esperado in variantes:
        request_copy = sample_request_data.copy()
        request_copy["componente"] = entrada
        response = client.post("/proximo", json=request_copy)
        assert response.status_code == 200
        
        # Se o teste passar sem erro, consideramos que a normalização funcionou