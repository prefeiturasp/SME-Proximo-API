# Etapa 1: build da imagem
FROM python:3.12-slim AS builder

WORKDIR /app

# Copia só os arquivos necessários para instalação das dependências
COPY requirements.txt .

# Instala dependências em um diretório isolado
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Etapa 2: imagem final para produção
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copia os pacotes Python já instalados da imagem anterior
COPY --from=builder /install /usr/local
COPY . .

# Comando para rodar o servidor com Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
