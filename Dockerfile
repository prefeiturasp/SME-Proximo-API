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

# Cria um usuário não-privilegiado para executar a aplicação
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copia os pacotes Python já instalados da imagem anterior
COPY --from=builder /install /usr/local

# Copia apenas os arquivos necessários para a aplicação
COPY main.py .
COPY routers/ ./routers/
COPY services/ ./services/
COPY utils/ ./utils/

# Altera a propriedade dos arquivos para o usuário appuser
RUN chown -R appuser:appuser /app

# Muda para o usuário não-privilegiado
USER appuser

# Comando para rodar o servidor com Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
