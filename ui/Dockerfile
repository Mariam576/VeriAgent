# VeriAgent interface image (small: no PyTorch, no database driver)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY ui/requirements.txt ui/requirements.txt
RUN pip install --upgrade pip && pip install -r ui/requirements.txt

COPY .streamlit .streamlit
COPY ui ui

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

# Usage statistics are switched off; the app listens on all interfaces inside
# the container and Docker Compose publishes it on 127.0.0.1 only.
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
