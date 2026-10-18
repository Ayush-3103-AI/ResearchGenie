# ResearchGenie API + web UI. No secrets in here: the model provider and any
# API key come from the environment at run time (see docker-compose.yml).
FROM python:3.11-slim

# Tectonic compiles paper.tex to PDF. Pinned static (musl) release, checked by
# sha256. Its TeX bundle is downloaded on the first compile and cached under
# ~/.cache/Tectonic (a volume in docker-compose.yml, so it happens once).
ARG TECTONIC_VERSION=0.17.0
ARG TARGETARCH
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
      amd64) arch=x86_64;  sum=8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7 ;; \
      arm64) arch=aarch64; sum=b10954a95404f3ab2328d2fa59a5ebab8e657f893fab096f98be8db7c0c979b8 ;; \
      *) echo "unsupported arch: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    url="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${arch}-unknown-linux-musl.tar.gz"; \
    python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], '/tmp/tectonic.tgz')" "$url"; \
    echo "${sum}  /tmp/tectonic.tgz" | sha256sum -c -; \
    tar -xzf /tmp/tectonic.tgz -C /usr/local/bin tectonic; \
    rm /tmp/tectonic.tgz; \
    tectonic --version

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements/base.txt && pip install --no-cache-dir .

# Run as an unprivileged user. Its home holds the tectonic cache and
# ~/.researchgenie; outputs/ is where every run folder is written.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache /app/outputs \
    && chown -R app:app /home/app /app/outputs
USER app

EXPOSE 8000
# Run from /app (not site-packages) so the API finds frontend/ and services/
# next to it and writes runs to /app/outputs.
CMD ["uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
