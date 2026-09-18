FROM python:3.11-slim

LABEL maintainer="Ahmed Hassan <developer@a2zsoc.com>"
LABEL description="Hyper-Agent OS & Swarm Substrate"

WORKDIR /app

# Copy package specifications
COPY pyproject.toml README.md ./

# Install package
COPY hyper_agent_os ./hyper_agent_os
RUN pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["hyper-os"]
CMD ["all"]
