FROM dapper-prod
USER root
COPY requirements-dev.txt constraints.txt /tmp/
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    -r /tmp/requirements-dev.txt -c /tmp/constraints.txt
# Models load from /assets only; dapper/scripts/download_models.py unsets these.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER isaac-sim
