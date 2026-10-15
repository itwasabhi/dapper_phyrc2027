FROM dapper-prod
USER root
COPY requirements-dev.txt constraints.txt /tmp/
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    -r /tmp/requirements-dev.txt -c /tmp/constraints.txt
USER isaac-sim
