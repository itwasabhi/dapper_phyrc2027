FROM phyrc-2027:isaac-6.0.1
USER root
COPY requirements-prod.txt constraints.txt /tmp/
# torch lives in Isaac's prebundle; expose it so pip sees it as installed.
ENV PYTHONPATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:/opt/isaac-compat
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    -r /tmp/requirements-prod.txt -c /tmp/constraints.txt
USER isaac-sim
