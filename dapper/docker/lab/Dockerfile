FROM dapper-dev
USER root
# v3.0.0-beta2.patch1: the Isaac Lab 3.0 line that supports Isaac Sim 6.0.x.
ARG ISAACLAB_REF=ffff603eafc6b74264a5261cc0183d6a65390d78
ADD https://github.com/isaac-sim/IsaacLab.git#${ISAACLAB_REF} /opt/IsaacLab
COPY requirements-lab.txt constraints.txt /tmp/
# --no-deps: Isaac Lab pins torch and other packages that must stay as in the base image.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements-lab.txt -c /tmp/constraints.txt \
    && /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps -c /tmp/constraints.txt \
       -e /opt/IsaacLab/source/isaaclab -e /opt/IsaacLab/source/isaaclab_physx \
       -e /opt/IsaacLab/source/isaaclab_assets \
    && chown -R isaac-sim:isaac-sim /opt/IsaacLab
USER isaac-sim
