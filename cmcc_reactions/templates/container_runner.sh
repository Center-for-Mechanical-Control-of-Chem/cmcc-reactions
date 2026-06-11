#!/usr/bin/env bash
# Launches the SLURMClient comm server on the host (bound to a Unix socket),
# creates an `sbatch` alias/shim that routes calls back to that server, then
# runs a the container through `singularity` with the socket + shim bound in so
# that `sbatch` inside the container is transparently serviced on the host.
# Everything is torn down once the singularity process exits.
#
# Usage:
#   ./container_runner.sh <singularity_image> [args-to-inner-script...]
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIF_IMAGE="`CONTAINER_PATH`"
shift || true

# Directory holding this script and the python helpers (SLURMServer_launcher.py,
# sbatch_client.py, NodeCommServer.py).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Per-run scratch dir for the socket, shim, and the inner job script.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/slurmshim.XXXXXX")"
SOCKET_FILE="${WORKDIR}/slurm.sock"
ALIAS_DIR="${WORKDIR}/bin"            # prepended to PATH inside the container
ALIAS_SCRIPT="${ALIAS_DIR}/sbatch"    # the fake sbatch

# Where everything lands inside the container.
C_WORKDIR="/opt/slurmshim"
C_SOCKET_FILE="${C_WORKDIR}/slurm.sock"
C_ALIAS_DIR="${C_WORKDIR}/bin"

SERVER_PID=""

# ---------------------------------------------------------------------------
# Cleanup: always runs on exit (normal, error, or signal)
# ---------------------------------------------------------------------------
cleanup() {
    local rc=$?
    echo "[cleanup] tearing down (exit ${rc})..." >&2
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -rf "${WORKDIR}" 2>/dev/null || true
    return $rc
}
trap cleanup EXIT INT TERM

mkdir -p "${ALIAS_DIR}"

# ---------------------------------------------------------------------------
# 1. Launch the SLURMClient server on the host, bound to the Unix socket.
# ---------------------------------------------------------------------------
echo "[launch] starting SLURMClient server on ${SOCKET_FILE}" >&2
( cd "${SCRIPT_DIR}" && exec "${PYTHON_BIN}" -m SLURMClient --socket="${SOCKET_FILE}" ) &
SERVER_PID=$!

# Wait for the socket file to appear (server is ready) before continuing.
for _ in $(seq 1 50); do
    [[ -S "${SOCKET_FILE}" ]] && break
    # bail early if the server died during startup
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[error] server exited during startup" >&2
        exit 1
    fi
    sleep 0.1
done
if [[ ! -S "${SOCKET_FILE}" ]]; then
    echo "[error] timed out waiting for socket ${SOCKET_FILE}" >&2
    exit 1
fi
#echo "[launch] server ready (pid ${SERVER_PID})" >&2

# ---------------------------------------------------------------------------
# 2. Create the `sbatch` alias script that routes to the SLURMClient.
#    (A real alias isn't inherited by subprocesses, so we use a PATH shim
#     script — the effective "alias" for the fake sbatch.)
# ---------------------------------------------------------------------------
cat > "${ALIAS_SCRIPT}" <<EOF
#!/usr/bin/env bash
# fake sbatch -> routes the call to the host SLURMClient over the bound socket
exec "${PYTHON_BIN}" -m SLURMClient "\$@"
EOF
chmod +x "${ALIAS_SCRIPT}"

# ---------------------------------------------------------------------------
# 4. Run the long-running singularity process with the socket + shim bound in.
#    PATH is prepended with the shim dir so `sbatch` resolves to our fake one.
# ---------------------------------------------------------------------------
singularity run \
    --bind "${WORKDIR}:${C_WORKDIR}" \
    --bind "${SCRIPT_DIR}:${C_WORKDIR}/scripts:ro" \
    --env "PATH=${C_ALIAS_DIR}:\$PATH" \
    --env "SLURM_SOCKET_FILE=${C_SOCKET_FILE}" \
    --env "CONTAINER_ARGS=${CONTAINER_ARGS}" \
    --env "PYTHONPATH=${C_WORKDIR}/scripts:${C_WORKDIR}:\$PYTHONPATH" \
    $CONTAINER_ARGS "${SIF_IMAGE}" "$@"

# trap on EXIT performs server kill + workdir removal