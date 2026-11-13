#!/usr/bin/env bash
# Set up this host as a self-hosted GitHub Actions GPU runner.
#
# Run as root (sudo). Get a registration token from
#   Settings -> Actions -> Runners -> New self-hosted runner
# (tokens expire after one hour).
#
#   sudo GH_REPO_URL=https://github.com/yncxcw/dl-inference \
#        RUNNER_TOKEN=<token> tools/ci/setup_gpu_runner.sh
#
# Optional environment:
#   RUNNER_USER     service account            (default: gh-runner)
#   RUNNER_HOME     runner install directory   (default: /opt/actions-runner)
#   RUNNER_WORK     job workspace directory    (default: /var/lib/actions-runner-work)
#   RUNNER_LABELS   extra labels               (default: gpu,rtx4090)
#   RUNNER_NAME     runner name                (default: $(hostname)-gpu)
#   RUNNER_VERSION  runner release, e.g. 2.329.0 (default: latest)
set -euo pipefail

: "${GH_REPO_URL:?set GH_REPO_URL, e.g. https://github.com/yncxcw/dl-inference}"
: "${RUNNER_TOKEN:?set RUNNER_TOKEN from the GitHub new-runner page}"
RUNNER_USER="${RUNNER_USER:-gh-runner}"
RUNNER_HOME="${RUNNER_HOME:-/opt/actions-runner}"
RUNNER_WORK="${RUNNER_WORK:-/var/lib/actions-runner-work}"
RUNNER_LABELS="${RUNNER_LABELS:-gpu,rtx4090}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-gpu}"
RUNNER_VERSION="${RUNNER_VERSION:-}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: run as root (sudo)" >&2
  exit 1
fi

for tool in docker nvidia-ctk nvidia-smi curl tar; do
  command -v "${tool}" >/dev/null || { echo "error: ${tool} not found" >&2; exit 1; }
done

echo "==> Service account: ${RUNNER_USER}"
if ! id "${RUNNER_USER}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${RUNNER_USER}"
fi
# Membership in the docker group is root-equivalent on this host. Only run
# workflows you trust (see the fork PR note in the CI docs).
usermod -aG docker "${RUNNER_USER}"

echo "==> NVIDIA container runtime for Docker"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L

echo "==> Runner binaries"
if [[ -z "${RUNNER_VERSION}" ]]; then
  RUNNER_VERSION="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
    | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -n 1)"
fi
[[ -n "${RUNNER_VERSION}" ]] || { echo "error: could not resolve runner version" >&2; exit 1; }
echo "runner version: ${RUNNER_VERSION}"

mkdir -p "${RUNNER_HOME}" "${RUNNER_WORK}"
if [[ ! -x "${RUNNER_HOME}/config.sh" ]]; then
  curl -fsSL -o /tmp/actions-runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  tar -xzf /tmp/actions-runner.tar.gz -C "${RUNNER_HOME}"
  rm -f /tmp/actions-runner.tar.gz
fi
chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_HOME}" "${RUNNER_WORK}"

echo "==> Register runner"
sudo -u "${RUNNER_USER}" bash -c "cd '${RUNNER_HOME}' && ./config.sh \
  --unattended --replace \
  --url '${GH_REPO_URL}' \
  --token '${RUNNER_TOKEN}' \
  --name '${RUNNER_NAME}' \
  --labels '${RUNNER_LABELS}' \
  --work '${RUNNER_WORK}'"

echo "==> systemd service"
(cd "${RUNNER_HOME}" && ./svc.sh install "${RUNNER_USER}" && ./svc.sh start)

echo "==> Weekly Docker cleanup"
cat >/etc/systemd/system/gh-runner-docker-prune.service <<UNIT
[Unit]
Description=Prune unused Docker data left by CI jobs

[Service]
Type=oneshot
ExecStart=/usr/bin/docker system prune --force --filter until=168h
UNIT
cat >/etc/systemd/system/gh-runner-docker-prune.timer <<UNIT
[Unit]
Description=Weekly Docker prune for CI runner

[Timer]
OnCalendar=Sun 03:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now gh-runner-docker-prune.timer

echo
echo "Done. Runner '${RUNNER_NAME}' labels: self-hosted,linux,x64,${RUNNER_LABELS}"
echo "Check:  systemctl list-units 'actions.runner.*'"
echo "Next:   trigger the CI workflow with workflow_dispatch and confirm gpu-test lands here."
