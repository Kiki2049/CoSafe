#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")/../.."  # CoLMDriver/
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.7/site-packages/spconv:${LD_LIBRARY_PATH}"

METHOD=${1:-tcp}
LATENCY=${2:-ideal}
SCENARIO=${3:-Interdrive_no_npc}

LOG_DIR="log/cosafe_${METHOD}"
mkdir -p "$LOG_DIR"

if [ "$METHOD" == "stop" ]; then
    echo "Stopping all CARLA and eval processes..."
    pkill -f "eval_cosafe.sh" 2>/dev/null || true
    pkill -f "cs_leaderboard_evaluator" 2>/dev/null || true
    pkill -f "CarlaUE4.sh" 2>/dev/null || true
    pkill -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
    echo "Done."
    exit 0
fi

CARLA_BIN="external_paths/carla_root/CarlaUE4.sh"
NUM_GROUPS=3
CARLA_GPU=0

ports=(2000 2010 2020)
eval_gpus=(1 2 3)

routes=(
    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
    "17,18,19,20,21,22,24,25,26,27,28,29,30,31,23"
    "32,33,34,35,36,37,38,39,40,41,42,43,44,45,46"
)


if ! [ -x "$CARLA_BIN" ]; then
    echo "Error: CARLA binary not found at $CARLA_BIN"
    exit 1
fi

for port in "${ports[@]}"; do
    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "Error: Port $port already in use. Run '$0 stop' first."
        exit 1
    fi
done

echo "============================================"
echo " CoSafe Parallel Evaluation"
echo " Method:   $METHOD"
echo " Latency:  $LATENCY"
echo " Scenario: $SCENARIO"
echo " Groups:   $NUM_GROUPS (CARLA on GPU $CARLA_GPU, eval on GPU ${eval_gpus[*]})"
echo " Logs:     $LOG_DIR/"
echo "============================================"

echo ""
echo "[1/3] Starting $NUM_GROUPS CARLA instances on GPU $CARLA_GPU..."
for i in $(seq 0 $((NUM_GROUPS - 1))); do
    echo "  CARLA $((i+1)): GPU $CARLA_GPU, port ${ports[$i]}"
    CUDA_VISIBLE_DEVICES=$CARLA_GPU "$CARLA_BIN" \
        --world-port=${ports[$i]} -prefer-nvidia -RenderOffScreen -nosound \
        > /dev/null 2>&1 &
done

echo ""
echo "[2/3] Waiting for CARLA servers to be ready..."
MAX_WAIT=60
for i in $(seq 0 $((NUM_GROUPS - 1))); do
    port=${ports[$i]}
    waited=0
    while ! ss -tlnp 2>/dev/null | grep -q ":${port} "; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $MAX_WAIT ]; then
            echo "  Warning: CARLA on port $port not ready after ${MAX_WAIT}s, proceeding anyway..."
            break
        fi
    done
    echo "  Port $port ready (${waited}s)"
done

echo ""
echo "[3/3] Starting $NUM_GROUPS evaluation processes..."
EVAL_PIDS=()
for i in $(seq 0 $((NUM_GROUPS - 1))); do
    echo "  Group $((i+1)): GPU ${eval_gpus[$i]}, port ${ports[$i]}, routes [${routes[$i]}]"
    nohup bash scripts/eval/eval_cosafe.sh \
        ${eval_gpus[$i]} ${ports[$i]} "$METHOD" "$LATENCY" "$SCENARIO" "${routes[$i]}" \
        > "$LOG_DIR/eval_g$((i+1)).log" 2>&1 &
    EVAL_PIDS+=($!)
done

echo ""
echo "All groups launched. Monitor with:"
echo "  tail -f $LOG_DIR/eval_g*.log"
echo "  nvidia-smi"
echo "  htop"
echo ""
echo "Stop all with:"
echo "  bash scripts/eval/run_parallel.sh stop"
echo ""

echo "Waiting for all evaluations to finish..."
FAILED=0
for i in $(seq 0 $((NUM_GROUPS - 1))); do
    pid=${EVAL_PIDS[$i]}
    if wait "$pid" 2>/dev/null; then
        echo "  Group $((i+1)) finished successfully."
    else
        echo "  Group $((i+1)) exited with error (PID $pid)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All $NUM_GROUPS groups completed successfully."
else
    echo "$FAILED/$NUM_GROUPS groups had errors. Check logs in $LOG_DIR/"
fi
