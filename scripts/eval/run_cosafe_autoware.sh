#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")/../.."

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.7/site-packages/spconv:${LD_LIBRARY_PATH:-}"

export SECONDS_GIVEN_PER_METERS=2.0
export INITIAL_SECONDS_DELAY=15.0

LATENCY=${1:-ideal}
SCENARIO=${2:-Interdrive_no_npc}
CARLA_PORT=2000
CARLA_GPU=0
EVAL_GPU=1
COMPOSE_DIR="docker/autoware"
LOG_DIR="log/cosafe_autoware"
mkdir -p "$LOG_DIR"

if [ "$LATENCY" == "realtime" ]; then
    export REALTIME_MODE=1
else
    export REALTIME_MODE=0
fi

if [ "$LATENCY" == "stop" ]; then
    echo "Stopping everything..."
    pkill -f "cs_leaderboard_evaluator" 2>/dev/null || true
    pkill -f "CarlaUE4" 2>/dev/null || true
    cd "$COMPOSE_DIR" && docker compose down 2>/dev/null || true
    echo "Done."
    exit 0
fi

town07_routes=(
    "r14_town07_ins_sl:2"
    "r15_town07_ins_sl:2"
    "r21_town07_ins_sr:2"
    "r22_town07_ins_sr:2"
)

town05_routes=(
    "r26_town05_ins_chaos:6"
    "r31_town05_ins_oppo:4" "r32_town05_ins_oppo:4"
    "r33_town05_ins_rl:4" "r34_town05_ins_rl:4"
    "r35_town05_ins_crosschange:4" "r36_town05_ins_crosschange:4"
    "r37_town05_ins_chaos:8" "r38_town05_ins_chaos:8"
    "r41_town05_ins_oppo:4" "r42_town05_ins_rl:4"
    "r43_town05_ins_crosschange:4" "r44_town05_ins_chaos:8"
)

echo "Starting CARLA on GPU $CARLA_GPU, port $CARLA_PORT..."
CUDA_VISIBLE_DEVICES=$CARLA_GPU external_paths/carla_root/CarlaUE4.sh \
    --world-port=$CARLA_PORT -prefer-nvidia -RenderOffScreen -nosound \
    > /dev/null 2>&1 &

waited=0
while ! ss -tlnp 2>/dev/null | grep -q ":${CARLA_PORT} "; do
    sleep 2; waited=$((waited + 2))
    if [ $waited -ge 60 ]; then echo "Warning: CARLA not ready after 60s"; break; fi
done
echo "CARLA ready (${waited}s)"

start_containers() {
    local town="$1"
    local max_egos="$2"
    echo ""
    echo "Starting $max_egos bridge+autoware container pairs for $town..."
    cd "$COMPOSE_DIR"

    sed -i "s|AUTOWARE_MAP_PATH: /maps/Town[0-9]*|AUTOWARE_MAP_PATH: /maps/${town}|g" docker-compose.yaml

    local services=""
    for e in $(seq 0 $((max_egos - 1))); do
        services="$services bridge_${e} autoware_${e}"
    done
    docker compose up -d $services

    cd - > /dev/null
    echo "Waiting 90s for Autoware to initialize..."
    sleep 90

    for e in $(seq 0 $((max_egos - 1))); do
        local port=$((8100 + e))
        curl -s "http://localhost:${port}/health" > /dev/null 2>&1 && \
            echo "  Bridge $e OK (port $port)" || \
            echo "  Bridge $e FAILED (port $port)"
    done
}

stop_containers() {
    echo "Stopping containers..."
    cd "$COMPOSE_DIR"
    docker compose down 2>/dev/null || true
    cd - > /dev/null
}

run_batch() {
    local town="$1"
    shift
    local routes=("$@")
    local total=${#routes[@]}

    echo ""
    echo "========================================"
    echo " Batch: $town ($total routes)"
    echo "========================================"

    local done_count=0
    for route_info in "${routes[@]}"; do
        local route_name=$(echo "$route_info" | cut -d: -f1)
        local vehicle_count=$(echo "$route_info" | cut -d: -f2)
        done_count=$((done_count + 1))

        echo ""
        echo "[$done_count/$total] $route_name (${vehicle_count} egos)"

        start_containers "$town" "$vehicle_count"

        CUDA_VISIBLE_DEVICES=$EVAL_GPU \
        bash scripts/eval/eval_driving_cosafe.sh \
            $vehicle_count $CARLA_PORT $route_name \
            autoware_b2d_agent autoware_config autoware $SCENARIO \
            2>&1 | tee -a "$LOG_DIR/${town}_${route_name}.log"

        stop_containers
    done
}

echo ""
echo "============================================"
echo " Autoware CoSafe Sequential Evaluation"
echo " Latency:  $LATENCY"
echo " Scenario: $SCENARIO"
echo " Town07:   ${#town07_routes[@]} routes"
echo " Town05:   ${#town05_routes[@]} routes"
echo " Total:    $(( ${#town07_routes[@]} + ${#town05_routes[@]} )) routes"
echo "============================================"

run_batch "Town05" "${town05_routes[@]}"

echo ""
echo "All batches done. Logs in $LOG_DIR/"
