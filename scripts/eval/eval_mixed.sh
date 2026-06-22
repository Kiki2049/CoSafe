#!/bin/bash

full_route_list=("r1_town05_ins_c:2" "r2_town05_ins_c:2" "r3_town05_ins_c:2" "r4_town06_ins_c:2" "r5_town06_ins_c:2" "r6_town07_ins_c:2" "r7_town05_ins_ss:2" "r8_town05_ins_ss:2" "r9_town06_ins_ss:2" "r10_town07_ins_ss:2" "r11_town05_ins_sl:2" "r12_town06_ins_sl:2" "r13_town05_ins_sl:2" "r14_town07_ins_sl:2" "r15_town07_ins_sl:2" "r16_town05_ins_sl:2" "r17_town05_ins_sr:2" "r18_town05_ins_sr:2" "r19_town05_ins_sr:2" "r20_town06_ins_sr:2" "r21_town07_ins_sr:2" "r22_town07_ins_sr:2" "r23_town05_ins_oppo:3" "r24_town05_ins_rl:3" "r25_town05_ins_crosschange:3" "r26_town05_ins_chaos:6" "r27_town06_hw_merge:3" "r28_town06_hw_c:6" "r29_town06_hw_merge:4" "r30_town06_hw_merge:4" "r31_town05_ins_oppo:4" "r32_town05_ins_oppo:4" "r33_town05_ins_rl:4" "r34_town05_ins_rl:4" "r35_town05_ins_crosschange:4" "r36_town05_ins_crosschange:4" "r37_town05_ins_chaos:8" "r38_town05_ins_chaos:8" "r39_town06_hw_c:8" "r40_town06_hw_c:8" "r41_town05_ins_oppo:4" "r42_town05_ins_rl:4" "r43_town05_ins_crosschange:4" "r44_town05_ins_chaos:8" "r45_town06_hw_merge:4" "r46_town06_hw_c:7")

if [ "$3" == "realtime" ]; then
    export REALTIME_MODE=1
    extra_tag='_realtime'
else
    export REALTIME_MODE=0
    extra_tag=''
fi
echo "REALTIME_MODE set to $REALTIME_MODE"

agent="cs_mixed_agent"
config="mixed_3way"
method_tag="mixed"

if [ -n "$5" ]; then
    IFS=',' read -ra specified_ids <<< "$5"
    route_list=()
    for specified_id in "${specified_ids[@]}"; do
        specified_id=$(echo "$specified_id" | xargs)
        route_name="r${specified_id}"
        found=false
        for full_route in "${full_route_list[@]}"; do
            full_route_name=$(echo "$full_route" | cut -d":" -f1)
            if [[ "$full_route_name" == ${route_name}_* ]]; then
                route_list+=("$full_route")
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo "Debug: No match found for $route_name"
        fi
    done
    if [ ${#route_list[@]} -eq 0 ]; then
        echo "Error: No matching routes found for IDs: $5"
        exit 1
    fi
    echo "Running specified routes: ${route_list[@]}"
else
    route_list=("${full_route_list[@]}")
    echo "Running all routes"
fi

scenario_type=$4
scenario_list=()
if [ "$scenario_type" == "Interdrive_all" ]; then
    scenario_list=("Interdrive_no_npc" "Interdrive_npc")
else
    scenario_list=("$scenario_type")
fi

for scenario in "${scenario_list[@]}"
do
    echo "Running scenario: $scenario"
    for route_info in "${route_list[@]}"
    do
        route_name=$(echo $route_info | cut -d":" -f1)
        vehicle_count=$(echo $route_info | cut -d":" -f2)
        echo "Route Index: $route_name  Vehicle Count: $vehicle_count"
        echo "tag: $method_tag$extra_tag"
        CUDA_VISIBLE_DEVICES=$1 bash scripts/eval/eval_driving.sh \
            $vehicle_count $2 $route_name \
            $agent $config $method_tag$extra_tag $scenario
    done
done
