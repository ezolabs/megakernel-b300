#!/bin/bash
# bench178.sh (#178) -- the #84 speed battery (prod-merge-20261030/raw/gate/box/gate.sh, STEPS=arm), box side,
# against 127.0.0.1:8002 only, on a box whose router rows are held out (maintenance window), so no prod traffic is
# in a measured rep (bench.py --max-live 0 also waits for #running-req 0 before each rep).
# Same commands as #84's arm: warm (unmeasured), think-ramp C1/C2/C4 x8 at seed 4242+C, max_tokens 300, then C8
# (new here, same form), then coldfill x8 at 60k (seed 777, offsets 47..54). The ~70k/~100k descriptive fills are
# dropped (out-of-rotation budget). Usage: L=<label> bash bench178.sh   (cwd /workspace/mk29/speed178)
ulimit -c 0
L=${L:?label}; T=/workspace/box-a-mk2; B=$T/qa/speed/bench.py; OFF=46
mkdir -p /workspace/mk29/speed178/$L && cd /workspace/mk29/speed178/$L || exit 1
ts(){ date -u +%FT%TZ; }
echo "[$(ts)] bench178 $L start"
nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv > gpu-before-$L.csv
nvidia-smi dmon -s pucv -d 1 > dmon-$L.txt 2>&1 & DM=$!
python3 $B think-ramp --on-box --max-live 0 --reps 1 --levels 1,4,16 --max-tokens 300 --seed 999 --arm warm > warm-$L.txt 2>&1
python3 $B coldfill --on-box --max-live 0 --arm warm --seed 777 --seed-offset $((OFF+0)) >> warm-$L.txt 2>&1
echo "[$(ts)] warm done"
sleep 3; : > acc-$L.txt
for C in 1 2 4 8; do
 s=$(date -u +%T)
 python3 $B think-ramp --on-box --max-live 0 --arm $L --seed $((4242+C)) --levels $C --reps 8 --max-tokens 300 --out think-$L.jsonl >> think-$L.stdout 2>&1
 e=$(date -u +%T); echo "C$C $(python3 $T/acc.py $s $e)" >> acc-$L.txt
 echo "[$(ts)] think C$C done"
done
for i in 1 2 3 4 5 6 7 8; do
 python3 $B coldfill --on-box --max-live 0 --arm $L --seed 777 --seed-offset $((OFF+i)) --out coldfill-$L.jsonl >> coldfill-$L.stdout 2>&1; sleep 3
done
echo "[$(ts)] coldfill done"
kill $DM 2>/dev/null
nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv > gpu-after-$L.csv
cat acc-$L.txt
echo "[$(ts)] BENCH-DONE $L"
