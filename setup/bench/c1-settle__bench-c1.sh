#!/bin/bash
# bench-c1.sh (#405) -- bench178.sh's warm and its C1 think-ramp cell, with n=32 requests instead of 8, nothing else.
# Box side, 127.0.0.1:8002 only, router rows held out (hand window); bench.py --max-live 0 waits for #running-req 0
# before each rep. Same settings as bench178's C1: think-ramp --levels 1 --seed 4243 --max-tokens 300; warm is
# bench178's warm unchanged. At C1 each rep is one request, so each think-$L.jsonl row is one request's decode
# (dec_med). Usage: L=<label> bash bench-c1.sh   (outputs /workspace/mk29/c1settle/<L>/)
ulimit -c 0
L=${L:?label}; T=/workspace/box-a-mk2; B=$T/qa/speed/bench.py; OFF=46; N=${N:-32}
mkdir -p /workspace/mk29/c1settle/$L && cd /workspace/mk29/c1settle/$L || exit 1
ts(){ date -u +%FT%TZ; }
echo "[$(ts)] bench-c1 $L start"
nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv > gpu-before-$L.csv
nvidia-smi dmon -s pucv -d 1 > dmon-$L.txt 2>&1 & DM=$!
python3 $B think-ramp --on-box --max-live 0 --reps 1 --levels 1,4,16 --max-tokens 300 --seed 999 --arm warm > warm-$L.txt 2>&1
python3 $B coldfill --on-box --max-live 0 --arm warm --seed 777 --seed-offset $((OFF+0)) >> warm-$L.txt 2>&1
echo "[$(ts)] warm done"
sleep 3; : > acc-$L.txt
C=1
s=$(date -u +%T)
python3 $B think-ramp --on-box --max-live 0 --arm $L --seed $((4242+C)) --levels $C --reps $N --max-tokens 300 --out think-$L.jsonl >> think-$L.stdout 2>&1
e=$(date -u +%T); echo "C$C $(python3 $T/acc.py $s $e)" >> acc-$L.txt
echo "[$(ts)] think C$C done"
kill $DM 2>/dev/null
nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv > gpu-after-$L.csv
cat acc-$L.txt
echo "[$(ts)] BENCH-DONE $L"
