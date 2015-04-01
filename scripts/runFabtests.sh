#!/bin/bash

if [ $# != 5 ]; then
	echo "Usage: $0 <test_bin_path> <provider_name> <all|quick> <server_addr> <client_addr>"
	exit 1
fi

BIN_PATH=$1
PROV=$2
TEST_TYPE=$3
SERVER=$4
CLIENT=$5

if [ $TEST_TYPE == "quick" ]; then
	declare -r TO=60s	
else
	declare -r TO=120s	
fi

declare -r ssh="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 -o BatchMode=yes"
declare -r tssh="timeout ${TO} ${ssh}"

output=""
pass_count=0
fail_count=0

simple_tests=(
    "cq_data"
    "dgram"
    "dgram_waitset"
    "msg"
    "msg_epoll"
    "poll"
    "rdm"
    "rdm_rma_simple"
    "rdm_shared_ctx"
    "rdm_tagged_search"
    "scalable_ep"
    "cmatose"
)

quick_tests=(
    "msg_pingpong -I 100 -S 1024"
    "msg_rma -o write -I 100 -S 1024"
    "msg_rma -o read -I 100 -S 1024"
    "msg_rma -o writedata -I 100 -S 1024"
    "rdm_atomic -I 100 -S 1024 -o all"
    "rdm_cntr_pingpong -I 100 -S 1024"
    "rdm_inject_pingpong -I 100 -S 128"
    "rdm_multi_recv -I 100 -S 1024"
    "rdm_pingpong -I 100 -S 1024"
    "rdm_rma -o write -I 100 -S 1024"
    "rdm_rma -o read -I 100 -S 1024"
    "rdm_rma -o writedata -I 100 -S 1024"
    "rdm_tagged_pingpong -I 100 -S 1024"
    "ud_pingpong -I 100 -S 1024"
    "rc_pingpong -n 100 -S 1024"
)

all_tests=(
    "msg_pingpong"
    "msg_rma -o write"
    "msg_rma -o read"
    "msg_rma -o writedata"
    "rdm_atomic -o all"
    "rdm_cntr_pingpong"
    "rdm_inject_pingpong"
    "rdm_multi_recv"
    "rdm_pingpong"
    "rdm_rma -o write"
    "rdm_rma -o read"
    "rdm_rma -o writedata"
    "rdm_tagged_pingpong"
    "ud_pingpong"
    "rc_pingpong"
)

unit_tests=(
    "av_test -d 192.168.10.1 -n 1"
    "dom_test -n 2"
    "eq_test"	
    "size_left_test")

function print_border 
{
	echo "--------------------------------------------------------------"
}

function cleanup_and_exit {
	cleanup
	exit 1
}

function cleanup {
        $ssh ${CLIENT} "ps aux | grep fi_ | grep -v grep | awk '{print \$2}' | xargs -r kill -9" > /dev/null
        $ssh ${SERVER} "ps aux | grep fi_ | grep -v grep | awk '{print \$2}' | xargs -r kill -9" > /dev/null
}

function run_test {
	local type=$1
	local test=$2
	local ret1=0
	local ret2=0
	local test_exe="fi_${test} -f $PROV"

	if [ "${type}" = 'unit' ]; then
		echo "Running test $test_exe"
		(set -x; $tssh $SERVER "${BIN_PATH}/fi_${test} -f $PROV") &
		
		p1=$!

		wait $p1
		ret1=$?

	elif [ "${type}" = 'client-server' ]; then
		echo "Running test $test_exe"
		(set -x; $tssh $SERVER "${BIN_PATH}/fi_${test} -f $PROV -s $SERVER") &
		p1=$!
		sleep 1s
		(set -x; $tssh $CLIENT "${BIN_PATH}/fi_${test} $SERVER -f $PROV") &
		p2=$!

		wait $p1
		ret1=$?

		wait $p2
		ret2=$?
	fi

	if [ $ret1 != 0 -o $ret1 != 0 ]; then
		if [ $ret1 == 124 -o $ret2 == 124 ]; then
			echo "Test timed out."
			cleanup
		fi
		printf -v output "%s%-50s%10s\n" "$output" "$test_exe:" "Fail"
		fail_count=$((fail_count + 1))
	else
		printf -v output "%s%-50s%10s\n" "$output" "$test_exe:" "Pass"
		pass_count=$((pass_count + 1))
	fi
	echo ""
}

trap cleanup_and_exit SIGINT

#unit tests
testset=("${unit_tests[@]}")  

for test in "${testset[@]}"; do
	run_test "unit" "$test"
done

#simple tests
testset=("${simple_tests[@]}")  

for test in "${testset[@]}"; do
	run_test "client-server" "$test"
done

#iterative tests
if [ $TEST_TYPE == "all" ]; then
	testset=("${all_tests[@]}")  
else
	testset=("${quick_tests[@]}")  
fi

for test in "${testset[@]}"; do
	run_test "client-server" "$test"
done


total=$(( $pass_count + $fail_count ))

printf "\n%-50s%10s\n" "Test" "Result"
print_border

printf "$output"
print_border

printf "%-50s%10d\n" "Total Pass" $pass_count
printf "%-50s%10d\n" "Total Fail" $fail_count
printf "%-50s%10.2f\n" "Percentage of Pass" `echo "scale=2; $pass_count * 100 / $total" | bc`
print_border

if (( $fail_count )); then
	exit 1
fi
