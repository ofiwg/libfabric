#!/usr/bin/env bash

#
# Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
# Copyright (c) 2016, Cray, Inc. All rights reserved.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

trap cleanup_and_exit SIGINT

#
# Default behavior with no args will use sockets provider with loopback
#
declare BIN_PATH
declare PROV="sockets"
declare TEST_TYPE="quick"
declare SERVER="127.0.0.1"
declare CLIENT="127.0.0.1"
declare EXCLUDE
declare GOOD_ADDR="192.168.10.1"
declare -i VERBOSE=0
declare COMPLEX_CFG
declare TIMEOUT_VAL="90"

declare -r c_outp=$(mktemp fabtests.c_outp.XXXXXX)
declare -r s_outp=$(mktemp fabtests.s_outp.XXXXXX)

declare -i skip_count=0
declare -i pass_count=0
declare -i fail_count=0

declare -ri FI_ENODATA=$(python -c 'import errno; print(errno.ENODATA)')
declare -ri FI_ENOSYS=$(python -c 'import errno; print(errno.ENOSYS)')

neg_unit_tests=(
	"dgram g00n13s"
	"rdm g00n13s"
	"msg g00n13s"
)

simple_tests=(
	"cm_data"
	"cq_data"
	"dgram"
	"dgram_waitset"
	"msg"
	"msg_epoll"
	"msg_sockets"
	"poll -t queue"
	"poll -t counter"
	"rdm"
	"rdm_rma_simple"
	"rdm_rma_trigger"
	"shared_ctx"
	"shared_ctx -e msg"
	"shared_ctx -e dgram"
	"rdm_tagged_peek"
	"scalable_ep"
	"cmatose"
	"rdm_shared_av"
)

short_tests=(
	"msg_pingpong -I 5"
	"msg_bw -I 5"
	"rma_bw -e msg -o write -I 5"
	"rma_bw -e msg -o read -I 5"
	"rma_bw -e msg -o writedata -I 5"
	"rma_bw -e rdm -o write -I 5"
	"rma_bw -e rdm -o read -I 5"
	"rma_bw -e rdm -o writedata -I 5"
	"msg_rma -o write -I 5"
	"msg_rma -o read -I 5"
	"msg_rma -o writedata -I 5"
	"msg_stream -I 5"
	"rdm_atomic -I 5 -o all"
	"rdm_cntr_pingpong -I 5"
	"rdm_multi_recv -I 5"
	"rdm_pingpong -I 5"
	"rdm_rma -o write -I 5"
	"rdm_rma -o read -I 5"
	"rdm_rma -o writedata -I 5"
	"rdm_tagged_pingpong -I 5"
	"rdm_tagged_bw -I 5"
	"dgram_pingpong -I 5"
	"rc_pingpong -n 5"
	"rc_pingpong -n 5 -e"
)

standard_tests=(
	"msg_pingpong"
	"msg_pingpong -v"
	"msg_pingpong -P"
	"msg_pingpong -P -v"
	"msg_bw"
	"rma_bw -e msg -o write"
	"rma_bw -e msg -o read"
	"rma_bw -e msg -o writedata"
	"rma_bw -e rdm -o write"
	"rma_bw -e rdm -o read"
	"rma_bw -e rdm -o writedata"
	"msg_rma -o write"
	"msg_rma -o read"
	"msg_rma -o writedata"
	"msg_stream"
	"rdm_atomic -o all -I 1000"
	"rdm_cntr_pingpong"
	"rdm_multi_recv"
	"rdm_pingpong"
	"rdm_pingpong -v"
	"rdm_pingpong -P"
	"rdm_pingpong -P -v"
	"rdm_rma -o write"
	"rdm_rma -o read"
	"rdm_rma -o writedata"
	"rdm_tagged_pingpong"
	"rdm_tagged_bw"
	"dgram_pingpong"
	"dgram_pingpong -v"
	"dgram_pingpong -P"
	"dgram_pingpong -P -v"
	"rc_pingpong"
)

unit_tests=(
	"av_test -d GOOD_ADDR -n 1 -s SERVER_ADDR"
	"dom_test -n 2"
	"eq_test"
	"cq_test"
	"size_left_test"
)

complex_tests=(
	"ubertest"
)

function errcho {
	>&2 echo $*
}

function print_border {
	echo "# --------------------------------------------------------------"
}

function print_results {
	local test_name=$1
	local test_result=$2
	local test_time=$3
	local server_out_file=$4
	local server_cmd=$5
	local client_out_file=$6
	local client_cmd=$7

	if [ $VERBOSE -eq 0 ] ; then
		# print a simple, single-line format that is still valid YAML
		printf "%-50s%10s\n" "$test_exe:" "$test_result"
	else
		# Print a more detailed YAML format that is not a superset of
		# the non-verbose output.  See ofiwg/fabtests#259 for a
		# rationale.
		emit_stdout=0
		case $test_result in
			Pass*)
				[ $VERBOSE -ge 3 ] && emit_stdout=1
				;;
			Notrun)
				[ $VERBOSE -ge 2 ] && emit_stdout=1
				;;
			Fail*)
				[ $VERBOSE -ge 1 ] && emit_stdout=1
				;;
		esac

		printf -- "- name:   %s\n" "$test_exe"
		printf -- "  result: %s\n" "$test_result"
		printf -- "  time:   %s\n" "$test_time"
		if [ $emit_stdout -eq 1 -a "$server_out_file" != "" ] ; then
			if [ "$server_cmd" != "" ] ; then
				printf -- "  server_cmd: %s\n" "$server_cmd"
			fi
			printf -- "  server_stdout: |\n"
			sed -e 's/^/    /' < $server_out_file
		fi
		if [ $emit_stdout -eq 1 -a "$client_out_file" != "" ] ; then
			if [ "$client_cmd" != "" ] ; then
				printf -- "  client_cmd: %s\n" "$client_cmd"
			fi
			printf -- "  client_stdout: |\n"
			sed -e 's/^/    /' < $client_out_file
		fi
	fi
}

function cleanup {
	${CLIENT_CMD} "ps -eo comm,pid | grep '^fi_' | awk '{print \$2}' | xargs kill -9" >& /dev/null
	${SERVER_CMD} "ps -eo comm,pid | grep '^fi_' | awk '{print \$2}' | xargs kill -9" >& /dev/null
	rm -f $c_outp $s_outp
}

function cleanup_and_exit {
	cleanup
	exit 1
}

# compute the duration in seconds between two integer values
# measured since the start of the UNIX epoch and print the result to stdout
function compute_duration {
	local -i s=$1
	local -i e=$2
	echo $(( $2 - $1))
}

function is_excluded {
	for i in $(echo "$EXCLUDE" | tr -s "," " "); do
		if [[ "$i" = "$1" ]]; then
			echo 1
			return
		fi
	done

	echo 0
}

function unit_test {
	local test=$1
	local is_neg=$2
	local ret1=0
	local test_exe=$(echo "fi_${test} -f $PROV" | \
	    sed -e "s/GOOD_ADDR/$GOOD_ADDR/g" -e "s/SERVER_ADDR/${S_INTERFACE}/g")
	local start_time
	local end_time
	local test_time

	local e=$(is_excluded $(echo "fi_${test}" | cut -d " " -f 1))
	if [ $e -eq 1 ]; then
		print_results "$test_exe" "Notrun" "0" "" ""
		skip_count+=1
		return
	fi

	start_time=$(date '+%s')

	cmd="${BIN_PATH}${test_exe}"
	${SERVER_CMD} "$cmd" &> $s_outp &
	p1=$!

	wait $p1
	ret=$?

	end_time=$(date '+%s')
	test_time=$(compute_duration "$start_time" "$end_time")

	if [ $is_neg -eq 1 -a $ret -eq $FI_ENODATA ]; then
		# negative test passed
		ret=0
	elif [ $is_neg -eq 1 ]; then
		# negative test failed
		ret=1
	fi
	if [[ $ret -eq $FI_ENODATA || $ret -eq $FI_ENOSYS ]]; then
		print_results "$test_exe" "Notrun" "$test_time" "$s_outp" "$cmd"
		skip_count+=1
	elif [ $ret -ne 0 ]; then
		print_results "$test_exe" "Fail" "$test_time" "$s_outp" "$cmd"
		if [ $ret -eq 124 ]; then
			cleanup
		fi
		fail_count+=1
	else
		print_results "$test_exe" "Pass" "$test_time" "$s_outp" "$cmd"
		pass_count+=1
	fi
}

function cs_test {
	local test=$1
	local ret1=0
	local ret2=0
	local test_exe="fi_${test} -f ${PROV}"
	local start_time
	local end_time
	local test_time

	local e=$(is_excluded $(echo "fi_${test}" | cut -d " " -f 1))
	if [ $e -eq 1 ]; then
		print_results "$test_exe" "Notrun" "0" "" ""
		skip_count+=1
		return
	fi

	start_time=$(date '+%s')

	s_cmd="${BIN_PATH}${test_exe} -s $S_INTERFACE"
	${SERVER_CMD} "$s_cmd" &> $s_outp &
	p1=$!
	sleep 1

	c_cmd="${BIN_PATH}${test_exe} -s $C_INTERFACE $S_INTERFACE"
	${CLIENT_CMD} "$c_cmd" &> $c_outp &
	p2=$!

	wait $p1
	ret1=$?

	wait $p2
	ret2=$?

	end_time=$(date '+%s')
	test_time=$(compute_duration "$start_time" "$end_time")

	if [[ $ret1 -eq $FI_ENODATA && $ret2 -eq $FI_ENODATA ]] ||
	   [[ $ret1 -eq $FI_ENOSYS && $ret2 -eq $FI_ENOSYS ]]; then
		print_results "$test_exe" "Notrun" "$test_time" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
		skip_count+=1
	elif [ $ret1 -ne 0 -o $ret2 -ne 0 ]; then
		print_results "$test_exe" "Fail" "$test_time" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
		if [ $ret1 -eq 124 -o $ret2 -eq 124 ]; then
			cleanup
		fi
		fail_count+=1
	else
		print_results "$test_exe" "Pass" "$test_time" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
		pass_count+=1
	fi
}

function complex_test {
	local test=$1
	local config=$2
	local test_exe="fi_${test}"
	local ret1=0
	local ret2=0
	local start_time
	local end_time
	local test_time

	local e=$(is_excluded $(echo "fi_${test}" | cut -d " " -f 1))
	if [ $e -eq 1 ]; then
		print_results "$test_exe" "Notrun" "0" "" ""
		skip_count+=1
		return
	fi

	start_time=$(date '+%s')

	s_cmd="${BIN_PATH}${test_exe} -s $S_INTERFACE -x"
	FI_LOG_LEVEL=error ${SERVER_CMD} "$s_cmd" &> $s_outp &
	p1=$!
	sleep 1

	c_cmd="${BIN_PATH}${test_exe} -s $C_INTERFACE -f ${PROV} -t $config $S_INTERFACE"
	FI_LOG_LEVEL=error ${CLIENT_CMD} "$c_cmd" &> $c_outp &
	p2=$!

	wait $p2
	ret2=$?

	wait $p1
	ret1=$?

	end_time=$(date '+%s')
	test_time=$(compute_duration "$start_time" "$end_time")

	# case: config file doesn't exist or invalid option provided
	if [ $ret1 -eq 1 -o $ret2 -eq 1 ]; then
		print_results "$test_exe" "Notrun" "0" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
		cleanup
		skip_count+=1
		return
	# case: test didn't run becasue some error occured
	elif [ $ret1 -ne 0 -o $ret2 -ne 0 ]; then
		printf "%-50s%s\n" "$test_exe:" "Server returns $ret1, client returns $ret2"
		print_results "$test_exe" "Fail [$f_cnt/$total]" "$test_time" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
                cleanup
                fail_count+=1
	else
		local f_cnt=$(cat $c_outp | awk -F': ' '/ENOSYS|ERROR/ {total += $2} END {print total}')
		local s_cnt=$(cat $c_outp | awk -F': ' '/Success/ {total += $2} END {print total}')
		local total=$(cat $c_outp | awk -F': ' '/Success|ENODATA|ENOSYS|ERROR/ {total += $2} END {print total}')
		if [ $f_cnt -eq 0 ]; then
			print_results "$test_exe" "Pass [$s_cnt/$total]" "$test_time" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
			pass_count+=1
		else
			print_results "$test_exe" "Fail [$f_cnt/$total]" "$test_time" "$s_outp" "$s_cmd" "$c_outp" "$c_cmd"
			cleanup
			fail_count+=1
		fi
	fi
}

function main {
	local complex_cfg="quick"

	if [[ $1 == "quick" ]]; then
		local -r tests="unit simple short"
	else
		local -r tests=$(echo $1 | sed 's/all/unit,simple,standard,complex/g' | tr ',' ' ')
		if [[ $1 == "all" ]]; then
			complex_cfg=$1
		fi
	fi

	if [[ -n "$COMPLEX_CFG" ]]; then
		complex_cfg="$COMPLEX_CFG"
	fi

	if [ $VERBOSE -eq 0 ] ; then
		printf "# %-50s%10s\n" "Test" "Result"
		print_border
	fi

	for ts in ${tests}; do
	case ${ts} in
		unit)
			for test in "${unit_tests[@]}"; do
				unit_test "$test" "0"
			done

			for test in "${neg_unit_tests[@]}"; do
				unit_test "$test" "1"
			done
		;;
		simple)
			for test in "${simple_tests[@]}"; do
				cs_test "$test"
			done
		;;
		short)
			for test in "${short_tests[@]}"; do
				cs_test "$test"
			done
		;;
		standard)
			for test in "${standard_tests[@]}"; do
				cs_test "$test"
			done
		;;
		complex)
			for test in "${complex_tests[@]}"; do
				complex_test $test $complex_cfg

			done
		;;
		*)
			errcho "Unknown test set: ${ts}"
			exit 1
		;;
	esac
	done

	total=$(( $pass_count + $fail_count ))

	print_border

	printf "# %-50s%10d\n" "Total Pass" $pass_count
	printf "# %-50s%10d\n" "Total Notrun" $skip_count
	printf "# %-50s%10d\n" "Total Fail" $fail_count

	if [[ "$total" > "0" ]]; then
		printf "# %-50s%10d\n" "Percentage of Pass" $(( $pass_count * 100 / $total ))
	fi

	print_border

	cleanup
	exit $fail_count
}

function usage {
	errcho "Usage:"
	errcho "  $0 [OPTIONS] [provider] [host] [client]"
	errcho
	errcho "Run fabtests using provider between host and client (default"
	errcho "'sockets' provider in loopback-mode).  Report pass/fail/notrun status."
	errcho
	errcho "Options:"
	errcho -e " -g\tgood IP address from <host>'s perspective (default $GOOD_ADDR)"
	errcho -e " -v\tprint output of failing"
	errcho -e " -vv\tprint output of failing/notrun"
	errcho -e " -vvv\tprint output of failing/notrun/passing"
	errcho -e " -t\ttest set(s): all,quick,unit,simple,standard,short,complex (default quick)"
	errcho -e " -e\texclude tests: cq_data,dgram_dgram_waitset,..."
	errcho -e " -p\tpath to test bins (default PATH)"
	errcho -e " -c\tclient interface"
	errcho -e " -s\tserver/host interface"
	errcho -e " -u\tconfigure option for complex tests"
	errcho -e " -T\ttimeout value in seconds"
	exit 1
}

while getopts ":vt:p:g:e:c:s:u:T:" opt; do
case ${opt} in
	t) TEST_TYPE=$OPTARG
	;;
	v) VERBOSE+=1
	;;
	p) BIN_PATH="${OPTARG}/"
	;;
	g) GOOD_ADDR=${OPTARG}
	;;
	e) EXCLUDE=${OPTARG}
	;;
	c) C_INTERFACE=${OPTARG}
	;;
	s) S_INTERFACE=${OPTARG}
	;;
	u) COMPLEX_CFG=${OPTARG}
	;;
	T) TIMEOUT_VAL=${OPTARG}
	;;
	:|\?) usage
	;;
esac

done

# base ssh command
declare bssh="ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=2 -o BatchMode=yes"
if [ -z "$(which timeout 2> /dev/null)" ]; then
	# forego timeout
	declare SERVER_CMD="eval"
	declare CLIENT_CMD="eval"
else
	declare SERVER_CMD="eval timeout ${TIMEOUT_VAL}"
	declare CLIENT_CMD="eval timeout ${TIMEOUT_VAL}"
	bssh="timeout ${TIMEOUT_VAL} ${bssh}"
fi

# shift past options
shift $((OPTIND-1))

if [[ $# -ge 4 ]]; then
	usage
fi

if [[ $# -ge 1 ]]; then
	PROV=$1
fi

if [[ $# -ge 2 ]]; then
	SERVER=$2
	SERVER_CMD="${bssh} ${SERVER}"
fi

if [[ $# -ge 3 ]]; then
	CLIENT=$3
	CLIENT_CMD="${bssh} ${CLIENT}"
fi

[ -z $C_INTERFACE ] && C_INTERFACE=$CLIENT
[ -z $S_INTERFACE ] && S_INTERFACE=$SERVER

main ${TEST_TYPE}
