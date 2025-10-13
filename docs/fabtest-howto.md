#测试:

  单机测试（服务器 + 客户端）：
  export FI_PROVIDER=udp
  ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" &
  sleep 2
  ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" 127.0.0.1

  或者一行版本：
  export FI_PROVIDER=udp && ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" & sleep 2 &&
  ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" 127.0.0.1

  关键变化：
  1. Provider 从 "rxd" 改为 "udp;ofi_rxd" （rxd 是 ofi_rxd 的简称，但必须指定底层的 udp provider）
  2. 路径从 /fabtests/bin/ 改为 /benchmarks/
  3. 需要同时运行服务器端和客户端进程

  如果要在两台机器上测试：
  - 服务器： export FI_PROVIDER=udp && ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd"
  - 客户端： export FI_PROVIDER=udp && ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" <server_ip>


#确认正在使用的是 rxd provider：

  1. 使用 -v 详细输出选项

  export FI_PROVIDER=udp && ./benchmarks/fi_rdm_pingpong -v -p "udp;ofi_rxd" & sleep 2 &&
  ./benchmarks/fi_rdm_pingpong -v -p "udp;ofi_rxd" 127.0.0.1

  2. 使用 FI_LOG_LEVEL 环境变量

  export FI_PROVIDER=udp
  export FI_LOG_LEVEL=info
  ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd"

  这会显示 libfabric 的日志信息，包括使用的 provider。

  3. 使用 fi_info 命令验证

  export FI_PROVIDER=udp
  fi_info -p "udp;ofi_rxd"

  输出会显示：
  provider: udp;ofi_rxd
  ...
  protocol: FI_PROTO_RXD

  4. 检查程序输出的 provider 信息

  某些测试程序会在开始时打印使用的 provider，可以查找类似以下的输出：
  ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" 2>&1 | grep -i "provider\|fabric"

  5. 使用 strace 追踪

  strace -e open,openat ./benchmarks/fi_rdm_pingpong -p "udp;ofi_rxd" 2>&1 | grep -i rxd

  这会显示是否加载了 rxd 相关的库文件。

  最简单的验证方法是查看 fi_info 的输出中 protocol 字段是否为 FI_PROTO_RXD：

  export FI_PROVIDER=udp && fi_info -p "udp;ofi_rxd" | grep -E "provider:|protocol:"

  预期输出：
  provider: udp;ofi_rxd
  protocol: FI_PROTO_RXD
