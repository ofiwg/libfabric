# RXD 硬件卸载架构设计

## 1. 总体原则

### 1.1 划分标准

**硬件实现（数据平面）：**
- 性能关键路径上的操作
- 需要低延迟响应的功能
- 可以用固定逻辑实现的功能
- 频繁执行的操作

**软件保留（控制平面）：**
- 需要灵活性的功能
- 复杂的状态机和策略决策
- 与系统集成相关的功能
- 异常和错误处理

## 2. 功能分工详细分析

### 2.1 应该在硬件实现的功能

#### 2.1.1 数据包处理（关键路径）

**✓ 数据包头部解析和验证**
```
硬件职责：
- 解析 rxd_base_hdr（version, type, flags, peer, seq_no）
- 验证协议版本
- 提取序列号和对端 ID
- 解析可选头部（SAR、RMA、Tag、Atom）

理由：每个数据包都需要，延迟敏感
实现：硬件状态机，流水线处理
```

**✓ 数据包头部生成**
```
硬件职责：
- 根据操作类型构造完整的数据包头部
- 自动填充序列号、时间戳
- 添加校验和/CRC

理由：减少 CPU 开销，提高发送吞吐量
实现：数据包构造引擎
```

**✓ 校验和/CRC 计算**
```
硬件职责：
- 对数据包进行校验和计算
- 接收时验证校验和

理由：计算密集型操作，硬件加速效果显著
实现：专用校验和计算单元
```

#### 2.1.2 可靠性机制（核心功能）

**✓ 序列号管理**
```
硬件职责：
- 维护每个对端的发送序列号（tx_seq_no）
- 维护每个对端的接收序列号（rx_seq_no）
- 自动递增发送序列号
- 验证接收数据包的序列号

理由：简单的计数操作，每个包都需要
实现：每对端一个硬件计数器
硬件结构：
  struct hw_peer_seq {
      uint64_t tx_seq_no;
      uint64_t rx_seq_no;
      uint64_t rx_expected;   // 期望接收的序列号
  };
```

**✓ 自动 ACK 生成**
```
硬件职责：
- 接收到数据包后自动生成 ACK
- 批量 ACK 合并（一个 ACK 确认多个连续包）
- ACK 延迟控制（减少 ACK 风暴）

理由：减少 ACK 往返延迟，提高效率
实现：ACK 生成引擎 + 延迟定时器
配置参数：
  - ack_delay: ACK 延迟时间
  - ack_batch: 批量 ACK 大小
```

**✓ 快速重传检测**
```
硬件职责：
- 检测重复 ACK（表示可能丢包）
- 触发快速重传（无需等待超时）
- 维护重传位图

理由：快速恢复，减少延迟
实现：重复 ACK 计数器 + 位图
```

**✓ 硬件定时器和超时检测**
```
硬件职责：
- 为每个未确认的包维护超时定时器
- 超时时自动触发重传
- 实现指数退避算法

理由：精确定时，减少 CPU 轮询
实现：硬件定时器轮 + 超时队列
硬件结构：
  struct hw_timer_entry {
      uint32_t peer_id;
      uint64_t seq_no;
      uint64_t timeout_us;    // 绝对超时时间
      uint8_t  retry_cnt;
  };
```

**✓ 重传逻辑**
```
硬件职责：
- 缓存未确认的数据包
- 超时或快速重传时自动重发
- 更新重传计数
- 达到最大重传次数时通知软件

理由：减少重传延迟，释放 CPU
实现：硬件重传缓冲区 + 重传引擎
限制：最大未确认窗口大小（如 256）
```

#### 2.1.3 流量控制

**✓ 窗口管理**
```
硬件职责：
- 维护每对端的发送窗口（tx_window_start, tx_window_end）
- 维护每对端的接收窗口（rx_window_start, rx_window_end）
- 自动检查序列号是否在窗口内
- 窗口满时暂停发送

理由：简单的范围检查，高频操作
实现：每对端的窗口寄存器
硬件结构：
  struct hw_flow_control {
      uint64_t tx_window_start;
      uint64_t tx_window_end;
      uint64_t rx_window_start;
      uint64_t rx_window_end;
      uint32_t unacked_cnt;
      uint32_t max_unacked;
  };
```

**✓ 拥塞控制（可选）**
```
硬件职责：
- 根据 RTT 和丢包率调整窗口大小
- 实现简单的拥塞避免算法（如 AIMD）

理由：动态适应网络状况
实现：拥塞控制状态机
```

#### 2.1.4 分段与重组（SAR）

**✓ 发送端分段**
```
硬件职责：
- 根据 MTU 大小自动分段大消息
- 为每个分段生成 seg_no
- 在第一个分段添加 SAR 头部

理由：减少软件开销，提高大消息吞吐量
实现：分段引擎 + scatter-gather DMA
硬件接口：
  struct hw_sar_tx_desc {
      uint64_t buf_addr;
      uint64_t total_size;
      uint32_t mtu_size;
      uint32_t tx_id;
      // 硬件自动分段并发送
  };
```

**✓ 接收端重组**
```
硬件职责：
- 根据 tx_id 和 seg_no 重组分段
- 将分段数据写入正确的偏移位置
- 所有分段到达后生成完成事件

理由：减少内存拷贝，利用硬件 DMA
实现：重组缓冲区 + 分段位图
硬件结构：
  struct hw_sar_rx_ctx {
      uint32_t tx_id;
      uint64_t total_size;
      uint64_t received_size;
      uint64_t num_segs;
      uint64_t seg_bitmap[4];  // 位图标记已收到的分段
      uint64_t buf_addr;        // 目标缓冲区地址
  };
```

#### 2.1.5 数据移动

**✓ DMA 引擎**
```
硬件职责：
- 发送：从用户缓冲区 DMA 到网络
- 接收：从网络 DMA 到用户缓冲区
- Scatter-Gather DMA 支持 IOV

理由：零拷贝，减少 CPU 参与
实现：高性能 DMA 引擎
```

**✓ 数据包重排序**
```
硬件职责：
- 缓存乱序到达的数据包
- 按序列号排序后交付软件
- 维护接收窗口内的乱序缓冲区

理由：减少软件处理复杂度
实现：硬件重排序缓冲区（ROB）
限制：缓冲区大小（如接收窗口大小）
```

### 2.2 必须保留在软件的功能

#### 2.2.1 控制平面

**✗ Provider 初始化**
```
软件职责：
- fi_getinfo() 处理
- 能力协商
- 资源分配策略
- 与 libfabric 框架集成

理由：需要灵活性，不是性能关键路径
```

**✗ 对端发现和地址解析**
```
软件职责：
- RTS/CTS 握手协商
- 地址向量（AV）管理
- 对端名称到地址的映射
- 对端连接状态管理（IDLE -> RTS_SENT -> CONNECTED）

理由：涉及复杂的协商和策略决策
软件结构：
  struct sw_peer_mgmt {
      enum rxd_peer_state state;
      char peer_name[RXD_NAME_LENGTH];
      fi_addr_t fiaddr;
      uint64_t peer_addr;
  };
```

**✗ 动态配置管理**
```
软件职责：
- 通过环境变量配置参数
- 运行时参数调整
- 性能调优

理由：需要灵活性，可能需要复杂的策略
配置项：
  - max_peers
  - retry_timeout (初始值，硬件使用)
  - max_retry (硬件使用)
  - max_unacked (硬件使用)
```

#### 2.2.2 复杂错误处理

**✗ 对端失败检测和恢复**
```
软件职责：
- 检测到达 max_retry 后的处理
- 标记对端失败
- 通知应用程序
- 清理相关资源

理由：涉及复杂的状态清理和应用通知
```

**✗ 资源耗尽处理**
```
软件职责：
- 内存不足处理
- 数据包池耗尽
- 重传缓冲区满

理由：需要系统级的资源管理
```

**✗ 异常数据包处理**
```
软件职责：
- 协议版本不匹配
- 无效的序列号范围
- 未知的对端

理由：异常情况，不是快速路径
```

#### 2.2.3 内存管理

**✗ 用户缓冲区管理**
```
软件职责：
- 内存注册（fi_mr_reg）
- 虚拟地址到物理地址映射
- 内存访问权限检查
- 内存反注册

理由：涉及操作系统的内存管理
与硬件接口：提供物理地址给硬件 DMA
```

**✗ 数据包池管理**
```
软件职责：
- 创建和销毁数据包池
- 监控池使用情况
- 动态调整池大小（可选）

理由：策略性决策
与硬件接口：预分配缓冲区给硬件
```

#### 2.2.4 完成队列管理

**✗ CQ/EQ 事件生成和管理**
```
软件职责：
- CQ/EQ 创建和绑定
- 事件格式转换
- 事件过滤和聚合
- 用户回调处理

理由：与 libfabric API 紧密相关
与硬件接口：硬件写入原始完成事件，软件转换为 libfabric 格式
```

#### 2.2.5 高级功能

**✗ RMA 和原子操作的复杂逻辑**
```
软件职责：
- RMA 权限检查
- 原子操作的类型检查
- Compare-and-swap 的比较逻辑（可选卸载）

理由：硬件可以卸载基本操作，复杂检查由软件完成
```

**✗ 统计和调试**
```
软件职责：
- 性能统计收集
- 日志记录
- 调试信息输出

理由：非性能关键，需要灵活性
与硬件接口：硬件提供计数器，软件读取和展示
```

## 3. 软硬件接口设计

### 3.1 接口架构

```
┌─────────────────────────────────────────────────────┐
│             libfabric API Layer                     │
│  (fi_send, fi_recv, fi_read, fi_write, ...)        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│          RXD Software Provider (控制平面)            │
│  - 对端管理（RTS/CTS）                               │
│  - 内存注册                                          │
│  - 错误处理                                          │
│  - 资源管理                                          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         Hardware Offload Interface                  │
│  - Work Queue (WQ) - 提交请求                        │
│  - Completion Queue (CQ) - 完成通知                  │
│  - Control/Status Registers - 配置和状态             │
│  - Doorbell - 通知硬件                               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│      Hardware Offload Engine (数据平面)             │
│  - 数据包处理                                        │
│  - 序列号管理                                        │
│  - ACK/重传                                          │
│  - SAR                                               │
│  - DMA                                               │
└─────────────────────────────────────────────────────┘
```

### 3.2 队列对（Queue Pair）模型

#### 3.2.1 Work Queue（工作请求队列）

**软件到硬件：提交发送/接收请求**

```c
// 工作请求描述符
struct hw_work_request {
    uint8_t  opcode;           // 操作类型
    uint8_t  flags;            // 控制标志
    uint16_t reserved;

    uint32_t peer_id;          // 对端 ID（硬件维护的索引）
    uint64_t wr_id;            // 工作请求 ID（软件用于匹配）

    // 数据缓冲区（Scatter-Gather List）
    uint32_t num_sge;
    struct {
        uint64_t addr;         // 物理地址
        uint32_t length;
        uint32_t lkey;         // 本地内存密钥
    } sgl[HW_MAX_SGE];

    // 操作特定数据
    union {
        // MSG/TAGGED
        struct {
            uint64_t tag;      // 标记（仅用于 TAGGED）
            uint64_t cq_data;  // 远程 CQ 数据（可选）
        } msg;

        // RMA
        struct {
            uint64_t remote_addr;
            uint32_t rkey;     // 远程内存密钥
        } rma;

        // ATOMIC
        struct {
            uint64_t remote_addr;
            uint32_t rkey;
            uint32_t datatype;
            uint32_t op;
            uint64_t operand;
            uint64_t compare;  // 用于 compare-and-swap
        } atomic;
    };
};

// 操作类型
enum hw_opcode {
    HW_OP_SEND,
    HW_OP_RECV,
    HW_OP_READ,
    HW_OP_WRITE,
    HW_OP_ATOMIC,
    HW_OP_ATOMIC_FETCH,
    HW_OP_ATOMIC_COMPARE,
};
```

**提交工作请求（软件操作）：**
```c
int rxd_hw_post_send(struct rxd_ep *ep, struct hw_work_request *wr)
{
    struct hw_send_queue *sq = &ep->hw_sq;
    uint32_t head = sq->head;

    // 1. 写入工作请求到共享内存队列
    memcpy(&sq->wqe[head], wr, sizeof(*wr));

    // 2. 更新队列头指针
    head = (head + 1) % sq->size;
    sq->head = head;

    // 3. 写入 doorbell 通知硬件（MMIO 写入）
    wmb();  // 内存屏障
    *(volatile uint32_t *)sq->doorbell = head;

    return 0;
}
```

#### 3.2.2 Completion Queue（完成队列）

**硬件到软件：通知操作完成**

```c
// 完成队列条目
struct hw_completion {
    uint64_t wr_id;            // 对应的工作请求 ID

    uint8_t  status;           // 完成状态
    uint8_t  opcode;           // 完成的操作类型
    uint16_t flags;            // 标志位

    uint32_t peer_id;          // 对端 ID
    uint32_t byte_len;         // 传输的字节数

    // 接收完成特定信息
    union {
        struct {
            uint64_t tag;      // 匹配的标记
            uint64_t cq_data;  // 接收到的远程 CQ 数据
        } recv;

        struct {
            uint64_t result;   // 原子操作结果
        } atomic;
    };

    uint64_t timestamp;        // 完成时间戳（用于统计）
};

// 完成状态
enum hw_completion_status {
    HW_COMP_SUCCESS,
    HW_COMP_RETRY_EXCEEDED,    // 达到最大重传次数
    HW_COMP_SEQUENCE_ERROR,    // 序列号错误
    HW_COMP_WINDOW_FULL,       // 窗口满
    HW_COMP_PEER_NOT_READY,    // 对端未就绪
    HW_COMP_ERROR,             // 其他错误
};
```

**轮询完成队列（软件操作）：**
```c
int rxd_hw_poll_cq(struct rxd_ep *ep, struct hw_completion *comp, int count)
{
    struct hw_recv_queue *cq = &ep->hw_cq;
    uint32_t tail = cq->tail;
    int polled = 0;

    while (polled < count) {
        struct hw_completion *cqe = &cq->cqe[tail];

        // 检查所有权位（硬件完成后翻转）
        if (cqe->owner != cq->expected_owner)
            break;

        // 复制完成事件
        memcpy(&comp[polled], cqe, sizeof(*cqe));
        polled++;

        // 更新队列尾指针
        tail = (tail + 1) % cq->size;
        if (tail == 0)
            cq->expected_owner = !cq->expected_owner;
    }

    cq->tail = tail;

    // 通知硬件已处理的完成事件数量（更新 consumer index）
    if (polled > 0)
        *(volatile uint32_t *)cq->consumer_idx_reg = tail;

    return polled;
}
```

### 3.3 控制和状态寄存器

#### 3.3.1 对端上下文表（硬件维护）

```c
// 每个对端的硬件上下文（硬件内部维护）
struct hw_peer_context {
    // 基本信息
    uint32_t peer_id;          // 硬件分配的对端 ID
    uint8_t  state;            // 连接状态
    uint8_t  reserved[3];

    // 序列号
    uint64_t tx_seq_no;        // 发送序列号（硬件自动递增）
    uint64_t rx_seq_no;        // 接收序列号（期望接收）

    // 流量控制
    uint64_t tx_window_start;
    uint64_t tx_window_end;
    uint64_t rx_window_start;
    uint64_t rx_window_end;
    uint32_t unacked_cnt;
    uint32_t max_unacked;

    // 重传控制
    uint32_t retry_timeout_us; // 重传超时（微秒）
    uint32_t max_retry;        // 最大重传次数

    // 统计
    uint64_t tx_pkts;
    uint64_t rx_pkts;
    uint64_t retx_pkts;
    uint64_t ack_sent;
    uint64_t ack_recv;
};
```

**软件配置对端上下文：**
```c
// 软件通过 MMIO 或命令队列配置对端
int rxd_hw_init_peer(struct rxd_ep *ep, uint32_t peer_id,
                     struct hw_peer_config *config)
{
    struct hw_peer_context *ctx = &ep->hw_peer_table[peer_id];

    // 通过 MMIO 写入配置
    ctx->tx_window_start = 0;
    ctx->tx_window_end = config->tx_window_size;
    ctx->rx_window_start = 0;
    ctx->rx_window_end = config->rx_window_size;
    ctx->max_unacked = config->max_unacked;
    ctx->retry_timeout_us = config->retry_timeout_us;
    ctx->max_retry = config->max_retry;
    ctx->state = HW_PEER_ACTIVE;

    return 0;
}
```

#### 3.3.2 全局配置寄存器

```c
// 硬件全局配置（MMIO 寄存器）
struct hw_global_config {
    uint32_t max_mtu_size;     // 最大 MTU
    uint32_t max_inline_size;  // 最大内联大小
    uint32_t ack_delay_us;     // ACK 延迟
    uint32_t ack_batch_size;   // 批量 ACK 大小

    // 能力标志
    uint32_t capabilities;
    #define HW_CAP_AUTO_ACK      (1 << 0)
    #define HW_CAP_AUTO_RETRY    (1 << 1)
    #define HW_CAP_SAR           (1 << 2)
    #define HW_CAP_REORDER       (1 << 3)
    #define HW_CAP_TIMESTAMP     (1 << 4)
};

// 初始化时配置
void rxd_hw_init(struct rxd_ep *ep)
{
    struct hw_global_config *cfg = ep->hw_global_config;

    cfg->max_mtu_size = 4096;
    cfg->max_inline_size = 512;
    cfg->ack_delay_us = 100;      // 100 微秒延迟 ACK
    cfg->ack_batch_size = 8;       // 每 8 个包一个 ACK
    cfg->capabilities = HW_CAP_AUTO_ACK | HW_CAP_AUTO_RETRY |
                        HW_CAP_SAR | HW_CAP_REORDER;
}
```

#### 3.3.3 统计和调试寄存器

```c
// 全局统计（只读，硬件更新）
struct hw_statistics {
    uint64_t total_tx_pkts;
    uint64_t total_rx_pkts;
    uint64_t total_tx_bytes;
    uint64_t total_rx_bytes;
    uint64_t total_retx_pkts;
    uint64_t total_ack_sent;
    uint64_t total_ack_recv;
    uint64_t total_timeouts;
    uint64_t total_seq_errors;
    uint64_t total_window_full;
};

// 软件读取统计
void rxd_hw_get_stats(struct rxd_ep *ep, struct hw_statistics *stats)
{
    volatile struct hw_statistics *hw_stats = ep->hw_stats_reg;
    memcpy(stats, (void *)hw_stats, sizeof(*stats));
}
```

### 3.4 中断和事件通知

**中断模式（可选）：**
```c
// 硬件可以通过 MSI-X 中断通知软件
enum hw_interrupt_type {
    HW_INT_CQ_NOT_EMPTY,       // 完成队列非空
    HW_INT_ERROR,              // 错误事件
    HW_INT_PEER_TIMEOUT,       // 对端超时
};

// 中断处理程序
irqreturn_t rxd_hw_interrupt(int irq, void *dev)
{
    struct rxd_ep *ep = dev;
    uint32_t int_status = *(volatile uint32_t *)ep->hw_int_status;

    if (int_status & HW_INT_CQ_NOT_EMPTY) {
        // 唤醒轮询线程或直接轮询 CQ
        rxd_ep_progress(&ep->util_ep);
    }

    if (int_status & HW_INT_ERROR) {
        // 读取错误状态并处理
        rxd_hw_handle_error(ep);
    }

    // 清除中断
    *(volatile uint32_t *)ep->hw_int_clear = int_status;

    return IRQ_HANDLED;
}
```

### 3.5 内存布局

```
软件分配的共享内存区域：

┌─────────────────────────────────────────────────────┐
│  Send Work Queue (WQE Ring Buffer)                 │
│  - 软件写入，硬件读取                                │
│  - Size: N * sizeof(hw_work_request)                │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Recv Work Queue (RWQ Ring Buffer)                  │
│  - 软件预 post 接收缓冲区                            │
│  - Size: M * sizeof(hw_recv_request)                │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Completion Queue (CQE Ring Buffer)                 │
│  - 硬件写入，软件读取                                │
│  - Size: K * sizeof(hw_completion)                  │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Peer Context Table                                 │
│  - 软件配置，硬件使用                                │
│  - Size: max_peers * sizeof(hw_peer_context)        │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Retransmit Buffer Pool                             │
│  - 硬件缓存未确认的数据包                            │
│  - Size: total_window_size * max_mtu_size           │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Reorder Buffer (per peer)                          │
│  - 硬件缓存乱序到达的数据包                          │
│  - Size: per_peer_window_size * max_mtu_size        │
└─────────────────────────────────────────────────────┘

MMIO 寄存器空间：
┌─────────────────────────────────────────────────────┐
│  Doorbell Registers (Write-only)                    │
│  - sq_doorbell: 通知发送队列有新请求                 │
│  - rq_doorbell: 通知接收队列有新缓冲区               │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Control Registers (Read-write)                     │
│  - global_config                                     │
│  - capabilities                                      │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Status Registers (Read-only)                       │
│  - statistics                                        │
│  - error_status                                      │
│  - interrupt_status                                  │
└─────────────────────────────────────────────────────┘
```

## 4. 关键操作流程

### 4.1 发送流程（软硬件协同）

```
软件（RXD Provider）                  硬件（Offload Engine）

1. fi_send() 调用
   ↓
2. 检查对端状态（软件维护）
   if (peer->state != CONNECTED)
       发起 RTS/CTS 握手（软件）
   ↓
3. 内存注册检查
   ↓
4. 构造 hw_work_request
   - peer_id
   - 物理地址
   - 长度
   ↓
5. 提交到 Send Queue
   ↓
6. 写 doorbell 通知硬件  ──────────→  7. 从 SQ 读取 WR
   ↓                                     ↓
   返回给应用                           8. 检查流量控制窗口
                                        if (tx_seq_no >= tx_window_end)
                                            暂停发送
                                        ↓
                                       9. 分配序列号
                                          tx_seq_no++
                                        ↓
                                       10. 判断是否需要分段
                                           if (size > MTU)
                                               分段发送（SAR）
                                        ↓
                                       11. 构造数据包头部
                                           - base_hdr
                                           - SAR/Tag/RMA 头部
                                        ↓
                                       12. DMA 读取用户数据
                                        ↓
                                       13. 发送到网络
                                        ↓
                                       14. 缓存到重传缓冲区
                                        ↓
                                       15. 设置超时定时器
                                        ↓
                                       16. 生成发送完成事件
                                           写入 CQ ────────→  17. 轮询 CQ
                                                                  ↓
                                                              18. 生成 libfabric 完成事件
                                                                  给用户 CQ
```

### 4.2 接收流程

```
硬件                                  软件

1. 从网络接收数据包
   ↓
2. 解析数据包头部
   - type, seq_no, peer_id
   ↓
3. 验证序列号
   if (seq_no == rx_seq_no)
       按序接收
   else if (seq_no in window)
       缓存到重排序缓冲区
   else
       丢弃
   ↓
4. 自动发送 ACK
   ↓
5. 匹配接收请求（从 RQ）
   或者放入意外消息队列
   ↓
6. DMA 写入用户缓冲区
   ↓
7. 更新 rx_seq_no
   ↓
8. 检查重排序缓冲区
   处理后续连续的包
   ↓
9. 生成接收完成事件  ────────→  10. 轮询 CQ
   写入 CQ                           ↓
                                11. 对于意外消息
                                    软件管理意外队列
                                    ↓
                                12. 匹配到 fi_recv 时
                                    软件移动数据
                                    ↓
                                13. 生成完成事件给用户
```

### 4.3 重传流程（硬件自动）

```
硬件

1. 超时定时器触发
   ↓
2. 检查重传次数
   if (retry_cnt >= max_retry)
       生成错误完成事件 ────────→  软件处理对端失败
   else
       retry_cnt++
   ↓
3. 从重传缓冲区取数据包
   ↓
4. 更新数据包头部
   - 保持原序列号
   - 可能更新时间戳
   ↓
5. 重新发送到网络
   ↓
6. 重新设置超时定时器
   timeout = base_timeout << retry_cnt  (指数退避)
```

### 4.4 ACK 处理（硬件自动）

```
硬件

接收到 ACK 包
   ↓
1. 解析 ACK 序列号
   acked_seq_no = ack_pkt.seq_no
   ↓
2. 释放重传缓冲区
   for (seq = tx_window_start; seq <= acked_seq_no; seq++)
       释放 seq 对应的数据包
       取消超时定时器
   ↓
3. 更新发送窗口
   tx_window_start = acked_seq_no + 1
   unacked_cnt -= (acked_seq_no - old_start + 1)
   ↓
4. 触发新的发送
   if (有待发送的请求 && 窗口有空间)
       继续发送
```

## 5. 性能考虑

### 5.1 延迟优化

**硬件卸载带来的延迟降低：**
- **ACK 生成**: 从软件数十微秒降低到硬件亚微秒级
- **重传响应**: 精确的硬件定时器，无软件轮询延迟
- **零拷贝**: DMA 直接访问用户缓冲区

### 5.2 吞吐量优化

**硬件卸载带来的吞吐量提升：**
- **CPU 释放**: 数据包处理由硬件完成，CPU 可运行应用逻辑
- **批量处理**: 硬件可批量发送/接收数据包
- **流水线**: 硬件流水线并行处理多个对端

### 5.3 资源限制

**硬件资源考虑：**
```
关键资源限制：
- 最大对端数: 受硬件上下文表大小限制（如 1024-4096）
- 重传缓冲区: 受片上或板载内存限制
- 并发未完成请求: 受硬件队列深度限制（如 1024-4096）
- 重排序缓冲区: 每对端接收窗口大小（如 256）

设计权衡：
- 片上 SRAM vs. 板载 DDR: 延迟 vs. 容量
- 静态分配 vs. 动态分配: 性能 vs. 灵活性
```

## 6. 实现建议

### 6.1 渐进式卸载策略

**阶段 1: 基础卸载（最小可行产品）**
- 序列号管理
- 自动 ACK 生成
- 基础重传逻辑
- DMA 数据移动

**阶段 2: 增强卸载**
- 完整的流量控制
- SAR 支持
- 数据包重排序
- 精确超时控制

**阶段 3: 高级优化**
- 拥塞控制
- 多路径支持
- 硬件 RMA/Atomic
- QoS 支持

### 6.2 兼容性考虑

**向后兼容：**
```c
// 软件检测硬件能力
struct rxd_ep *ep = ...;

if (ep->hw_caps & HW_CAP_AUTO_RETRY) {
    // 使用硬件重传
    use_hw_retransmit(ep);
} else {
    // 回退到软件重传
    use_sw_retransmit(ep);
}
```

**渐进式回退：**
- 硬件能力不足时，由软件补充
- 例如：硬件只支持固定窗口大小，软件实现动态调整

### 6.3 验证和测试

**功能验证：**
- 使用现有 fabtests 套件
- 丢包和延迟注入测试
- 多对端压力测试
- 异常场景测试

**性能基准：**
- 延迟测试：fi_rdm_pingpong
- 吞吐量测试：fi_rdm_bandwidth
- CPU 使用率：对比软件实现

## 7. 类似设计参考

### 7.1 业界参考

**RoCE (RDMA over Converged Ethernet)：**
- 硬件实现可靠传输层
- 类似的 QP、CQ 模型
- 拥塞控制（RoCE v2）

**iWARP (Internet Wide Area RDMA Protocol)：**
- 在 TCP 上实现 RDMA
- 部分实现可硬件卸载

**InfiniBand：**
- 完整的硬件卸载可靠传输
- 多种 QoS 级别

### 7.2 RXD 的独特需求

**与 RDMA 的区别：**
- RXD 目标是通用数据报网络（UDP）
- 不假设底层网络提供任何可靠性
- 需要支持更大的 RTT 和丢包率
- 需要更灵活的对端发现机制

## 8. 总结

### 8.1 推荐的分工

| 功能 | 实现位置 | 优先级 | 复杂度 |
|------|---------|--------|--------|
| 序列号管理 | 硬件 | 高 | 低 |
| ACK 生成 | 硬件 | 高 | 低 |
| 重传逻辑 | 硬件 | 高 | 中 |
| 流量控制 | 硬件 | 高 | 中 |
| 数据包 DMA | 硬件 | 高 | 中 |
| SAR | 硬件 | 中 | 中 |
| 重排序 | 硬件 | 中 | 高 |
| 对端发现（RTS/CTS） | 软件 | 高 | 中 |
| 内存注册 | 软件 | 高 | 高 |
| 错误恢复 | 软件 | 高 | 高 |
| 资源管理 | 软件 | 高 | 中 |
| 统计和调试 | 软件（读硬件） | 低 | 低 |

### 8.2 预期收益

**性能提升：**
- 延迟：降低 30-50%（主要来自硬件 ACK）
- 吞吐量：提升 2-3 倍（CPU 释放）
- CPU 使用率：降低 50-70%

**代价：**
- 硬件复杂度增加
- 灵活性降低
- 调试难度增加

### 8.3 设计原则

1. **快速路径卸载**：将性能关键的数据平面操作卸载到硬件
2. **控制平面保留软件**：保持灵活性和可维护性
3. **清晰的接口**：使用标准的队列对模型，便于软件开发
4. **渐进式卸载**：支持部分卸载和软件回退
5. **资源高效**：在硬件资源限制下最大化性能收益
