# uet-rxd: Ultra Ethernet libfabric Provider - 项目计划

**最后更新：** 2025-01-14
**状态：** 规划阶段
**预计周期：** 3-5 个月

---

## 执行摘要

构建全球首个 Ultra Ethernet libfabric provider（用户空间实现），基于成熟的 RXD 架构，支持 HPC 和 AI Base profiles，填补 UET 软件生态空白，使能应用开发在硬件到来前开始。

**核心价值：**
- 🥇 全球首创
- ⚡ 快速实现（3-5 个月）
- 🎯 高影响力（AI/HPC 社区）
- 📚 教育和研究价值
- 💼 职业发展机会

---

## 1. 项目目标

### 1.1 主要目标

**技术目标：**
- 实现符合 UET 规范的 libfabric provider
- 支持 HPC Profile（100%）
- 支持 AI Base Profile（100%）
- 通过 fabtests 验证（≥95% 通过率）
- 性能可接受（与 RXD 基准相比 ±10%）

**生态目标：**
- 使能 UET 应用开发和测试
- 为硬件厂商提供参考实现
- 为标准组织提供实践反馈
- 创建教育资源

**个人目标：**
- 展示系统编程能力
- 在新兴技术领域建立专家形象
- 高质量开源贡献
- 技术博客发表

### 1.2 非目标（明确不做）

❌ AI Full Profile 的高级功能（Multicast, In-network aggregation）
❌ 生产级性能优化（这是原型实现）
❌ 硬件卸载实现（阶段 1 不做）
❌ 完整的 congestion control（可选）
❌ 所有平台支持（先聚焦 Linux）

---

## 2. Profile 支持策略

### 2.1 支持范围

| Profile | 支持度 | 优先级 | 状态 |
|---------|--------|--------|------|
| **HPC Profile** | 100% | P0 | 计划中 |
| **AI Base Profile** | 100% | P0 | 计划中 |
| **AI Full Profile** | 部分（60%） | P2 | 明确说明限制 |

### 2.2 功能清单

**HPC Profile（P0 - 必须）：**
```
✅ Reliable, Ordered Delivery (ROD mode)
✅ RMA operations (read/write)
✅ Atomic operations (基础)
✅ PDC management
✅ Strong ordering guarantees
```

**AI Base Profile（P0 - 必须）：**
```
✅ Reliable, Unordered Delivery (RUD mode)
✅ Selective Acknowledgment (SACK)
✅ PDC lifecycle management
✅ Basic flow control
⚠️ Basic congestion control (可选)
```

**AI Full Profile（P2 - 部分）：**
```
⚠️ RUDI mode (计划 v2.0)
⚠️ Advanced congestion control (可选)
❌ Multicast (用户空间不可行)
❌ In-network aggregation (需要硬件)
```

---

## 3. 技术方案

### 3.1 架构设计

```
Application (MPI, NCCL, etc.)
         ↓
   libfabric API
         ↓
   uet-rxd Provider (~2500 行 C)
   ├─ Protocol Translator (RXD → UET)
   ├─ SACK Implementation
   ├─ PDC Manager
   ├─ Multi-mode Support (ROD/RUD)
   └─ Performance Optimization
         ↓
   UDP Provider (libfabric)
         ↓
   Kernel Network Stack
         ↓
   Standard Ethernet NIC
```

**核心设计决策：**
- 基于 RXD provider 架构（已验证、稳定）
- 用户空间实现（快速迭代、易调试）
- 协议转换层（RXD ↔ UET 头部格式）
- 模块化设计（便于测试和扩展）

### 3.2 关键技术挑战

| 挑战 | 解决方案 | 复杂度 | 代码量 |
|------|---------|--------|--------|
| **RUD mode** | 去除顺序要求，保持可靠性 | ⭐⭐⭐ | 400 行 |
| **SACK** | 位图跟踪 + 选择性重传 | ⭐⭐⭐⭐ | 600 行 |
| **PDC 管理** | Peer → PDC 语义转换 | ⭐⭐ | 300 行 |
| **协议格式** | 头部转换 + 扩展头 | ⭐⭐ | 400 行 |
| **测试验证** | fabtests + 自定义测试 | ⭐⭐⭐ | 800 行 |

### 3.3 参考资源

**已获取：**
- ✅ UET 内核驱动源码（uecon.ko，4460 行）
- ✅ RXD provider 源码（libfabric/prov/rxd）
- ✅ UET 协议规范（lore.kernel.org）

**待查阅：**
- ⚠️ UET 正式规范（等待 UEC 发布）
- ⚠️ UET profile 详细定义

---

## 4. 实施计划

### 4.1 阶段划分

```
阶段 0: 准备工作              (Week 1-2)    ✅ 已完成
阶段 1: HPC Profile           (Week 3-6)    ← 当前
阶段 2: AI Base Profile       (Week 7-10)
阶段 3: 测试和优化            (Week 11-13)
阶段 4: 文档和发布            (Week 14-16)
```

---

### 阶段 0：准备工作（已完成）✅

**目标：** 技术调研、方案设计、资源准备

**已完成任务：**
- ✅ RXD provider 架构分析
- ✅ UET 协议规范研究
- ✅ UET 内核驱动源码下载
- ✅ Profile 支持策略制定
- ✅ 项目计划编写
- ✅ 技术博客策略制定

**交付物：**
- ✅ `/sdc/libfabric/docs/` 下所有分析文档
- ✅ 下载的 UET driver 源码（188KB mbox）
- ✅ 项目计划（本文档）

---

### 阶段 1：HPC Profile 实现

**时间：** Week 3-6（4 周）
**目标：** 实现 HPC Profile 完整功能

#### Week 3：基础框架

**任务：**
```bash
1. 设置开发环境
   - Fork libfabric 仓库
   - 创建 prov/uet_rxd/ 目录
   - 配置 build system

2. 复制和适配 RXD provider
   - cp -r prov/rxd prov/uet_rxd
   - 修改 configure.ac, Makefile.am
   - 重命名符号（rxd_* → uet_*）

3. 基础编译通过
   - 修复编译错误
   - 确保可以加载为 provider
   - 简单的 fi_getinfo() 测试
```

**验收标准：**
- ✅ 代码可编译
- ✅ Provider 可被 libfabric 识别
- ✅ `fi_info -p uet_rxd` 可列出 provider

**预计代码量：** ~500 行（主要是重命名和配置）

---

#### Week 4：协议头部转换

**任务：**
```bash
1. 定义 UET PDS 头部结构
   - 参考 uecon.ko 和 UET 规范
   - 定义 uet_pds_hdr, uet_sar_hdr 等
   - 在 uet_proto.h 中

2. 实现协议转换层
   - rxd_base_hdr → uet_pds_hdr
   - 处理扩展头（SAR, RMA, Atomic）
   - 头部序列化/反序列化

3. ROD mode 基础实现
   - 保持 RXD 的顺序传递逻辑
   - 适配到 UET 协议格式
```

**验收标准：**
- ✅ 头部转换正确（手工验证）
- ✅ 简单的 send/recv 可以工作
- ✅ tcpdump 可以看到 UET 头部

**预计代码量：** ~400 行

---

#### Week 5：RMA 和 Atomic

**任务：**
```bash
1. RMA 操作适配
   - fi_read/fi_write 实现
   - UET RMA 头部格式
   - 地址和密钥处理

2. Atomic 操作适配
   - 基础原子操作（fetch_add, compare_swap）
   - UET Atomic 头部格式
   - 结果返回处理

3. PDC 管理实现
   - Peer → PDC 语义转换
   - PDC ID 生成和管理
   - PDC 生命周期（创建、超时、清理）
```

**验收标准：**
- ✅ fi_read/fi_write 可用
- ✅ fi_atomic 基础操作可用
- ✅ PDC 正确创建和清理

**预计代码量：** ~500 行

---

#### Week 6：HPC Profile 测试

**任务：**
```bash
1. 单元测试
   - 协议转换测试
   - PDC 管理测试
   - 错误处理测试

2. fabtests 验证
   - fi_rdm_pingpong
   - fi_rdm_bandwidth
   - fi_rma_bw
   - 目标：80% 通过率

3. 性能基准测试
   - 延迟（pingpong）
   - 吞吐量（bandwidth）
   - 与 RXD 对比

4. Bug 修复和稳定性
```

**验收标准：**
- ✅ fabtests 通过率 ≥80%
- ✅ 性能在 RXD 的 ±20% 范围内
- ✅ 无明显内存泄漏或崩溃

**预计代码量：** ~300 行（测试代码）

**阶段 1 里程碑：** ✅ HPC Profile 可用

---

### 阶段 2：AI Base Profile 实现

**时间：** Week 7-10（4 周）
**目标：** 实现 AI Base Profile 核心功能（RUD + SACK）

#### Week 7：RUD Mode 实现

**任务：**
```bash
1. RUD mode 数据结构
   - 添加 delivery_mode 字段到 PDC
   - High-water mark (HWM) 管理
   - 重复检测机制

2. RUD 接收逻辑
   - 去除顺序要求
   - 立即交付数据包
   - 更新 HWM
   - 仍跟踪 PSN（为 SACK）

3. RUD 发送逻辑
   - 与 ROD 共享大部分代码
   - 标记 delivery_mode 在头部

4. 模式切换机制
   - 根据 fi_info hints 选择模式
   - 运行时配置（环境变量）
```

**验收标准：**
- ✅ RUD mode 基础功能可用
- ✅ 可以乱序接收并正确交付
- ✅ 无重复交付

**预计代码量：** ~400 行

---

#### Week 8-9：SACK 实现

**任务：**
```bash
1. SACK 数据结构（Week 8）
   - 定义 uet_sack_pkt
   - 256-bit bitmap 管理
   - 乱序包跟踪队列

2. SACK 生成（Week 8）
   - 接收端生成 SACK
   - bitmap 构建逻辑
   - SACK 发送策略（时机、频率）

3. SACK 处理（Week 9）
   - 发送端解析 SACK
   - 识别丢失的包
   - 选择性重传逻辑

4. Hybrid ACK/SACK（Week 9）
   - 周期性 full ACK（每 32 包）
   - 超时 fallback
   - 避免 SACK 丢失导致的死锁
```

**验收标准：**
- ✅ SACK 正确生成和解析
- ✅ 选择性重传工作正常
- ✅ 丢包场景（10%）恢复时间改善 ≥30%

**预计代码量：** ~600 行

---

#### Week 10：AI Base 测试和优化

**任务：**
```bash
1. RUD mode 测试
   - 乱序场景测试
   - 性能测试（vs ROD）
   - Collective 操作测试（如果可能）

2. SACK 压力测试
   - 高丢包率场景（5%, 10%, 20%）
   - 大窗口场景
   - 并发连接测试

3. 性能优化
   - Profiling（gprof, perf）
   - 热点优化
   - 内存优化

4. Bug 修复
```

**验收标准：**
- ✅ RUD mode 通过所有测试
- ✅ SACK 在高丢包下稳定
- ✅ 性能数据记录完整

**预计代码量：** ~300 行（测试）

**阶段 2 里程碑：** ✅ AI Base Profile 可用

---

### 阶段 3：集成测试和优化

**时间：** Week 11-13（3 周）
**目标：** 完整测试、性能优化、稳定性提升

#### Week 11：完整测试

**任务：**
```bash
1. fabtests 完整套件
   - 所有相关测试
   - 目标：95% 通过率
   - 记录未通过的原因

2. 长时间稳定性测试
   - 48 小时连续运行
   - 内存泄漏检测（valgrind）
   - 资源泄漏检测

3. 边界条件测试
   - 大消息（>MTU）
   - 小消息（<64B）
   - 零长度消息
   - 错误注入测试
```

**验收标准：**
- ✅ fabtests 通过率 ≥95%
- ✅ 48 小时无崩溃
- ✅ 无内存泄漏

---

#### Week 12：性能优化和基准测试

**任务：**
```bash
1. 性能 profiling
   - 识别热点函数
   - CPU cache 分析
   - 系统调用开销分析

2. 针对性优化
   - 快速路径优化
   - 减少内存拷贝
   - 批量操作

3. 完整性能基准
   - OSU micro-benchmarks
   - 与 RXD 详细对比
   - 与 uecon.ko 对比（如果可能）
   - 记录所有数据
```

**验收标准：**
- ✅ 延迟 vs RXD：±10%
- ✅ 吞吐量 vs RXD：±10%
- ✅ SACK 优势：丢包恢复 ≥30% 提升
- ✅ RUD 优势：collective ≥10% 提升

---

#### Week 13：代码质量和清理

**任务：**
```bash
1. 代码审查
   - 检查所有 TODO/FIXME
   - 代码风格一致性
   - 注释完整性

2. 错误处理增强
   - 所有错误路径检查
   - 错误消息清晰
   - 优雅降级

3. 日志和调试支持
   - 分级日志（debug/info/warn/error）
   - 调试开关（环境变量）
   - 性能计数器

4. 内存管理审查
   - 所有 malloc/free 配对
   - 对象池使用正确
   - 边界情况处理
```

**验收标准：**
- ✅ 所有 TODO 解决或记录
- ✅ 代码风格一致
- ✅ Valgrind 无警告

**阶段 3 里程碑：** ✅ 代码质量达标

---

### 阶段 4：文档和发布

**时间：** Week 14-16（3 周）
**目标：** 完整文档、GitHub 发布、技术博客

#### Week 14：文档编写

**任务：**
```bash
1. README.md
   - 项目介绍
   - 快速开始
   - Feature 列表
   - Profile 支持说明
   - 性能数据

2. Architecture.md
   - 架构设计
   - 组件说明
   - 数据流程
   - 设计决策

3. API.md
   - libfabric API 映射
   - 使用示例
   - 配置选项
   - 环境变量

4. Performance.md
   - 性能数据详细
   - 与 RXD 对比
   - 调优建议
   - 已知限制

5. Contributing.md
   - 贡献指南
   - 代码规范
   - 提交流程
```

**验收标准：**
- ✅ 所有文档完整且清晰
- ✅ 代码示例可运行
- ✅ 外部审查通过（至少 1 人）

---

#### Week 15：GitHub 发布准备

**任务：**
```bash
1. 仓库准备
   - GitHub repo 创建
   - License 文件（MIT）
   - .gitignore 配置
   - CI/CD 设置（GitHub Actions）

2. Release 准备
   - 创建 v0.1.0 tag
   - Release notes 编写
   - 变更日志（CHANGELOG.md）

3. 社区准备
   - Issue templates
   - PR template
   - Code of Conduct
   - GitHub Discussions 启用

4. 测试最终版本
   - 从 GitHub 全新 clone 测试
   - 确保构建脚本正确
   - 文档链接检查
```

**验收标准：**
- ✅ GitHub 仓库专业完整
- ✅ CI/CD 正常工作
- ✅ 全新环境可构建

---

#### Week 16：技术博客和推广

**任务：**
```bash
1. 技术博客撰写
   - 基于准备好的模板
   - 2500 字左右
   - 包含图表和代码示例
   - 外部审查

2. 视觉材料准备
   - 架构对比图
   - 性能对比图
   - 协议演进图
   - Demo 截图

3. LinkedIn 发布
   - 完整博客文章
   - 配图
   - GitHub 链接

4. 多平台推广
   - Twitter/X（线程）
   - Hacker News（Show HN）
   - Reddit (r/networking, r/HPC)
   - libfabric 邮件列表
   - Ultra Ethernet Consortium

5. 响应反馈
   - 及时回复评论
   - 记录问题和建议
   - 快速修复严重 bug
```

**验收标准：**
- ✅ 博客发布（LinkedIn + 其他平台）
- ✅ GitHub 仓库 public
- ✅ 至少 3 个平台推广

**阶段 4 里程碑：** ✅ 项目正式发布

---

## 5. 时间线和里程碑

### 5.1 甘特图

```
Week    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
        |===|===|===|===|===|===|===|===|===|===|===|===|===|===|===|
准备     [✓✓]
HPC基础         [---]
HPC功能             [-----]
HPC测试                 [---]
RUD Mode                    [-----]
SACK                            [-------]
AI测试                                  [---]
集成测试                                    [-----]
优化                                            [---]
清理                                                [---]
文档                                                    [-----]
发布                                                        [---]
推广                                                            [---]

里程碑：
M0 (Week 2):  ✅ 计划完成
M1 (Week 6):  □ HPC Profile 可用
M2 (Week 10): □ AI Base Profile 可用
M3 (Week 13): □ 代码质量达标
M4 (Week 16): □ 正式发布
```

### 5.2 关键里程碑

| 里程碑 | 时间 | 标准 | 交付物 |
|--------|------|------|--------|
| **M0: 计划完成** | Week 2 | ✅ 完成 | 所有分析文档 |
| **M1: HPC 可用** | Week 6 | fabtests ≥80% | HPC Profile 实现 |
| **M2: AI Base 可用** | Week 10 | RUD+SACK 工作 | AI Base Profile 实现 |
| **M3: 代码达标** | Week 13 | fabtests ≥95% | 完整测试和优化 |
| **M4: 正式发布** | Week 16 | 博客+GitHub | 公开发布 |

### 5.3 关键路径

```
Critical Path:
准备 → HPC基础框架 → 协议转换 → RMA/Atomic →
RUD Mode → SACK → 测试优化 → 文档 → 发布

最长路径：16 周
无缓冲区（所有任务串行）

风险缓解：
- Week 11-13 有 3 周缓冲用于测试和修复
- 某些任务可并行（如文档可提前开始）
```

---

## 6. 资源需求

### 6.1 开发环境

**硬件：**
- 开发机器：1 台（Linux，8GB+ RAM）
- 测试机器：2 台（用于网络测试）
- 可选：访问云主机（AWS/GCP，用于不同网络环境测试）

**软件：**
```bash
必需：
- Linux (Ubuntu 22.04 或 RHEL 8+)
- GCC 9+
- Autotools (autoconf, automake, libtool)
- libfabric 依赖（libnl, libibverbs 等）
- Git

开发工具：
- Vim/VS Code
- GDB
- Valgrind
- perf/gprof
- tcpdump/wireshark

测试工具：
- libfabric fabtests
- OSU micro-benchmarks
- iperf3/netperf
```

### 6.2 时间投入

**全职投入：** 推荐

**时间分配：**
```
编码：      60% (Week 3-10, ~48 小时/周)
测试：      20% (Week 11-13, ~16 小时/周)
文档：      15% (Week 14-15, ~12 小时/周)
推广：      5%  (Week 16, ~4 小时/周)

总计：约 640 小时（16 周 × 40 小时/周）
```

**兼职可行性：**
- 如果每周投入 20 小时
- 需要延长到 30-32 周（7-8 个月）
- 风险：失去先发优势

### 6.3 外部依赖

**低风险依赖：**
- ✅ libfabric 源码（开源，稳定）
- ✅ RXD provider（已有，成熟）
- ✅ UET driver 源码（已下载）

**中等风险依赖：**
- ⚠️ UET 正式规范（预计 Q2 2025 发布）
- ⚠️ Profile 详细定义（可能与 driver 不一致）

**应对策略：**
- 基于现有 driver 实现
- 规范发布后快速调整
- 与 UEC 保持沟通

---

## 7. 风险管理

### 7.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **协议理解错误** | 中 | 高 | 参考 uecon.ko，尽早测试 |
| **性能不达标** | 中 | 中 | 基于 RXD，性能可预期 |
| **SACK 实现复杂** | 中 | 中 | 参考 TCP SACK，分步实现 |
| **fabtests 通过率低** | 低 | 高 | 预留 3 周测试修复时间 |
| **内存泄漏/崩溃** | 低 | 高 | Valgrind，代码审查 |

### 7.2 时间风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **某个阶段超时** | 中 | 中 | 3 周缓冲时间 |
| **UET 规范大变** | 低 | 高 | 基于已有 driver，影响有限 |
| **个人时间不足** | 中 | 高 | 全职投入，减少其他承诺 |

### 7.3 竞争风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **官方 provider 提前发布** | 低 | 高 | 快速执行，强调"首个" |
| **其他人同时实现** | 低 | 中 | 定期分享进展，建立先发优势 |

### 7.4 社区风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **博客反响平淡** | 中 | 低 | 多平台推广，主动联系相关人士 |
| **代码无人用** | 中 | 低 | 技术价值本身存在，长期有效 |
| **负面评论** | 低 | 低 | 诚实说明限制，欢迎建设性反馈 |

---

## 8. 成功标准

### 8.1 技术成功标准

**必须达到（P0）：**
- ✅ 代码编译并可作为 libfabric provider 加载
- ✅ HPC Profile 100% 功能实现
- ✅ AI Base Profile 100% 功能实现
- ✅ fabtests 通过率 ≥95%
- ✅ 无严重内存泄漏或崩溃
- ✅ 性能在 RXD 基准的 ±20% 范围

**期望达到（P1）：**
- ✅ SACK 在丢包场景下性能提升 ≥30%
- ✅ RUD mode 在 collective 操作中提升 ≥10%
- ✅ 48 小时稳定性测试通过
- ✅ 协议边界情况发现 ≥2 个

**可选达到（P2）：**
- ⚠️ AI Full Profile 部分功能（RUDI mode）
- ⚠️ 多平台支持（FreeBSD）

### 8.2 生态成功标准

**短期（发布后 1 个月）：**
- ✅ LinkedIn 浏览量 ≥3,000
- ✅ GitHub Stars ≥20
- ✅ 有意义的技术讨论 ≥3 个
- ✅ 至少 1 个外部用户尝试使用

**中期（发布后 3 个月）：**
- ✅ GitHub Stars ≥50
- ✅ 贡献者（除作者外）≥2
- ✅ Issue/PR ≥10
- ✅ 被相关文章/博客引用 ≥2 次

**长期（发布后 6 个月）：**
- ✅ 成为 UET 开发的参考实现
- ✅ 被 UEC 或 libfabric 官方注意
- ✅ 会议演讲邀请或论文发表机会

### 8.3 个人成功标准

**职业发展：**
- ✅ 技术博客被广泛阅读
- ✅ LinkedIn 人脉扩展（≥50 新连接）
- ✅ 展示系统编程和网络协议能力
- ✅ 开源贡献组合增强

**学习成长：**
- ✅ 深入理解 RDMA 协议
- ✅ 掌握 libfabric 架构
- ✅ 提升 C 编程和调试能力
- ✅ 实践开源项目管理

---

## 9. 沟通和报告

### 9.1 进度跟踪

**每周检查点：**
- 周五：回顾本周进度
- 更新 PLAN.md 中的状态
- 记录问题和决策
- 调整下周计划

**每月回顾：**
- 月末：完整的进度审查
- 对比计划 vs 实际
- 风险重新评估
- 必要时调整计划

### 9.2 文档更新

**持续更新：**
- `PLAN.md` - 每周更新进度
- `CHANGELOG.md` - 每个功能完成时
- `TODO.md` - 动态任务列表
- `DECISIONS.md` - 重要设计决策记录

### 9.3 外部沟通

**定期分享：**
- Week 6: HPC Profile 完成，小范围分享
- Week 10: AI Base Profile 完成，扩大分享
- Week 16: 正式发布，全面推广

**社区互动：**
- libfabric 邮件列表
- Ultra Ethernet Consortium
- 相关 Slack/Discord 频道

---

## 10. 下一步行动

### 10.1 立即行动（Week 3 开始）

```bash
Day 1-2: 环境准备
□ Fork libfabric 到个人 GitHub
□ 本地 clone 并编译
□ 熟悉 RXD provider 代码结构
□ 阅读 libfabric provider 开发文档

Day 3-5: 项目搭建
□ 创建 prov/uet_rxd/ 目录
□ 复制 RXD provider 代码
□ 修改 configure.ac, Makefile.am
□ 重命名符号（rxd_* → uet_*）
□ 确保可以编译

Day 6-7: 验证基础功能
□ 编译并安装 libfabric
□ 测试 fi_info -p uet_rxd
□ 运行 fi_rdm_pingpong（可能失败，但要跑）
□ 设置调试环境（gdb, valgrind）
```

### 10.2 准备检查清单

**开发环境：**
- ✅ Linux 机器（已有）
- ⚠️ libfabric 源码（需要 clone）
- ⚠️ 依赖库安装
- ⚠️ 测试工具安装

**参考资料：**
- ✅ UET driver 源码（已下载）
- ✅ 所有分析文档（已完成）
- ⚠️ libfabric provider 开发指南（需要查阅）
- ⚠️ RXD provider 详细代码阅读

**知识准备：**
- ✅ UET 协议理解
- ⚠️ libfabric API 深入学习
- ⚠️ RXD 架构深入理解

---

## 11. 附录

### 11.1 相关文档

**项目文档（/sdc/libfabric/docs/）：**
- `rxd.md` - RXD Provider 架构文档
- `rxd_hardware_offload.md` - 硬件卸载设计
- `ultra_ethernet_analysis.md` - UET 与 libfabric 关系
- `ue_driver_nature.md` - UET driver 性质分析
- `uet_full_system_design.md` - 全系统设计方案对比
- `uet_profiles_rxd_support.md` - Profile 支持分析
- `uet_blog_objective_style.md` - 技术博客策略
- `PLAN.md` - 本文档

**外部资源：**
- libfabric: https://github.com/ofiwg/libfabric
- Ultra Ethernet: https://ultraethernet.org/
- UET driver RFC: https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/
- UET driver 源码: `/sdc/20250307_nikolay_ultra_ethernet_driver_introduction.mbx`

### 11.2 代码仓库计划

**GitHub 仓库结构：**
```
uet-rxd/
├── README.md
├── LICENSE (MIT)
├── CHANGELOG.md
├── PLAN.md (本文档)
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── performance.md
│   └── contributing.md
├── prov/uet_rxd/
│   ├── src/
│   │   ├── uet_init.c
│   │   ├── uet_ep.c
│   │   ├── uet_msg.c
│   │   ├── uet_rma.c
│   │   ├── uet_atomic.c
│   │   ├── uet_proto.c
│   │   └── ...
│   ├── include/
│   │   ├── uet.h
│   │   ├── uet_proto.h
│   │   └── ...
│   └── Makefile.am
├── tests/
│   ├── unit/
│   └── integration/
└── .github/
    ├── workflows/ (CI/CD)
    └── ISSUE_TEMPLATE/
```

### 11.3 联系和支持

**项目负责人：** [Your Name]
**Email:** [your.email@example.com]
**LinkedIn:** [Your LinkedIn Profile]
**GitHub:** @[yourusername]

**问题报告：**
- GitHub Issues: (发布后创建)
- Email: [your.email@example.com]

**贡献：**
- 欢迎 Pull Requests
- 请先开 Issue 讨论大的改动
- 遵循 CONTRIBUTING.md 指南

---

## 12. 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2025-01-14 | 初始计划创建 | [Your Name] |

---

## 13. 批准和承诺

**项目启动：** 2025-01-15（预计）

**目标完成：** 2025-05-15（16 周后）

**承诺：**
- ✅ 全职投入 16 周
- ✅ 遵循计划，按时交付里程碑
- ✅ 高质量代码和文档
- ✅ 开放透明的开发过程
- ✅ 对社区负责

**签名：** ________________
**日期：** ________________

---

**项目口号：** "Building the future of AI networking, one packet at a time." 🚀

---

**END OF PLAN**
