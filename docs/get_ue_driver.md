# 如何获取 Ultra Ethernet 驱动源代码

## 1. 官方 Patch 系列位置

这个 patch 系列已经提交到 Linux 内核邮件列表（LKML）：

**邮件列表归档：**
- **主链接**: https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/
- **作者**: Nikolay Aleksandrov (Nvidia)
- **时间**: 2025年3月19日

## 2. 下载方法

### 方法 1: 下载完整 Patch 系列（推荐）

**使用 `b4` 工具（Linux 内核开发者标准工具）：**

```bash
# 1. 安装 b4 工具
pip install b4

# 2. 下载完整的 patch 系列
b4 am https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/

# 这会生成一个 .mbx 文件，包含所有 13 个 patch
# 文件名类似: v1_20250319_razor_ultra_ethernet_driver_introduction.mbx
```

**应用到内核源码树：**

```bash
# 3. 切换到你的 Linux 内核源码目录
cd /path/to/linux

# 4. 应用 patch 系列
git am /path/to/v1_*_ultra_ethernet*.mbx

# 或者如果不用 git，使用 patch 命令
patch -p1 < /path/to/extracted_patches/*.patch
```

### 方法 2: 手动下载单个 Patch

访问 lore.kernel.org，每个 patch 都有单独的链接：

**Cover Letter (00/13):**
```
https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/
```

**单个 Patch 链接模式:**
```
https://lore.kernel.org/netdev/[message-id]/raw
```

**下载示例：**
```bash
# 下载 cover letter
wget https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/raw

# 下载所有 patch (需要知道每个 message-id)
# 通常可以从 cover letter 页面的链接找到
```

### 方法 3: 使用 curl 下载整个线程

```bash
# 下载完整的邮件线程（mbox 格式）
curl -L "https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/t.mbox.gz" | gunzip > ultra_ethernet.mbox

# 查看 mbox 文件
less ultra_ethernet.mbox

# 从 mbox 提取 patch
git am ultra_ethernet.mbox
```

### 方法 4: 使用 lei (Local Email Interface)

```bash
# lei 是 Linux 邮件搜索工具
lei q -o ultra-ethernet \
  's:Ultra Ethernet driver introduction' \
  --threads \
  https://lore.kernel.org/netdev/

cd ultra-ethernet
# 所有相关邮件都在这里
```

## 3. 查看 Patch 系列内容

**Patch 列表（13个补丁）：**

```
00/13: [RFC PATCH] Ultra Ethernet driver introduction (cover letter)
01/13: drivers: ultraeth: add initial skeleton and kconfig option
02/13: drivers: ultraeth: add context support
03/13: drivers: ultraeth: add new genl family
04/13: drivers: ultraeth: add job support
05/13: drivers: ultraeth: add tunnel udp device support
06/13: drivers: ultraeth: add initial PDS infrastructure
07/13: drivers: ultraeth: add request and ack receive support
08/13: drivers: ultraeth: add request transmit support
09/13: drivers: ultraeth: add support for coalescing ack
10/13: drivers: ultraeth: add sack support
11/13: drivers: ultraeth: add nack support
12/13: drivers: ultraeth: add initiator and target idle timeout support
13/13: HACK: drivers: ultraeth: add char device
```

## 4. 源代码文件结构

应用 patch 后，你会得到：

```
linux/
├── drivers/
│   ├── ultraeth/
│   │   ├── Kconfig                    # 配置选项
│   │   ├── Makefile                   # 编译配置
│   │   ├── uecon.c                    # 软件设备模型 (324 行)
│   │   ├── uet_chardev.c              # 字符设备 (264 行)
│   │   ├── uet_context.c              # 上下文管理 (274 行)
│   │   ├── uet_job.c                  # Job 管理 (456 行)
│   │   ├── uet_main.c                 # 主入口 (41 行)
│   │   ├── uet_netlink.c              # Netlink 接口 (113 行)
│   │   ├── uet_netlink.h              # Netlink 头文件 (29 行)
│   │   ├── uet_pdc.c                  # PDC 实现 (1122 行)
│   │   └── uet_pds.c                  # PDS 实现 (481 行)
│   └── ...
├── include/
│   ├── net/ultraeth/
│   │   ├── uecon.h                    # (28 行)
│   │   ├── uet_chardev.h              # (11 行)
│   │   ├── uet_context.h              # (47 行)
│   │   ├── uet_job.h                  # (80 行)
│   │   ├── uet_pdc.h                  # (170 行)
│   │   └── uet_pds.h                  # (110 行)
│   └── uapi/linux/
│       ├── ultraeth.h                 # 用户空间 API (536 行)
│       └── ultraeth_nl.h              # Netlink API (116 行)
└── Documentation/
    └── netlink/specs/
        └── ultraeth.yaml              # Netlink 规范 (218 行)

总计约 4460 行代码
```

## 5. 编译和测试

### 5.1 编译模块

```bash
# 进入内核源码目录
cd linux/

# 配置内核（启用 Ultra Ethernet）
make menuconfig
# 导航到: Device Drivers -> Ultra Ethernet support
# 选择 <M> 编译为模块或 <Y> 内置

# 或者直接修改 .config
echo "CONFIG_ULTRAETH=m" >> .config

# 编译模块
make M=drivers/ultraeth

# 编译结果
ls drivers/ultraeth/*.ko
# 应该看到: ultraeth.ko (或分离的 ultraeth.ko 和 uecon.ko)
```

### 5.2 加载模块

```bash
# 加载模块
sudo insmod drivers/ultraeth/ultraeth.ko

# 查看模块信息
modinfo ultraeth.ko

# 查看内核日志
dmesg | grep ultraeth

# 查看 netlink 接口
ip link help 2>&1 | grep ultraeth
```

### 5.3 创建 UET 设备

```bash
# 使用 netlink 创建 Ultra Ethernet tunnel 设备
sudo ip link add name ue0 type ultraeth ...

# 或使用字符设备进行测试（如果启用了 chardev patch）
ls /dev/uet*
```

## 6. 其他资源

### 6.1 官方网站

**Ultra Ethernet Consortium:**
- https://ultraethernet.org/

**规范文档（即将发布）:**
- Ultra Ethernet Transport (UET) Specification
- Packet Delivery Sublayer (PDS) Specification

### 6.2 邮件列表讨论

**查看完整讨论:**
- https://lore.kernel.org/netdev/?q=ultra+ethernet

**订阅邮件列表:**
```bash
# netdev 邮件列表
# 访问: https://lore.kernel.org/netdev/
```

### 6.3 相关会议和演讲

**Netdev 0x19 Conference (Zagreb, Croatia):**
- "Networking For AI BoF" 会议
- 链接: https://netdevconf.info/0x19/sessions/bof/networking-for-ai-bof.html

## 7. Git 仓库（可能的位置）

虽然目前这是一个 RFC patch，还没有合入主线内核，但可能存在开发仓库：

**可能的位置（需要确认）:**

```bash
# 1. 检查 Nvidia 的 GitHub
# https://github.com/NVIDIA

# 2. 检查 Ultra Ethernet Consortium 的仓库
# 可能在 https://github.com/ultraethernet/ （如果存在）

# 3. 作者可能的个人仓库
# 搜索 Nikolay Aleksandrov 的 GitHub/GitLab
```

**目前最可靠的方式是从 lore.kernel.org 获取 patch。**

## 8. 快速开始脚本

创建一个一键下载和应用脚本：

```bash
#!/bin/bash
# download_ue_driver.sh

set -e

# 安装依赖
echo "Installing dependencies..."
pip install b4

# 下载 patch 系列
echo "Downloading Ultra Ethernet driver patches..."
b4 am https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/

# 查找生成的 mbox 文件
MBOX=$(ls -t v*ultra*ethernet*.mbx | head -1)

if [ -z "$MBOX" ]; then
    echo "Error: mbox file not found"
    exit 1
fi

echo "Downloaded: $MBOX"

# 如果在内核源码目录，提示应用
if [ -f "Makefile" ] && grep -q "Linux kernel" Makefile 2>/dev/null; then
    echo ""
    echo "You are in a Linux kernel source tree."
    echo "To apply the patches, run:"
    echo "  git am $MBOX"
else
    echo ""
    echo "To apply these patches to a Linux kernel source tree:"
    echo "  cd /path/to/linux"
    echo "  git am /path/to/$MBOX"
fi
```

**使用方法：**

```bash
chmod +x download_ue_driver.sh
./download_ue_driver.sh
```

## 9. 验证下载的代码

下载完成后，验证关键文件：

```bash
# 检查关键文件是否存在
cd linux/  # 假设已应用 patch

# 验证目录结构
test -d drivers/ultraeth && echo "✓ drivers/ultraeth exists"
test -f drivers/ultraeth/uecon.c && echo "✓ uecon.c exists"
test -f drivers/ultraeth/uet_pds.c && echo "✓ uet_pds.c exists"

# 统计代码行数
find drivers/ultraeth include/net/ultraeth include/uapi/linux/ultraeth* \
  -name '*.[ch]' -exec wc -l {} + | tail -1
# 应该接近 4460 行

# 查看 Kconfig
cat drivers/ultraeth/Kconfig
```

## 10. 故障排除

### 问题 1: b4 工具无法安装

```bash
# 方法 1: 使用 pip3
pip3 install b4

# 方法 2: 从源码安装
git clone https://git.kernel.org/pub/scm/utils/b4/b4.git
cd b4
python setup.py install
```

### 问题 2: 无法应用 patch

```bash
# 确保你在正确的内核版本上
# Ultra Ethernet driver 针对较新的内核

# 检查内核版本
make kernelversion

# 如果有冲突，可以手动查看 patch 内容
vim /path/to/ultra_ethernet.mbox

# 或者使用 git format-patch 提取单个文件
```

### 问题 3: 找不到 message-id

```bash
# 访问 lore.kernel.org 网页
# 浏览器打开: https://lore.kernel.org/netdev/20250319191946.GP9311@nvidia.com/
# 点击每个 patch 的链接，在 URL 中可以看到 message-id
```

## 11. 下一步

下载并应用 patch 后：

1. **阅读代码**
   ```bash
   cd drivers/ultraeth
   vim uecon.c uet_pds.c
   ```

2. **理解架构**
   - 阅读 `uet_pds.h` 了解数据结构
   - 阅读 `uet_pdc.c` 了解协议实现

3. **测试功能**
   - 编译并加载模块
   - 使用 chardev 或 netlink 接口测试

4. **参与开发**
   - 加入邮件列表讨论
   - 提供反馈和改进建议

---

**总结：最简单的方法是使用 `b4 am` 命令从 lore.kernel.org 下载完整的 patch 系列。**
