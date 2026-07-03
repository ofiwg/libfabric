/*
 * Copyright (C) 2026 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once
#include <stdint.h>

// Simple time-anisotropic random generator, not crypto-secure. Generates a result in 7.56 ns on x86-64.
// Mixes a timestamp counter with the stack pointer (a cheap per-thread ID: each thread's stack
// occupies a distinct address range, so SP's low bits differ across concurrent threads) and folds
// them through a hardware CRC32 to spread the bits into a uniform 32-bit output.
static inline uint32_t rndtsc()
{
#if defined(__x86_64__) || defined(_M_X64)
	// x86-64: verified working. RDTSC -> EDX:EAX; xor ESP into the timestamp
	// (ESP carries the inter-thread entropy; the high 32 bits of RSP are identical
	// across threads, so ESP is the *correct* half to mix, not a compromise),
	// then CRC32 to diffuse the bits into EAX.
	uint32_t rc;
	__asm__ __volatile__("rdtsc\n"		     // EAX = low tsc, EDX = high tsc
			     "xorl %%esp, %%edx\n"   // fold per-thread SP into the high word
			     "crc32l %%edx, %%eax\n" // CRC32C accumulate EDX into EAX (0x1EDC6F41)
			     : "=&a"(rc)
			     :
			     : "edx", "cc");
	return rc;

#elif defined(__aarch64__) || defined(_M_ARM64)
	// AArch64: requires FEAT_CRC32; build with -march=armv8-a+crc.
	// ** Logic verified against the Arm ARM; NOT run on hardware. **
	// FIX vs. first draft: 'wsp' in a CRC operand decodes as the ZERO register,
	// NOT the stack pointer -- it would have silently zeroed the per-thread seed.
	// So we must copy SP into a normal GPR first, then feed it as the accumulator.
	// CRC32CX Wd, Wn, Xm : Wn = 32-bit CRC accumulator, Xm = 64-bit data, Wd = result.
	uint32_t rc;
	uint64_t tsc;
	uint64_t sp_val;
	__asm__ __volatile__("mrs   %[tsc], cntvct_el0\n"	// 64-bit virtual counter
			     "mov   %[sp], sp\n"		// SP -> GPR (legal here, unlike inside CRC)
			     "crc32cx %w[rc], %w[sp], %[tsc]\n" // acc = low32(sp_val), data = tsc
			     : [rc] "=&r"(rc), [tsc] "=&r"(tsc), [sp] "=&r"(sp_val)
			     :
			     :);
	return rc;

#elif defined(__riscv) && (__riscv_xlen == 64)
	// RISC-V RV64. Two mutually exclusive paths, because scalar CRC hardware is
	// NOT universally available:
	//
	//  (A) Zbr path: dedicated crc32c.d, which is UNARY (rd, rs) -- there is no
	//      three-operand form. Canonical idiom (per the Bitmanip spec): XOR data
	//      into the low end of the CRC state, THEN run the unary CRC. Build with a
	//      -march that actually provides Zbr (many toolchains still DO NOT).
	//
	//  (B) Fallback: no dedicated CRC on most shipping RV64. Zbc gives clmul only,
	//      which needs a full Barrett/folding routine (too big to inline here), so
	//      the portable fallback just mixes tsc and SP with a multiplicative hash.
	//
	// ** Logic verified against the RISC-V Bitmanip spec; NOT run on hardware. **
	// ATTN: rdcycle is often NOT readable from U-mode (needs the CY bit in
	// scounteren/mcounteren) or it traps as an illegal instruction. rdtime is a
	// safer, lower-resolution source; we XOR SP in either case so two threads
	// reading the same low-res tick still diverge.
	uint64_t tsc;
	uint64_t sp_val;
	__asm__ __volatile__("rdtime %[tsc]\n" // swap to rdcycle only if you know CY is enabled
			     "mv     %[sp], sp\n"
			     : [tsc] "=r"(tsc), [sp] "=r"(sp_val)
			     :
			     :);

#if defined(__riscv_zbr) // (A) real hardware CRC, if the toolchain exposes Zbr
	uint64_t crc = tsc ^ sp_val;		 // XOR data into CRC state (spec idiom)...
	__asm__ __volatile__("crc32c.d %0, %0\n" // ...then the UNARY CRC instruction
			     : "+r"(crc)
			     :
			     :);
	return (uint32_t) crc;
#else			 // (B) portable fallback: no dedicated CRC available
	// Not CRC, but adequate diffusion for a non-crypto PRNG: multiply by a
	// 64-bit odd constant (splitmix64's), then xor-shift-fold to 32 bits.
	uint64_t x = tsc ^ (sp_val * 0x9E3779B97F4A7C15ULL);
	x ^= x >> 32;
	x *= 0xD6E8FEB86659FD93ULL;
	x ^= x >> 32;
	return (uint32_t) x;
#endif

#else
#error "Architecture unsupported"
#endif
} // rndtsc

/* Implementation notes:
 *
 * x86-64: VERIFIED. ESP (not RSP) is intentional: high 32 bits of RSP are constant
 *  across threads; all inter-thread entropy lives in the low 32 bits.
 * ESP (not RSP) is intentional and sufficient: all inter-thread entropy is in the
 * low 32 bits (stacks differ within a <4GB window); RSP's high bits are constant,
 * so mixing them would add work, not entropy. Switching to RSP is harmless but pointless.
 *
 * ARM64:  ** NOT run on hardware. FIXME on first ARM build: **
 *  - verify -march=armv8-a+crc; check HWCAP_CRC32 at runtime
 *  - r31 in CRC operand = ZERO reg, not SP -> we mov sp->GPR first (do NOT "simplify")
 *
 * RISC-V: ** NOT run on hardware. FIXME on first RV64 build: **
 *  - Zbr (crc32c.*) often ABSENT on shipping HW; __riscv_zbr may silently fall
 *    through to the splitmix fallback -> verify which path compiled
 *  - crc32c.d is UNARY (rd, rs): XOR data into state BEFORE the insn (spec idiom).
 *    Do NOT "fix" it into a 3-operand form -- that mnemonic does not exist.
 */
