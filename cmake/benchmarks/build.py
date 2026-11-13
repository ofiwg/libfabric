#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
"""Compare matched TCP/sockets builds from independent source snapshots."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--jobs', type=int, default=8)
parser.add_argument('--runs', type=int, default=3)
args = parser.parse_args()
source = Path(__file__).resolve().parents[2]
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
if any(output.iterdir()):
    parser.error('output directory must be empty')
if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=source).returncode:
    parser.error('commit source changes before benchmarking')
env = os.environ.copy()
env.update(CC=shutil.which('cc'), CFLAGS='-O2 -DNDEBUG')
providers = 'verbs efa psm2 psm3 opx cxi ucx usnic lpp shm sm2 lnx rxm rxd mrail udp'.split()
hooks = 'perf trace profile monitor hook_debug hook_hmem dmabuf_peer_mem'.split()
cmake_options = [f'-DLIBFABRIC_PROVIDER_{name.upper()}=OFF' for name in providers]
cmake_options += [f'-DLIBFABRIC_HOOK_{name.upper()}=OFF' for name in hooks]
cmake_options += [
    '-DLIBFABRIC_PROVIDER_TCP=ON', '-DLIBFABRIC_PROVIDER_SOCKETS=ON',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_FLAGS=',
    '-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG', '-DBUILD_SHARED_LIBS=ON',
    '-DLIBFABRIC_BUILD_TESTING=OFF', '-DLIBFABRIC_BUILD_FABTESTS=OFF',
    '-DLIBFABRIC_ENABLE_LIBURING=OFF', '-DLIBFABRIC_ENABLE_XPMEM=OFF',
    '-DLIBFABRIC_KDREG2_MONITOR=OFF',
    # configure.ac's int128 probe omits stddef.h and fails on NULL. Match its
    # disabled result so the timing and size comparison use equivalent code.
    '-D_have_int128_atomics=OFF',
]
autotools_options = [
    '--enable-only', '--enable-tcp', '--enable-sockets', '--disable-static',
    '--disable-debug', '--disable-profile', '--without-uring',
    '--disable-xpmem', '--disable-kdreg2',
] + [f'--disable-{name}' for name in providers + hooks]
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=source).decode().split('\0')
results = {'host': platform.platform(), 'jobs': args.jobs, 'runs': [],
           'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()}
for name, cmd in {'compiler': ['cc', '--version'], 'cmake': ['cmake', '--version'],
                  'ninja': ['ninja', '--version'], 'make': ['make', '--version'],
                  'autoreconf': ['autoreconf', '--version']}.items():
    results[name] = subprocess.check_output(cmd, text=True).splitlines()[0]


def run(cmd, cwd, log):
    start = time.perf_counter()
    with log.open('a') as stream:
        stream.write(json.dumps(cmd) + '\n')
        stream.flush()
        subprocess.run(cmd, cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


for repetition in range(args.runs):
    # Alternate order to reduce systematic filesystem-cache and thermal bias.
    for system in (['cmake', 'autotools'] if repetition % 2 == 0 else ['autotools', 'cmake']):
        root = output / f'{system}-{repetition}'
        snapshot = root / 'source'
        snapshot.mkdir(parents=True)
        for relative in filter(None, files):
            original = source / relative
            if original.is_file():
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        build = root / 'build'
        build.mkdir()
        log = root / 'build.log'
        row = {'system': system, 'repetition': repetition, 'bootstrap': 0}
        if system == 'autotools':
            row['bootstrap'] = run(['autoreconf', '-fi'], snapshot, log)
            configure = [str(snapshot / 'configure'), *autotools_options]
            compile_command = ['make', f'-j{args.jobs}']
        else:
            configure = ['cmake', '-S', str(snapshot), '-B', str(build), '-G', 'Ninja', *cmake_options]
            compile_command = ['ninja', '-j', str(args.jobs)]
        row['configure'] = run(configure, build, log)
        row['compile'] = run(compile_command, build, log)
        row['total'] = row['bootstrap'] + row['configure'] + row['compile']
        row['noop'] = statistics.median(run(compile_command, build, log) for _ in range(5))
        incremental = []
        for _ in range(3):
            os.utime(snapshot / 'prov/tcp/src/xnet_ep.c', None)
            incremental.append(run(compile_command, build, log))
        row['incremental'] = statistics.median(incremental)
        library = next((build / 'src/.libs' if system == 'autotools' else build).glob('libfabric.so.1.*'))
        stripped = root / 'libfabric.stripped.so'
        shutil.copy2(library, stripped)
        subprocess.run(['strip', '--strip-unneeded', str(stripped)], check=True)
        row['stripped_bytes'] = stripped.stat().st_size
        row['size'] = subprocess.check_output(['size', str(stripped)], text=True)
        symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
        (root / 'symbols.txt').write_text('\n'.join(sorted(line.split()[-1] for line in symbols.splitlines())) + '\n')
        results['runs'].append(row)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)
