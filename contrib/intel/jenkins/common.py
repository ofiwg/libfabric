import collections
import ci_site_config
import subprocess
import sys

def get_node_name(host, interface):
   # This is the pattern we follow in SFS team cluster
   return "%s-%s" % (host, interface)

def run_command(command):
    print(" ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    print(p.returncode)
    while True:
        out = p.stdout.read(1)
        if (out == "" and p.poll() != None):
            break
        if (out != ""):
            sys.stdout.write(out)
            sys.stdout.flush()
    if (p.returncode != 0):
        print("exiting with " + str(p.poll()))
        sys.exit(p.returncode)


Prov = collections.namedtuple('Prov', 'name enable force_dl')
prov_list = [
    Prov("bgq",        False, False),
    Prov("efa",        False, False),
    Prov("gni",        False, False),
    Prov("hook_debug", False, False),
    Prov("mrail",      False, False),
    Prov("perf",       False, False),
    Prov("psm",        False, False),
    Prov("psm2",       True,  True ),
    Prov("psm3",       True,  False),
    Prov("rstream",    False, False),
    Prov("rxd",        True,  False),
    Prov("rxm",        True,  False),
    Prov("shm",        True,  False),
    Prov("sockets",    True,  False),
    Prov("tcp",        True,  False),
    Prov("udp",        True,  False),
    Prov("usnic",      False, False),
    Prov("verbs",      True,  False),
]

