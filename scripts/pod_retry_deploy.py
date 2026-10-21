"""Retry cheap community GPU pods until one works, then launch training. Log: logs/retry.log"""
import json, re, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path("/home/nand/Documents/personal-projects/ShowdownAgent")
KEY = re.search(r"'(rpa_[^']+)'", Path.home().joinpath(".runpod/config.toml").read_text()).group(1)
RPC = str(Path.home() / ".local/bin/runpodctl")
SSHKEY = str(Path.home() / ".runpod/ssh/runpodctl-ssh-key")
# RTX 3080 Ti dropped: it kept landing on a host in ES that never boots containers.
GPUS = [("NVIDIA GeForce RTX 3090", 16), ("NVIDIA GeForce RTX 3090", 12),
        ("NVIDIA RTX PRO 4500 Blackwell", 12)]  # RTX 5090 ($0.69/h) dropped: ~3x fewer steps per dollar
# Dead man's switch: delete the pod 40 min after boot unless training was launched.
GUARD = ("bash -c '(sleep 2400; [ -f /workspace/launched ] || runpodctl remove pod $RUNPOD_POD_ID) & "
         "exec /start.sh'")
log_f = open(ROOT / "logs/retry.log", "a", buffering=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, file=log_f)

def gql(query):
    req = urllib.request.Request("https://api.runpod.io/graphql", json.dumps({"query": query}).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def deploy(gpu, vcpu):
    q = ('mutation { podFindAndDeployOnDemand(input: { cloudType: COMMUNITY, gpuCount: 1, gpuTypeId: %s, '
         'minVcpuCount: %d, minMemoryInGb: 16, containerDiskInGb: 30, volumeInGb: 5, volumeMountPath: "/workspace", '
         'name: "porygon-train", imageName: "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", startSsh: true, '
         'ports: "22/tcp", dockerArgs: %s, env: [{key: "RUNPOD_API_KEY", value: "%s"}] }) '
         '{ id costPerHr vcpuCount } }') % (json.dumps(gpu), vcpu, json.dumps(GUARD), KEY)
    return (gql(q).get("data") or {}).get("podFindAndDeployOnDemand")

def ssh_addr(pod_id):
    for _ in range(120):
        out = subprocess.run([RPC, "ssh", "info", pod_id], capture_output=True, text=True).stdout
        try:
            d = json.loads(out)
            if "ip" in d: return d["ip"], str(d["port"])
        except json.JSONDecodeError: pass
        time.sleep(5)

def cuda_ok(ip, port):
    for _ in range(12):
        r = subprocess.run(["timeout", "40", "ssh", "-i", SSHKEY, "-p", port, "-o", "StrictHostKeyChecking=accept-new",
                            "-o", "ConnectTimeout=10", f"root@{ip}",
                            "python -c 'import torch; torch.zeros(1).cuda(); print(\"CUDAOK\")' 2>&1 | tail -1"],
                           capture_output=True, text=True)
        if r.stdout.strip(): return r.stdout.strip() == "CUDAOK"
        time.sleep(5)
    return False

deadline, bad = time.time() + 12 * 3600, 0
while time.time() < deadline:
    for gpu, vcpu in GPUS:
        try: pod = deploy(gpu, vcpu)
        except Exception as e: log("deploy error", e); continue
        if not pod: continue
        log("got", gpu, pod)
        addr = ssh_addr(pod["id"])
        if addr and cuda_ok(*addr):
            log("ready", pod["id"], addr)
            r = subprocess.run(["/home/nand/Documents/personal-projects/ShowdownAgent/scripts/pod_launch_resume.sh", *addr, str(pod["costPerHr"]), str(pod["vcpuCount"])],
                               capture_output=True, text=True)
            log(r.stdout[-1500:], r.stderr[-800:])
            if "LAUNCHED" in r.stdout:
                (ROOT / "logs/pod_addr").write_text(f"{pod['id']} {addr[0]} {addr[1]}\n"); log("DONE"); raise SystemExit
        log("bad pod, deleting", pod["id"], addr)
        subprocess.run([RPC, "pod", "delete", pod["id"]], capture_output=True)
        bad += 1
        if bad >= 4:
            log("4 bad pods, giving up"); raise SystemExit
    time.sleep(120)
log("gave up after 12h")
