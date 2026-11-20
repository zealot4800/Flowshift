#!/usr/bin/env python3
"""Measure paired framing/shaped transfers on two accessible SSH guests (default: FlowShift)."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import select
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parent
CASES = ("file-transfer-500mb", "file-transfer-1gb", "file-transfer-2gb",
         "streaming-60s-720p", "streaming-120s-720p", "streaming-180s-720p")


def ssh(target, port, key):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-p", str(port)]
    if key:
        command += ["-i", str(key)]
    return command + [target]


def remote(connection, argv, *, timeout=30, data=None):
    result = subprocess.run(connection + [shlex.join(argv)], input=data,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode()


def source_archive():
    content = io.BytesIO()
    hashes = {}
    paths = [ROOT / "flowshift", ROOT / "create-workload"]
    paths += sorted((ROOT / "configs").glob("*.json"))
    paths += sorted((ROOT / "src").rglob("*.py"))
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for path in paths:
            name = str(path.relative_to(ROOT))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name)
    return content.getvalue(), hashes


def summary(case, run, baseline, shaped, received):
    assert shaped["sha256"] == received["sha256"], "Receiver checksum mismatch"
    assert shaped["original_bytes"] == received["recovered_bytes"], "Receiver size mismatch"
    assert shaped["data_frames"] == received["data_frames"], "Receiver frame count mismatch"
    assert shaped["dummy_frames"] == received["dummy_frames"], "Receiver dummy count mismatch"
    original = shaped["original_bytes"]
    duration = shaped["duration_seconds"]
    return dict(case=case, repetition=run, original_bytes=original,
                total_tunnel_overhead_percent=100 * (shaped["framed_bytes"] / original - 1),
                shaping_over_framed_baseline_percent=shaped["data_framing_overhead_percent"],
                charged_extra_percent=100 * shaped["charged_extra_bytes"] / original,
                framing_duration_s=baseline["duration_seconds"], shaped_duration_s=duration,
                completion_time_overhead_percent=100 * (duration / baseline["duration_seconds"] - 1),
                shaped_payload_mbps=original * 8 / duration / 1e6,
                framing_cpu_s=baseline["process_cpu_seconds"], shaped_cpu_s=shaped["process_cpu_seconds"],
                policy_selection_cpu_s=shaped["selection_cpu_seconds"],
                max_controller_delay_ms=1000 * shaped["max_controller_delay_s"],
                deadline_violations=shaped["controller_deadline_violations"],
                padding_actions=shaped["actions"]["padding"],
                dummy_actions=shaped["actions"]["dummy"],
                pacing_actions=shaped["actions"]["pacing"],
                fragment_actions=shaped["actions"]["fragment"], reconstructed=True)


def transfer(sender, receiver, directory, args, case, method, run):
    document = json.loads((ROOT / "configs" / f"{case}.json").read_text())
    streaming = document["workload_type"] == "streaming"
    source = "video-720p.mp4" if streaming else f"file-{document['profile']}.bin"
    destination = f"{directory}/received/{case}/{run}/{method}"
    # Print readiness after bind/listen, without consuming the one accepted connection.
    code = ("import json,socket; from flowshift.receiver import receive_connection; "
            "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
            f"s.bind(({args.receiver_bind!r},{args.data_port})); s.listen(1); "
            "print('READY',flush=True); c,_=s.accept(); "
            "c.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1); "
            f"r=receive_connection(c,{destination!r}); print(json.dumps(r.to_dict())); c.close(); s.close()")
    process = subprocess.Popen(receiver + [shlex.join(["env", f"PYTHONPATH={directory}/src", "python3", "-u", "-c", code])],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timeout = max(600, document.get("duration_seconds", 0) + 120)
    try:
        if not select.select([process.stdout], [], [], 20)[0] or process.stdout.readline().strip() != b"READY":
            raise RuntimeError("Receiver failed to become ready")
        command = [f"{directory}/flowshift", "stream-send" if streaming else "send",
                   f"{directory}/workloads/{source}", "--config", f"{directory}/configs/{case}.json",
                   "--host", args.receiver_ip, "--port", str(args.data_port),
                   "--method", method, "--seed", str(42 + run)]
        if args.sender_bind:
            command += ["--bind", args.sender_bind]
        sent = remote(sender, command, timeout=timeout)
        output, error = process.communicate(timeout=30)
        if process.returncode:
            raise RuntimeError(error.decode(errors="replace"))
        result = {"sender": json.loads(sent), "receiver": json.loads(output)}
        assert result["sender"]["sha256"] == result["receiver"]["sha256"], "Receiver checksum mismatch"
        assert result["sender"]["original_bytes"] == result["receiver"]["recovered_bytes"], "Receiver size mismatch"
        log = args.output / f"{case}-{run}-{method}.json"
        log.write_text(json.dumps(result, indent=2) + "\n")
        # Verified above; drop the copy so long runs do not fill the receiver's /tmp.
        remote(receiver, ["rm", "-rf", destination])
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender-ssh", default="zealot@192.168.56.11")
    parser.add_argument("--receiver-ssh", default="zealot@192.168.56.12")
    parser.add_argument("--sender-port", type=int, default=22)
    parser.add_argument("--receiver-port", type=int, default=22)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--receiver-ip", default="10.10.0.2", help="receiver address reachable from sender VM")
    parser.add_argument("--receiver-bind", default="0.0.0.0")
    parser.add_argument("--sender-bind")
    parser.add_argument("--data-port", type=int, default=9443)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--method", choices=("flowshift", "adaptive"), default="flowshift",
                        help="shaping policy compared against the framing-only baseline")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "reports" / "flowshift_vm")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run.json").exists():
        parser.error("output already contains a run; choose a fresh --output directory")
    sender = ssh(args.sender_ssh, args.sender_port, args.key)
    receiver = ssh(args.receiver_ssh, args.receiver_port, args.key)
    archive, hashes = source_archive()
    started = datetime.now(timezone.utc)
    directory = f"/tmp/flowshift-benchmark-{started.strftime('%Y%m%dT%H%M%S%f')}"
    metadata = dict(started_at=started.isoformat(), status="preflight", source_sha256=hashes,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    remote_directory=directory,
                    measurement_basis="application tunnel bytes; excludes TCP/IP headers, ACKs and retransmissions",
                    baseline="same framed tunnel and workload with Policy (no shaping)",
                    streaming="real-time FFmpeg encode and transport; receiver saves stream without playback",
                    obfuscation="applied action counts; classifier evasion is not measured")
    rows = []
    def save():
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if rows:
            with (args.output / "comparison.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    save()
    try:
        for role, connection in (("sender", sender), ("receiver", receiver)):
            metadata[role + "_host"] = remote(connection, ["uname", "-a"]).strip()
            print(f"{role}: {metadata[role + '_host']}", flush=True)
            remote(connection, ["mkdir", "-p", directory])
            remote(connection, ["tar", "-xzf", "-", "-C", directory], data=archive)
        metadata["status"] = "running"; save()
        for case in args.cases:
            document = json.loads((ROOT / "configs" / f"{case}.json").read_text())
            flag = "--file" if document["workload_type"] == "file-transfer" else "--stream"
            profile = document["profile"].removesuffix("-720p")
            print(remote(sender, [f"{directory}/create-workload", flag, profile], timeout=180), flush=True)
            for run in range(args.repetitions):
                results = {}
                # Alternate order across repetitions to reduce a fixed warm-cache advantage.
                methods = ("framing", args.method) if run % 2 == 0 else (args.method, "framing")
                for method in methods:
                    print(f"Running {case}, {method}, repetition {run + 1}", flush=True)
                    results[method] = transfer(sender, receiver, directory, args, case, method, run)
                rows.append(summary(case, run, results["framing"]["sender"],
                                    results[args.method]["sender"], results[args.method]["receiver"]))
                save(); print(json.dumps(rows[-1]), flush=True)
        _, after_hashes = source_archive()
        if after_hashes != hashes:
            raise RuntimeError("Local source/configuration changed during benchmark")
        metadata["status"] = "complete"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
