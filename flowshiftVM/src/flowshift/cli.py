"""Command-line interface for the FlowShift prototype."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import secrets

from .policy import FlowShift, Policy, TargetShift, load_config, load_document
from .receiver import receive_once
from .sender import send_file, send_video


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
FILE_CONFIG = CONFIG_DIR / "file-transfer-500mb.json"
STREAM_CONFIG = CONFIG_DIR / "streaming-60s-720p.json"
PROFILE = CONFIG_DIR / "flowshift-profile.json"  # written by ./export-profile
POLICIES = {"flowshift": TargetShift, "adaptive": FlowShift, "framing": Policy}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowshift")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser('inspect', aliases=['schedule'], help='validate online controller settings')
    inspect.add_argument('--config', default=str(FILE_CONFIG))
    inspect.add_argument('--count', type=int, default=1, help=argparse.SUPPRESS)

    receive = subparsers.add_parser("receive", help="receive and restore one file")
    receive.add_argument("--bind", default="0.0.0.0")
    receive.add_argument("--port", type=int, default=9443)
    receive.add_argument("--output-dir", default="received")
    receive.add_argument("--play", action="store_true")

    send = subparsers.add_parser("send", help="transform and send a file")
    send.add_argument("file")
    send.add_argument("--host", required=True)
    send.add_argument("--port", type=int, default=9443)
    send.add_argument("--config", default=str(FILE_CONFIG))
    send.add_argument("--bind")

    stream = subparsers.add_parser(
        "stream-send", help="encode and transform an actual video in real time"
    )
    stream.add_argument("video")
    stream.add_argument("--host", required=True)
    stream.add_argument("--port", type=int, default=9443)
    stream.add_argument("--config", default=str(STREAM_CONFIG))
    stream.add_argument("--bind")
    for command in (send, stream):
        command.add_argument('--byte-fraction', type=float)
        command.add_argument('--method', choices=sorted(POLICIES), default='flowshift',
                             help='obfuscation policy run by the common pipeline')
        command.add_argument('--settings', help='policy settings JSON; flowshift defaults to '
                             'configs/flowshift-profile.json, the others take none')
        command.add_argument('--seed', type=int, help='policy random seed; drawn fresh when omitted')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in ('inspect', 'schedule'):
        print(json.dumps(load_config(args.config).to_dict(), indent=2))
        return
    config = None
    if args.command != 'receive':
        config = load_config(args.config)
        if args.byte_fraction is not None:
            config = replace(config, byte_fraction=args.byte_fraction)
        config.validate()
        seed = secrets.randbits(32) if args.seed is None else args.seed
        settings = json.loads(Path(args.settings).read_text()) if args.settings else None
        if args.method == 'flowshift':
            settings = settings or json.loads(PROFILE.read_text())
            # The sender knows its own service: the dataset class of this workload.
            kind = load_document(args.config)['workload_type']
            settings['service'] = settings['workload_services'][kind]
        controller = POLICIES[args.method](config, settings, seed)
    if args.command == "receive":
        stats = receive_once(
            args.bind,
            args.port,
            args.output_dir,
            play=args.play,
        )
    elif args.command == "send":
        experiment = load_document(args.config)
        if experiment.get("workload_type") != "file-transfer":
            raise ValueError("configuration is not for file transfer")
        source = Path(args.file)
        expected_size = int(experiment["size_bytes"])
        if not source.is_file() or source.stat().st_size != expected_size:
            raise ValueError(
                f"configuration requires a {expected_size}-byte input file"
            )
        stats = send_file(
            args.host,
            args.port,
            args.file,
            config,
            bind_address=args.bind,
            controller=controller,
        )
    else:
        experiment = load_document(args.config)
        if experiment.get("workload_type") != "streaming":
            raise ValueError("configuration is not for streaming")
        stats = send_video(
            args.host,
            args.port,
            args.video,
            config,
            int(experiment["duration_seconds"]),
            int(experiment["width"]),
            int(experiment["height"]),
            int(experiment["video_bitrate_kbps"]),
            int(experiment["audio_bitrate_kbps"]),
            bind_address=args.bind,
            controller=controller,
        )
    if isinstance(stats, dict):
        stats['policy_seed'] = seed
    print(json.dumps(stats if isinstance(stats, dict) else stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
