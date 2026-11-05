"""Freeze inputs and run a named, bounded answer regression profile.

No implicit builds, checkpoint discovery, GPU execution, privilege escalation or
retries. Each run keeps its own exclusive evidence directory and frozen tooling.
"""

import argparse
from pathlib import Path
import subprocess
import sys

from .profiles import PROFILE_NAMES, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, C2048_PROFILES, profile_modes


def evidence_directory(value):
    path = Path(value)
    if path.is_symlink():
        raise ValueError("evidence directory must not be a symlink")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("evidence path is not a directory")
    return path


def run_frozen(evidence, command, stage=None):
    # Validate before executing the snapshot, then let the frozen supervisor
    # independently enforce identities, actual limits and prerequisites again.
    from .gate_common import dependency_identity, gate_identity, provenance_identity, approved_contract
    from .gate_common import client_identity

    root = evidence_directory(evidence)
    gate_identity(root)
    dependency_identity(root)
    provenance_identity(root)
    client_identity(root)
    profile = approved_contract(root)['profile']
    if command == "run" and stage in profile_modes(profile):
        arguments = [str(root / "run-gate.py"), stage]
    elif command == "closeout" and stage is None:
        arguments = [str(root / "closeout.py")]
    else:
        raise ValueError("unknown frozen command or stage")
    result = subprocess.run([sys.executable, *arguments], check=False)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ui_prepare = commands.add_parser('prepare-ui-client', help='verify prepared UI artifacts and freeze identity; never execute them')
    for name in ('prepared', 'antimatter-repo', 'electron-dist', 'metadata'):
        ui_prepare.add_argument('--' + name, type=Path, required=True)
    prepare = commands.add_parser("prepare", help="freeze explicit inputs; never run the model")
    for name, description in (
        ("repo", "Beholdd Git checkout to snapshot"),
        ("binary", "already built HAG-enabled beholdd executable"),
        ("model", "supported Qwen3-0.6B-Q8_0 checkpoint"),
        ("device", "explicit /dev/dri/renderD… path for the native stages"),
        ("evidence", "new evidence directory; must not already exist"),
    ):
        prepare.add_argument("--" + name, type=Path, required=True, help=description)
    prepare.add_argument("--profile", choices=PROFILE_NAMES, default="list-c128",
                         help="finite workload and qualification strategy (default: list-c128)")
    prepare.add_argument("--prompt-inspector", type=Path, help="required native metadata inspector for generated and UI profiles")
    prepare.add_argument('--ui-client-identity', type=Path, help='verified local UI identity for a UI profile')
    prepare.add_argument('--ui-client-host', choices=('same', 'separate'), default='separate',
                         help='UI profiles only: same backend host or a separate host (default: separate)')
    prepare.add_argument("--runtime-phase-diagnostics", action="store_true",
                         help="freeze non-qualifying HAG answers phase receipts for the checkpoint-rendered history profile")
    prepare.add_argument("--antimatter-repo", type=Path, help="optional committed Antimatter client repository")
    prepare.add_argument("--antimatter-commit", help="supported immutable Antimatter commit")
    prepare.add_argument("--node", type=Path, help="explicit native Node executable to freeze")
    prepare.add_argument("--esbuild", type=Path, help="explicit native esbuild executable for preparation")
    run = commands.add_parser("run", help="execute one contained stage without retries")
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--stage", choices=("self-check", "cpu-reference", "answers", "timing"), required=True)
    closeout = commands.add_parser("closeout", help="verify finished stages and stopped crash capture")
    closeout.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'prepare-ui-client':
            from .ui_client import freeze_ui_identity
            freeze_ui_identity(arguments.prepared, arguments.antimatter_repo, arguments.electron_dist, arguments.metadata)
            print(f'Frozen local UI identity: {arguments.metadata.resolve()}')
            print('No model, Electron process or network request started.')
            return 0
        if arguments.command == "prepare":
            from .inputs import prepare as freeze
            from .antimatter import validate_options
            client_options = {name: getattr(arguments, name) for name in
                              ('antimatter_repo', 'antimatter_commit', 'node', 'esbuild')}
            if arguments.profile in UI_PROFILES:
                if any(value is not None for value in client_options.values()):
                    raise ValueError('UI identity cannot combine with legacy provider options')
                has_client = False
            else:
                has_client = validate_options(*client_options.values(), arguments.profile)
            from .gate_common import phase_diagnostic_contract, validate_ui_client_host
            validate_ui_client_host(arguments.profile, arguments.ui_client_host)
            phase_diagnostic_contract(arguments.profile, arguments.runtime_phase_diagnostics)
            if (arguments.profile in (GENERATED_HISTORY_PROFILE, *UI_PROFILES)) != (arguments.prompt_inspector is not None):
                raise ValueError('prompt inspector is required only for generated and UI conversation profiles')
            inspector_options = {'prompt_inspector': arguments.prompt_inspector} if arguments.prompt_inspector is not None else {}
            diagnostic_options = {'runtime_phase_diagnostics': True} if arguments.runtime_phase_diagnostics else {}
            topology_options = {'ui_client_host': arguments.ui_client_host} if arguments.profile in UI_PROFILES else {}
            freeze(arguments.repo, arguments.binary, arguments.model, arguments.device, arguments.evidence,
                   profile=arguments.profile, **(client_options if has_client else {}), **diagnostic_options, **inspector_options, **topology_options,
                   **({'ui_client_identity': arguments.ui_client_identity} if arguments.ui_client_identity is not None else {}))
            print(f"Prepared exclusive evidence: {arguments.evidence.resolve()}")
            print(f"Profile: {arguments.profile}")
            if arguments.runtime_phase_diagnostics:
                print("Runtime phase diagnostics: HAG answers only; raw logs are not qualifying evidence. All workload limits are unchanged.")
            if arguments.profile in C2048_PROFILES:
                print("CPU HTTP deadline: 600s. Arm a fresh 1500s watcher only after CPU retirement.")
                print("Every stage requires Tctl at most 60C for ten seconds before launch; the running cutoff remains 90C.")
            if arguments.profile in (PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE):
                print("Native provider client deadline: 240s.")
            if arguments.profile == GENERATED_HISTORY_PROFILE:
                print("Generated-history stages: self-check, cpu-reference, answers; two genuine chat requests per backend, no SSE or timing.")
                print("First prompt exactly 467 tokens; second prompt at most 491; each reply allowance 128. Metadata inspection precedes each CPU turn.")
                print("Native provider client deadline: 240s. Exact raw first-answer history and per-turn CPU parity are required.")
            if arguments.profile == UI_HISTORY_PROFILE:
                print('UI-history stages: self-check, current-wire CPU reference, then the native loopback admission broker.')
                if arguments.ui_client_host == 'same':
                    print('Start the separately contained frozen UI on this host after answers-ui-ready.json appears; its frozen URL connects directly.')
                    print('Backend and UI memory caps total 4 GiB across their two units, with separate HAG device accounting.')
                else:
                    print('Connect the separately contained frozen local UI through an explicit SSH loopback tunnel after answers-ui-ready.json appears.')
                print('Two requests only; first prompt467, second at most491, M128; parity and healthy reset precede each buffered SSE delivery.')
                print('Full closeout also requires imported host-qualified UI report, process and retirement receipts.')
            if arguments.profile == UI_STOP_PROFILE:
                print('UI-stop stages: self-check, one current-wire CPU reuse reference, then two native UI requests.')
                print('The real UI stops the M64 count after visible numeric content; a fresh conversation checks M16 ready parity on the same server.')
                if arguments.ui_client_host == 'same':
                    print('Start the separately contained frozen UI on this host after answers-ui-ready.json appears; its frozen URL connects directly.')
                    print('Backend and UI memory caps total 4 GiB across their two units, with separate HAG device accounting.')
                else:
                    print('Connect the separately contained frozen local UI through an explicit SSH loopback tunnel after answers-ui-ready.json appears.')
                print('Both prompts require metadata inspection before inference. UI deadline 355s; each request deadline 240s.')
                print('Full closeout requires cancellation/reset/reuse evidence and imported host-qualified UI process retirement.')
            if arguments.profile == HISTORY_PROFILE:
                print("Seeded first-turn history, not a live two-turn conversation; exact prompt 495 tokens, reply allowance 128.")
            if arguments.profile == TEMPLATE_HISTORY_PROFILE:
                print("Checkpoint-rendered seeded history, not a live two-turn conversation; exact prompt 491 tokens, reply allowance 128.")
            if arguments.profile == LIFECYCLE_PROFILE:
                print("Lifecycle stages: self-check, cpu-reference, answers, then explicit watcher stop and closeout; no timing stage.")
                print("Length-limited TCP parity, cancellation, reuse and unload/reload; no complete-answer or semantic qualification.")
            print("No model executed. Run self-check, then cpu-reference before starting native work.")
            return 0
        return run_frozen(arguments.evidence, arguments.command, getattr(arguments, "stage", None))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"HAG qualification: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
