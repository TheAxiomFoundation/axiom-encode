"""Paired, randomized local encoder A/B supervisor (macOS, subscription logins).

Usage (run under caffeinate on AC power):

    LOCAL_AB_PINS=<pins dir> caffeinate -dimsu python3 -I run_ab.py \\
        --out <run dir> --citations <citations.txt> \\
        --arm main=<worktree> --arm branch=<worktree> \\
        [--reps 3] [--parallel 3] [--seed 20261020] [--timeout-min 75]

Every citation/rep is one group: all arms start at the same moment, so load
and time of day are shared across arms. Rounds are shuffled independently
with ``random.Random(seed)``; orders are written to ``order_r<N>.txt``.

Outputs in ``--out``: ``runs/<arm>/r<N>/<slug>/`` (one encode each),
``pairs.jsonl`` (finished groups), ``retries.jsonl`` (infra-discarded groups),
``discarded/``, ``progress.md``, ``load.log`` and ``interim-25pct.md`` /
``interim-50pct.md`` / ``interim-75pct.md`` (descriptive snapshots written when
that share of groups is done). Nothing is deleted: a run dir that already
exists is renamed to ``<dir>.discarded-<timestamp>`` by run_one.sh, and
infra-discarded groups move to ``discarded/``. With ``--archive-dir`` the whole
``--out`` tree is rsynced (copy only, replaced files kept under
``_versions/``) to a durable directory after every group and at the end; the
directory must be outside git work trees and iCloud Drive.

Infra handling: a group with a network or rate-limit error in codex output,
or a system sleep during the group (``pmset -g log``), is discarded for every
arm and retried later in the same round (max ``--max-infra-retries``). A
timeout is measured on the wall clock and polled, because
``Popen.wait(timeout)`` uses a monotonic clock that stops while macOS sleeps.
Timeouts are recorded (``"timeout": true`` in run.json), not retried.

Stop at a clean group boundary with ``touch <out>/STOP``. Ctrl-C or SIGTERM
kills the running encodes of every group and exits. Re-running the same
command removes STOP and resumes: finished and given-up groups in
``pairs.jsonl`` are skipped. Do not run two supervisors on one ``--out``.

Infra detection reads codex ``error=`` lines in stdout.log, codex JSON error
events in leftover temp files, and stderr.log. Errors that the ``claude``
reviewer CLI reports inside a completed encode are not detected.
"""

import argparse
import datetime
import json
import os
import random
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFRA_PATTERNS = [
    (
        re.compile(
            r"nodename nor servname|failed to lookup address|Network is unreachable"
            r"|waiting for network|Connection refused|error sending request",
            re.I,
        ),
        "network",
    ),
    (
        re.compile(r"rate.?limit|usage limit|Too Many Requests|\b429\b|quota", re.I),
        "rate_limit",
    ),
]
lock = threading.Lock()


def slug(citation: str) -> str:
    return citation.replace("/", "_").replace(":", "_")


class Supervisor:
    def __init__(self, args):
        self.out = Path(args.out).resolve()
        self.arms = dict(item.split("=", 1) for item in args.arm)
        self.args = args
        self.out.mkdir(parents=True, exist_ok=True)
        self.pause_until = 0.0
        self.live = {}  # pid -> process group of running encodes

    # -- logging -----------------------------------------------------------
    def log_jsonl(self, name, rec):
        with lock, open(self.out / name, "a") as handle:
            handle.write(json.dumps(rec) + "\n")

    def progress(self, line):
        with lock, open(self.out / "progress.md", "a") as handle:
            handle.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")

    # -- environment checks ------------------------------------------------
    @staticmethod
    def on_battery() -> bool:
        try:
            out = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=20
            ).stdout
        except Exception:
            return False
        return "Battery Power" in out

    @staticmethod
    def net_ok() -> bool:
        try:
            r = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "--max-time",
                    "10",
                    "https://chatgpt.com/",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return r.stdout.strip() not in ("", "000")
        except Exception:
            return False

    def wait_ready(self):
        while time.time() < self.pause_until and not (self.out / "STOP").exists():
            time.sleep(30)
        if self.on_battery():
            self.progress(
                "WARNING on battery power; caffeinate cannot prevent clamshell sleep"
            )
        while not self.net_ok():
            self.progress("network down; pausing 60 s")
            time.sleep(60)

    @staticmethod
    def slept_between(t0, t1):
        try:
            out = subprocess.run(
                ["pmset", "-g", "log"], capture_output=True, text=True, timeout=120
            ).stdout
        except Exception:
            return None
        for line in out.splitlines():
            if "Entering Sleep state" not in line or " Sleep " not in line[:60]:
                continue
            try:
                ts = datetime.datetime.strptime(
                    line[:25], "%Y-%m-%d %H:%M:%S %z"
                ).timestamp()
            except ValueError:
                continue
            if t0 <= ts <= t1:
                return line[:160]
        return None

    @staticmethod
    def infra_errors(run_dir: Path):
        # Codex errors reach stdout.log as `error=...` lines (the runner deletes
        # its temp files after each completed call); only those lines are
        # scanned there, so source text in the log cannot match.
        stdout = run_dir / "stdout.log"
        if stdout.is_file():
            for line in stdout.read_text(errors="replace").splitlines():
                if line.strip().startswith("error="):
                    for rx, kind in INFRA_PATTERNS:
                        if rx.search(line):
                            return kind, line.strip()[:300]
        files = list((run_dir / "tmp").glob("tmp*")) + [run_dir / "stderr.log"]
        for path in files:
            if not path.is_file():
                continue
            for line in path.read_text(errors="replace").splitlines():
                text = None
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        if event.get("type") == "error":
                            text = str(event.get("message", ""))
                        elif (
                            isinstance(event.get("item"), dict)
                            and event["item"].get("type") == "error"
                        ):
                            text = str(event["item"].get("message", ""))
                except ValueError:
                    text = line
                if not text:
                    continue
                for rx, kind in INFRA_PATTERNS:
                    if rx.search(text):
                        return kind, text[:300]
        return None

    # -- one group -----------------------------------------------------------
    def run_group(self, job):
        cit, idx, retry = job["citation"], job["run_index"], job["infra_retry"]
        self.wait_ready()
        if (self.out / "STOP").exists():
            return "stopped"
        outs = {
            arm: self.out / "runs" / arm / f"r{idx}" / slug(cit) for arm in self.arms
        }
        t0 = time.time()
        procs = {}
        for arm, wt in self.arms.items():
            # Register each encode under the lock as it starts, so an
            # interrupt can always kill it.
            with lock:
                proc = subprocess.Popen(
                    [str(HERE / "run_one.sh"), arm, wt, cit, str(outs[arm])],
                    start_new_session=True,
                )
                self.live[proc.pid] = proc
            procs[arm] = proc
        status = {}
        deadline = t0 + self.args.timeout_min * 60
        for arm, proc in procs.items():
            while proc.poll() is None and time.time() < deadline:
                time.sleep(15)
            if proc.poll() is not None:
                status[arm] = "done"
                continue
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except OSError:
                    pass
                time.sleep(10)
            proc.wait()
            status[arm] = "timeout"
        with lock:
            for proc in procs.values():
                self.live.pop(proc.pid, None)
        infra = {}
        slept = self.slept_between(t0, time.time())
        if slept:
            infra["system"] = ("sleep", slept)
        for arm in self.arms:
            rj = outs[arm] / "run.json"
            rec = (
                json.loads(rj.read_text())
                if rj.exists()
                else {
                    "side": arm,
                    "citation": cit,
                    "rc": -9,
                    "wall_s": int(time.time() - t0),
                }
            )
            rec.update(
                run_index=idx, status=status[arm], infra_retry=retry, pair_started=t0
            )
            if status[arm] == "timeout":
                rec["timeout"] = True
            found = self.infra_errors(outs[arm]) or infra.get("system")
            if found:
                infra[arm] = found
                rec["infra_error"] = {"kind": found[0], "sample": found[1]}
            outs[arm].mkdir(parents=True, exist_ok=True)
            rj.write_text(json.dumps(rec))
        if infra:
            disc = (
                self.out
                / "discarded"
                / f"{slug(cit)}-r{idx}-try{retry}-{int(time.time())}"
            )
            disc.mkdir(parents=True, exist_ok=True)
            for arm in self.arms:
                if outs[arm].exists():
                    shutil.move(str(outs[arm]), str(disc / arm))
            self.log_jsonl(
                "retries.jsonl",
                {
                    "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "citation": cit,
                    "run_index": idx,
                    "infra_retry": retry,
                    "errors": {k: list(v) for k, v in infra.items()},
                },
            )
            kinds = sorted({v[0] for v in infra.values()})
            self.progress(f"RETRY {cit} r{idx} (infra {kinds}, try {retry})")
            if "rate_limit" in kinds:
                self.progress("rate limit; pausing all workers for 30 min")
                with lock:
                    self.pause_until = max(self.pause_until, time.time() + 1800)
            return "retry"
        summary = []
        for arm in self.arms:
            rec = json.loads((outs[arm] / "run.json").read_text())
            state = "TIMEOUT" if rec.get("timeout") else f"rc={rec['rc']}"
            summary.append(f"{arm} {state} {round(rec['wall_s'] / 60)}m")
        self.log_jsonl(
            "pairs.jsonl",
            {
                "citation": cit,
                "run_index": idx,
                "infra_retry": retry,
                "status": status,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        self.progress(f"pair done: {cit} r{idx} | " + " | ".join(summary))
        self.after_group()
        return "ok"

    # -- durability ------------------------------------------------------------
    def after_group(self):
        """Write interim snapshots at 25/50/75% and copy the run to the archive."""

        with lock:
            done = sum(
                1
                for line in open(self.out / "pairs.jsonl")
                if isinstance(json.loads(line).get("status"), dict)
            )
            due = [
                pct
                for pct in (25, 50, 75)
                if done >= self.total_groups * pct / 100 and pct not in self.snapshots
            ]
            self.snapshots.update(due)
        for pct in due:
            arms = list(self.arms)
            result = subprocess.run(
                [
                    "python3",
                    "-I",
                    str(HERE / "analyze_ab.py"),
                    str(self.out),
                    "--baseline",
                    arms[0],
                    "--treatment",
                    arms[1],
                    "--no-write",
                ],
                capture_output=True,
                text=True,
            )
            (self.out / f"interim-{pct:02d}pct.md").write_text(
                f"Interim snapshot at {done}/{self.total_groups} groups, "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')}. Descriptive only; the "
                "pre-registered analysis runs once, on the complete run.\n\n"
                + result.stdout
            )
            self.progress(f"interim snapshot {pct}% written ({done} groups)")
        self.archive()

    def archive(self):
        if not self.args.archive_dir:
            return
        dest = Path(self.args.archive_dir)
        dest.mkdir(parents=True, exist_ok=True)
        # Copy only; never delete in the archive. Replaced files are kept
        # under _versions/<timestamp>/.
        subprocess.run(
            [
                "rsync",
                "-a",
                "--backup",
                f"--backup-dir={dest / '_versions' / time.strftime('%Y%m%dT%H%M%S')}",
                f"{self.out}/",
                f"{dest}/",
            ],
            capture_output=True,
        )

    # -- queue ---------------------------------------------------------------
    def build_queue(self):
        cits = [
            line.strip()
            for line in open(self.args.citations)
            if line.strip() and not line.startswith("#")
        ]
        rng = random.Random(self.args.seed)
        queue = []
        for idx in range(1, self.args.reps + 1):
            order = cits[:]
            rng.shuffle(order)
            (self.out / f"order_r{idx}.txt").write_text("\n".join(order) + "\n")
            queue += [
                {"citation": c, "run_index": idx, "infra_retry": 0} for c in order
            ]
        done, retried = set(), {}
        if (self.out / "pairs.jsonl").exists():
            for line in open(self.out / "pairs.jsonl"):
                rec = json.loads(line)
                if (
                    isinstance(rec.get("status"), dict)
                    or rec.get("status") == "infra_gave_up"
                ):
                    done.add((rec["citation"], rec["run_index"]))
        if (self.out / "retries.jsonl").exists():
            for line in open(self.out / "retries.jsonl"):
                rec = json.loads(line)
                key = (rec["citation"], rec["run_index"])
                retried[key] = max(retried.get(key, -1), rec["infra_retry"])
        return [
            {
                **job,
                "infra_retry": retried.get((job["citation"], job["run_index"]), -1) + 1,
            }
            for job in queue
            if (job["citation"], job["run_index"]) not in done
        ], len(done)

    def load_logger(self, stop):
        while not stop.is_set():
            up = subprocess.run(
                ["uptime"], capture_output=True, text=True
            ).stdout.strip()
            with open(self.out / "load.log", "a") as handle:
                handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {up}\n")
            stop.wait(600)

    def kill_live(self, *_args):
        with lock:
            live = list(self.live.values())
        for proc in live:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except OSError:
                    pass
        self.progress(
            f"INTERRUPTED: killed {len(live)} running encodes; resume with the same command"
        )
        os._exit(130)

    def run(self):
        self.snapshots = set()
        stop_file = self.out / "STOP"
        if stop_file.exists():
            stop_file.unlink()
        signal.signal(signal.SIGINT, self.kill_live)
        signal.signal(signal.SIGTERM, self.kill_live)
        queue, skipped = self.build_queue()
        self.total_groups = len(queue) + skipped
        self.progress(
            f"START seed={self.args.seed} arms={list(self.arms)} jobs={len(queue)} "
            f"(skipped {skipped} completed) parallel={self.args.parallel}"
        )
        stop = threading.Event()
        threading.Thread(target=self.load_logger, args=(stop,), daemon=True).start()
        qlock = threading.Lock()

        def worker():
            while not (self.out / "STOP").exists():
                with qlock:
                    if not queue:
                        return
                    job = queue.pop(0)
                try:
                    result = self.run_group(job)
                except Exception as exc:  # keep the other groups running
                    self.progress(
                        f"SUPERVISOR ERROR on {job['citation']} r{job['run_index']}: {exc!r}"
                    )
                    continue
                if result == "stopped":
                    with qlock:
                        queue.insert(0, job)
                    return
                if result != "retry":
                    continue
                if job["infra_retry"] + 1 > self.args.max_infra_retries:
                    self.progress(f"GIVE UP {job['citation']} r{job['run_index']}")
                    self.log_jsonl("pairs.jsonl", {**job, "status": "infra_gave_up"})
                    continue
                job = {**job, "infra_retry": job["infra_retry"] + 1}
                with qlock:
                    pos = (
                        max(
                            [
                                i
                                for i, j in enumerate(queue)
                                if j["run_index"] == job["run_index"]
                            ],
                            default=-1,
                        )
                        + 1
                    )
                    queue.insert(pos, job)

        threads = [threading.Thread(target=worker) for _ in range(self.args.parallel)]
        for thread in threads:
            thread.start()
            time.sleep(5)
        for thread in threads:
            thread.join()
        stop.set()
        self.progress(
            f"END remaining_jobs={len(queue)} stop_file={(self.out / 'STOP').exists()}"
        )
        self.archive()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--citations", required=True)
    parser.add_argument(
        "--arm", action="append", required=True, help="name=worktree (repeat; 2+ arms)"
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--parallel", type=int, default=3, help="groups run at the same time"
    )
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--timeout-min", type=float, default=75)
    parser.add_argument("--max-infra-retries", type=int, default=3)
    parser.add_argument(
        "--archive-dir",
        help="durable copy of --out after every group (outside git and iCloud)",
    )
    args = parser.parse_args()
    if args.archive_dir:
        archive = Path(args.archive_dir).expanduser().resolve()
        if "Mobile Documents" in str(archive) or "CloudDocs" in str(archive):
            parser.error("--archive-dir must not be in iCloud Drive")
        probe = archive
        while not probe.exists():
            probe = probe.parent
        inside_git = (
            subprocess.run(
                ["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            == "true"
        )
        if inside_git:
            parser.error("--archive-dir must not be inside a git work tree")
        args.archive_dir = str(archive)
    if len(args.arm) < 2:
        parser.error("need at least two --arm name=worktree")
    if not os.environ.get("LOCAL_AB_PINS"):
        parser.error("set LOCAL_AB_PINS")
    Supervisor(args).run()


if __name__ == "__main__":
    main()
