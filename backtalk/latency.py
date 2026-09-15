# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DIAGNOSTIC LATENCY TELEMETRY — not part of the voice pipeline's
behavior. This module only ever logs; it never influences timing,
buffering, cancellation, or any decision the voice line makes. Every
public function is wrapped so a bug in here can never break a turn.

One PTT turn = one turn_start() at key release, some number of mark()
calls as that turn's stages complete, and exactly one turn_end() when
the turn is genuinely done (audio finished, the turn came back empty,
or it was interrupted). Marks after turn_end() are dropped rather than
misattributed to whatever turn happens to be "current" by then.

Clock: time.perf_counter(), monotonic and high-resolution. Every line
is prefixed [lat#N] where N is a turn id that increments on every
turn_start(), so a turn's marks can be grepped out unambiguously even
if two turns' log lines interleave.
"""
import threading
import time

from backtalk.vlog import log

_lock = threading.Lock()
_turn_id = 0
_t0 = None
_last = None
_seen: set[str] = set()
_active = False


def turn_start():
    """Call once, at PTT release. Starts a new turn's clock and id, and
    RETURNS that id as this turn's immutable token.

    Ownership contract: the caller must carry this exact return value
    forward through whatever it does next (recording, dispatch,
    synthesis, playback) and present it back to every mark()/turn_end()
    call for this turn. Nothing downstream may recover or infer its
    token any other way -- in particular, never by reading whatever
    turn happens to be current at call time, which is exactly the bug
    this replaces (a late-arriving call from turn N reading turn N+1's
    id off shared state and getting misattributed to it)."""
    global _turn_id, _t0, _last, _seen, _active
    try:
        with _lock:
            _turn_id += 1
            _t0 = time.perf_counter()
            _last = _t0
            _seen = set()
            _active = True
            tid = _turn_id
        log(f"[lat#{tid}] turn_start (ptt release)")
        return tid
    except Exception:
        return None


def current_token():
    """The active turn's token right now, or None.

    Only safe to call from a spot that is PROVABLY sequential with the
    one call site that mints tokens (ears.record_held, via turn_start):
    right after that call returns control to its caller, on the same
    single-threaded dispatch path, before anything else in the process
    could possibly have started a newer turn. It is not a general
    "ask which turn owns me" API -- nothing else should call this to
    recover ownership after the fact."""
    try:
        with _lock:
            return _turn_id if _active else None
    except Exception:
        return None


def mark(label: str, token):
    """Log one stage for the turn identified by `token`, once per turn
    per label. Silently does nothing if: no turn has started yet, that
    turn already ended, a NEWER turn has since started (token no longer
    equals the current turn id -- this is the ownership check: a stale
    caller can never attribute its own late-arriving call to whatever
    turn happens to be current by the time it runs), or this label
    already fired for this turn (later sentences of the same reply
    re-hit the same instrumentation points; only the first counts,
    matching how the existing "to first" log behaves). `token=None`
    can never match a real turn (ids start at 1), so untracked callers
    (the startup warmup ping, typed/console/greeting speech) are always
    a safe no-op."""
    global _last
    try:
        with _lock:
            if _t0 is None or not _active or token != _turn_id \
                    or label in _seen:
                return
            _seen.add(label)
            now = time.perf_counter()
            since_start = now - _t0
            since_last = now - _last
            _last = now
            tid = _turn_id
        log(f"[lat#{tid}] {label}: +{since_last:.3f}s (t={since_start:.3f}s)")
    except Exception:
        pass


def turn_end(reason: str, token):
    """Close the turn identified by `token`: completed, cancelled, or
    empty (a turn that produced no sentences). Same ownership check as
    mark() -- a stale completion/cancellation from a turn that is no
    longer current is dropped, never attributed to whatever turn is
    active now. Idempotent and safe to call from a cancellation
    handler -- it only logs and flips a flag, so it cannot change what
    cancellation itself does."""
    global _active
    try:
        with _lock:
            if _t0 is None or not _active or token != _turn_id:
                return
            now = time.perf_counter()
            total = now - _t0
            tid = _turn_id
            _active = False
        log(f"[lat#{tid}] turn_end ({reason}): total={total:.3f}s")
    except Exception:
        pass
