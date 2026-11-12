"""
StrategyBot - Zone Rush bot (team build).

Single file, Python standard library only.  Entry point: act(state).

Team build = bot_dev plus:
  - a per-opponent shield-reaction model: each owner's P(shield) is learned
    for L1 cells, L2 cells with 0 stays, and L2 cells with 1+ stays. L2 raids
    are valued by walking their stays against the owner's 4-turn cooldown.
    Hopeless L2 raids are refused, and after a catch the bot backs off that
    (owner, cell), which ends the raid-freeze loop;
  - rule-robustness fixes from a README/rulebook audit: own shield cooldown,
    STAY while frozen, penalty freezes not mistaken for shields, and a rival's
    backfire on our land not blamed on us.
Frozen turns never build take/upgrade progress (rule confirmed by the team,
2026-11-10), so the in-game detection of the opposite reading and
the probe that served it are switched off (fz_detect, fz_probe).
Results and commands: runs/team_bot_log.md.

Layout of this file
  1. PARAMS          - every tunable weight / threshold in one place
  2. Snap            - reads `state` defensively into flat arrays (never mutates it)
  3. Memory          - per-game globals: turn counter, scores, progress, shields,
                       freezes, token order, rival behaviour statistics
  4. Tracker         - rebuilds the hidden information from consecutive snapshots
  5. Planner         - scores capture / raid / upgrade plans (2-step look-ahead)
  6. Shield logic    - expected value of SHIELD this turn
  7. act(state)      - safety shell: always returns a member of legal_actions

Board indexing follows the README: board[row][col], (0, 0) top-left,
NORTH = row - 1.  The bot verifies this from its own moves and adapts if the
server turns out to use a different orientation.
"""

import math
import os
import sys
import time

TOTAL_TURNS = 80
MOVES = ("NORTH", "EAST", "SOUTH", "WEST")
OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}
README_DIRS = {"NORTH": (-1, 0), "SOUTH": (1, 0), "WEST": (0, -1), "EAST": (0, 1)}

# Plan kinds
NEUTRAL, RAID1, RAID2, UPGRADE = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# 1. Parameters (tools/ may override them through PARAMS_OVERRIDE)
# ---------------------------------------------------------------------------
PARAMS = {
    # time budget per call (ms); the server allows 200 ms (2 s on turn 1)
    "budget_ms": 110.0,
    # Elo weighting of rivals
    "sigma_base": 30.0,
    "sigma_per_turn": 5.0,
    "w_uniform_mix": 0.35,
    # races for neutral cells
    "race_ahead": 0.55,
    "race_tie_lose": 0.35,
    "race_tie_win": 0.05,
    "race_value_scale": 0.5,
    # raids
    "rho_prior": 0.45,          # P(owner shields | a rival stays on its land)
    "rho_prior_n": 2.0,         # strength of that prior (pseudo-observations)
    "raid_caught_delay": 3.0,   # turns lost when a raid is shielded
    "raid2_caught_delay": 3.0,
    "bait_value": 25.0,         # value of making a shield-happy owner backfire
    # upgrades
    "upg_def_frac": 0.0,        # extra defence value of upgrading a cell near rivals (x v*G)
    "upg_def_radius": 4,
    # cell durability: learned per-turn raid hazard for level-1 / level-2 cells
    "lam1_prior": 0.045,        # tuned: 0.01 << 0.02 < 0.03 < 0.045 ~ 0.06
    "lam2_prior": 0.004,
    "lam2_prior_fz": 0.012,     # when frozen stays build progress, shields defend L2 poorly
    # Once the frozen-progress rule is detected, only the raid valuation switches
    # to its fz form. Switching the L2 hazard and the shield value too was measured
    # worse (runs/improve_log.md, E3/E5), so both are off.
    "fz_hazard": False,
    "fz_shield": False,
    # Frozen stays never build progress (rule confirmed), so never switch to
    # the fz reading: a false detection would turn the shield-reaction model off.
    "fz_detect": False,
    "lam_prior_n": 300.0,       # prior strength in cell-turns
    "use_life": True,
    "expo_k": 0.0,              # hazard multiplier slope by distance to nearest rival bot
    "expo_r": 2.5,
    # shield decision
    "q_prior": 0.55,            # P(rival standing on my cell stays this turn)
    "q_prior_n": 2.0,
    "q_committed": 0.85,        # P(stay) once it has already stayed on my L2 cell
    "shield_giveup": 0.25,      # chance a frozen L1 raider gives up afterwards
    "shield_recatch": 0.9,      # chance we can catch an L2 raider again later
    "shield_early_frac": 0.5,   # value of catching an L2 raider before its last stay
    "rival_rate": 14.0,         # value per turn of a rival's time (for freezes)
    "backfire_turns": 2.0,
    "cooldown_cost": 6.0,
    "shield_margin": 0.0,
    # planning
    "chain_top": 14,
    "chain_top2": 6,
    "chain_depth": 2,           # plans per chain (1 = greedy rate); 3 tested worse
    "hysteresis": 0.10,
    "contest_block_p": 0.3,     # P(an adjacent rival moves into the same cell)
    "home_upgrade": False,      # never plan to upgrade the home cell
    # --- F1: per-opponent shield reaction (default rules only) ---------------
    # P(owner shields | non-frozen rival on its non-home cell, shield ready),
    # learned per owner in 3 buckets: L1 cell, L2 with 0 stays, L2 with 1+ stays.
    "react_model": True,        # accepted: runs/team_bot_log.md (F1)
    "r1_prior": 0.45,
    "r20_prior": 0.15,
    "r2k_prior": 0.60,
    "react_n": 2.0,             # prior strength (pseudo-observations)
    "react_veto": 0.6,          # refuse an exposed L2 raid when the 1+-stay rate >= this ...
    "react_veto_n": 3,          # ... and the owner has been seen this many times in that bucket
    "backoff_turns": 80,        # after a catch on (owner, L2 cell): assume it always shields there
    "backoff_h": 0.95,
    "fz_probe": False,          # stay once on the first free turn after an L2 catch (see _fz_probe);
                                # only useful if frozen stays built progress, which they don't
    # --- F2: raid hazard that follows the game phase -------------------------
    "hazard_mode": 0,           # 0 = flat lam priors (bot_dev); 1 = phase profile (rejected, F2)
    "h1_prof": (0.004, 0.012, 0.014, 0.021),   # L1 per-turn hazard at turns 10/30/50/70
    "h2_prof": (0.001, 0.002, 0.002, 0.004),   # L2
    "h_prior_n": 4.0,           # prior strength of the learned scale, in expected raids
    "h1_mult": 1.0,
    "h2_mult": 1.0,
}
PARAMS_OVERRIDE = {}
P = dict(PARAMS)
P.update(PARAMS_OVERRIDE)


def _profile(points):
    """Per-turn hazard for turns 0..80, linear between the values at turns 10/30/50/70."""
    xs = (10.0, 30.0, 50.0, 70.0)
    out = [0.0] * (TOTAL_TURNS + 2)
    for u in range(TOTAL_TURNS + 2):
        if u <= xs[0]:
            v = points[0]
        elif u >= xs[-1]:
            v = points[-1]
        else:
            i = int((u - xs[0]) // 20.0)
            f = (u - xs[i]) / 20.0
            v = points[i] + f * (points[i + 1] - points[i])
        out[u] = float(v)
    return out


_PROF = {1: _profile(P["h1_prof"]), 2: _profile(P["h2_prof"])}

_DEBUG = bool(os.environ.get("ZR_DEBUG"))
_MEM = None


# ---------------------------------------------------------------------------
# 2. Snapshot of the state
# ---------------------------------------------------------------------------
def _attr(obj, name, default):
    try:
        v = getattr(obj, name)
    except Exception:
        return default
    return default if v is None else v


class Snap(object):
    """Flat, read-only copy of everything the bot needs from `state`."""

    def __init__(self, state):
        board = state.board
        first = board[0]
        if hasattr(first, "owner") or hasattr(first, "occupant"):
            cells = list(board)                       # flat list of 64
        else:
            cells = [c for row in board for c in row]  # 8 x 8 list of lists
        size = len(cells)
        n = int(round(math.sqrt(size)))
        self.n = n
        self.size = size
        self.occ = [int(_attr(c, "occupant", 0)) for c in cells]
        self.own = [int(_attr(c, "owner", 0)) for c in cells]
        self.lvl = [2 if _attr(c, "is_upgraded", False) else 1 for c in cells]
        self.home = [bool(_attr(c, "is_home", False)) for c in cells]
        self.frz = [bool(_attr(c, "is_frozen", False)) for c in cells]
        self.val = [int(_attr(c, "base_points", 1)) for c in cells]
        self.pos = {}
        for i, b in enumerate(self.occ):
            if b:
                self.pos[b] = i
        self.me = int(state.bot_id)
        self.token = bool(_attr(state, "is_priority_token", False))
        raw = list(_attr(state, "legal_actions", ["STAY"])) or ["STAY"]
        self.legal_map = {}
        for a in raw:
            self.legal_map.setdefault(str(a).strip().upper(), a)
        self.legal = set(self.legal_map)
        self.homes = {}
        for i in range(size):
            if self.home[i] and self.own[i]:
                self.homes[self.own[i]] = i
        self.bots = sorted(self.pos)

    def frozen(self, b):
        p = self.pos.get(b)
        return p is not None and self.frz[p]


# ---------------------------------------------------------------------------
# 3. Per-game memory (globals)
# ---------------------------------------------------------------------------
class Memory(object):
    def __init__(self, snap):
        self.n = snap.n
        self.val = tuple(snap.val)
        self.me = snap.me
        self.turn = 0
        self.prev = None
        self.last_action = None
        self.last_shield_legal = False
        ids = set(snap.bots) | {1, 2, 3, 4}
        self.ids = sorted(ids)
        self.scores = dict((b, 0.0) for b in ids)
        self.prog = dict((b, 0) for b in ids)
        self.frz_run = dict((b, 0) for b in ids)
        self.last_shield = dict((b, -100) for b in ids)
        self.stats = dict((b, {"opp": 0, "sh": 0, "spam": 0, "rv_opp": 0, "rv_stay": 0}) for b in ids)
        self.tok_k = {0, 1, 2, 3}
        self.homes = dict(snap.homes)
        for b, p in snap.pos.items():          # fallback: start cell = home
            if b not in self.homes:
                self.homes[b] = p
        self.dmap = dict(README_DIRS)
        self.target = None
        # rule detection: does a frozen bot's forced STAY build progress?
        self.fz_prog = False
        self.frz_seen_on_cell = {}
        # raid hazard bookkeeping: index = level (1 or 2)
        self.cellturns = [0, 0, 0]
        self.raids = [0, 0, 0]
        self.my_cellturns = [0, 0, 0]
        self.my_raids = [0, 0, 0]
        # phase-hazard bookkeeping: expected raids under the profile (index = level)
        self.exp_raids = [0.0, 0.0, 0.0]
        self.my_exp_raids = [0.0, 0.0, 0.0]
        # F1: react[owner][bucket] = [chances, shields]; bucket 0 = L1, 1 = L2 0 stays, 2 = L2 1+ stays
        self.react = dict((b, [[0, 0], [0, 0], [0, 0]]) for b in ids)
        self.backoff = {}                        # (owner, cell) -> turn we were caught there
        self.blamed = {}                         # frozen bot -> (shielder, its previous last_shield)
        self.inferred = []                       # debug: shields inferred last update


def _manh(n, a, b):
    ar, ac = divmod(a, n)
    br, bc = divmod(b, n)
    return abs(ar - br) + abs(ac - bc)


def _is_new_game(mem, snap):
    if mem is None:
        return True
    if mem.me != snap.me or mem.n != snap.n:
        return True
    if mem.turn >= TOTAL_TURNS and _is_start(snap):
        return True
    if mem.val != tuple(snap.val):
        return True
    prev = mem.prev
    if prev is None:
        return False
    for i in range(snap.size):                 # ownership never reverts to neutral
        if prev.own[i] and not snap.own[i]:
            return True
    return False


def _is_start(snap):
    """Start-of-game board: only homes are owned and every bot is on a home."""
    for i in range(snap.size):
        if snap.own[i] and not snap.home[i]:
            return False
    return all(snap.home[p] for p in snap.pos.values())


def _skipped_turns(mem, snap):
    """Bots move at most one cell a turn: a bigger jump means we were not
    called on some turns. Returns how many extra turns passed (usually 0)."""
    prev = mem.prev
    if prev is None:
        return 0
    jump = 0
    for b, p in snap.pos.items():
        q = prev.pos.get(b)
        if q is not None:
            jump = max(jump, _manh(snap.n, p, q))
    return max(0, jump - 1)


# ---------------------------------------------------------------------------
# 4. Tracker
# ---------------------------------------------------------------------------
def _learn_dirs(mem, prev, cur, action):
    """Check the README direction table against our own observed move."""
    if action not in MOVES:
        return
    a, b = prev.pos.get(mem.me), cur.pos.get(mem.me)
    if a is None or b is None or a == b:
        return
    n = mem.n
    ar, ac = divmod(a, n)
    br, bc = divmod(b, n)
    d = (br - ar, bc - ac)
    if abs(d[0]) + abs(d[1]) != 1 or d == mem.dmap[action]:
        return
    old = mem.dmap[action]
    mem.dmap[action] = d
    mem.dmap[OPPOSITE[action]] = (-d[0], -d[1])
    if (old[0] == 0) != (d[0] == 0):            # axis changed -> transpose the other pair
        for other in MOVES:
            if other != action and other != OPPOSITE[action]:
                r, c = mem.dmap[other]
                mem.dmap[other] = (c, r)


def _token_update(mem, t, me, has_token):
    ks = set(k for k in mem.tok_k if ((((t - 1 + k) % 4) + 1) == me) == has_token)
    if not ks:
        ks = set(k for k in range(4) if ((((t - 1 + k) % 4) + 1) == me) == has_token)
    mem.tok_k = ks


def _holder(mem, u):
    k = 0 if 0 in mem.tok_k else min(mem.tok_k)
    return ((u - 1 + k) % 4) + 1


def _first_in_order(mem, u, a, b):
    """True if seat a comes before seat b in priority order on turn u."""
    h = _holder(mem, u)
    return ((a - h) % 4) < ((b - h) % 4)


def _undo_blame(mem, b):
    """b stayed frozen longer than any shield freezes: its freeze was a penalty,
    so the shield we inferred from it never happened."""
    rec = mem.blamed.pop(b, None)
    if rec is None:
        return
    s, t, prev_last, bkey = rec
    if mem.last_shield.get(s) == t:
        mem.last_shield[s] = prev_last
    if bkey is not None and mem.backoff.get(bkey) == t:
        del mem.backoff[bkey]


def _update(mem, prev, cur):
    """Fold the turn that just resolved (prev -> cur) into memory."""
    t = mem.turn - 1
    me = mem.me
    bots = [b for b in cur.bots if b in prev.pos]
    moved = dict((b, prev.pos[b] != cur.pos[b]) for b in bots)
    was_frz = dict((b, prev.frozen(b)) for b in bots)
    newly = {}
    for b in bots:
        if cur.frozen(b):
            if was_frz[b] and mem.frz_run[b] < 2:
                mem.frz_run[b] += 1
                newly[b] = False
            else:
                if was_frz[b]:
                    # frozen 3+ turns in a row: no shield freezes that long, so it
                    # is a penalty freeze (rulebook 4.4); undo the shield we blamed
                    _undo_blame(mem, b)
                mem.frz_run[b] = 1
                newly[b] = not was_frz[b]
        else:
            mem.frz_run[b] = 0
            newly[b] = False

    # --- who shielded on turn t (every shield freezes somebody) ---
    shielders = set()
    blame = {}
    i_shielded = mem.last_action == "SHIELD" and mem.last_shield_legal
    for b in bots:
        if not newly[b]:
            continue
        x = cur.pos[b]
        o = prev.own[x]
        if o == me and b != me and not i_shielded:
            # only the owner catches on its own land (rulebook 2.6) and we did
            # not shield, so b's own shield backfired on our cell
            shielders.add(b)
            blame[b] = b
        elif moved[b]:                            # a moving bot did not shield
            if o and o != b:
                shielders.add(o)
                blame[b] = o
        elif o and o != b and not newly.get(o, False):
            shielders.add(o)                      # caught on o's land
            blame[b] = o
        else:
            shielders.add(b)                      # its own shield backfired
            blame[b] = b
    if mem.last_action == "SHIELD" and mem.last_shield_legal:
        shielders.add(me)
    mem.inferred = sorted(shielders)
    for b, s in blame.items():
        bkey = None
        if b == me and s != me and not moved[b]:
            x = cur.pos[b]
            if prev.lvl[x] == 2 and not prev.home[x]:
                bkey = (s, x)                     # caught on a rival's L2 cell: back off it
                mem.backoff[bkey] = t
        mem.blamed[b] = (s, t, mem.last_shield.get(s, -100), bkey)

    # --- behaviour statistics (before last_shield is updated) ---
    for o in bots:
        if was_frz[o] or t <= mem.last_shield[o] + 4:
            continue
        intruders = [b for b in bots if b != o and prev.own[prev.pos[b]] == o and not was_frz[b]]
        st = mem.stats[o]
        if intruders:
            st["opp"] += 1
            if o in shielders:
                st["sh"] += 1
        elif o in shielders:
            st["spam"] += 1
        if o != me:
            # F1: which reaction bucket did the owner face? (the most advanced raid)
            bk = -1
            for b in intruders:
                x = prev.pos[b]
                if prev.home[x]:
                    continue
                k = 0 if prev.lvl[x] == 1 else (1 if mem.prog.get(b, 0) <= 0 else 2)
                if k > bk:
                    bk = k
            if bk >= 0:
                rc = mem.react.setdefault(o, [[0, 0], [0, 0], [0, 0]])[bk]
                rc[0] += 1
                if o in shielders:
                    rc[1] += 1
    for b in bots:
        x = prev.pos[b]
        o = prev.own[x]
        if o and o != b and not prev.home[x] and not was_frz[b]:
            st = mem.stats[b]
            st["rv_opp"] += 1
            if not moved[b] and b not in shielders:
                st["rv_stay"] += 1

    for s in shielders:
        mem.last_shield[s] = t

    # --- rule detection: do frozen stays build progress? (off: rule confirmed) ---
    for b in (bots if P["fz_detect"] else ()):
        if moved[b]:
            continue
        x = cur.pos[b]
        took = prev.own[x] != b and cur.own[x] == b
        if was_frz[b] and (took or (prev.own[x] == b and prev.lvl[x] == 1 and cur.lvl[x] == 2)):
            mem.fz_prog = True                    # took / upgraded while frozen
        elif took and prev.lvl[x] == 2 and prev.own[x] and mem.prog[b] + 1 < 3 \
                and mem.frz_seen_on_cell.get(b) == x:
            mem.fz_prog = True                    # level-2 take too soon: frozen turns counted
    for b in bots:
        if cur.frozen(b):
            mem.frz_seen_on_cell[b] = cur.pos[b]
        elif moved[b]:
            mem.frz_seen_on_cell.pop(b, None)

    # --- progress (consecutive effective stays on the current cell) ---
    for b in bots:
        x = cur.pos[b]
        frozen_counts = was_frz[b] and mem.fz_prog
        if moved[b] or b in shielders or newly[b] or (was_frz[b] and not frozen_counts):
            mem.prog[b] = 0
            continue
        mem.prog[b] += 1
        if prev.own[x] != b and cur.own[x] == b:
            mem.prog[b] = 0                       # take completed
        elif prev.own[x] == b and prev.lvl[x] == 1 and cur.lvl[x] == 2:
            mem.prog[b] = 0                       # upgrade completed
        elif cur.own[x] != b and prev.own[x] != cur.own[x]:
            mem.prog[b] = 0                       # owner changed under us

    # --- raid hazard: owned non-home cell-turns and how many got raided ---
    pt = (0.0, _PROF[1][min(max(t, 1), TOTAL_TURNS)], _PROF[2][min(max(t, 1), TOTAL_TURNS)])
    for i in range(cur.size):
        po = prev.own[i]
        if po and not prev.home[i]:
            lv = prev.lvl[i]
            mem.cellturns[lv] += 1
            mem.exp_raids[lv] += pt[lv]
            raided = cur.own[i] and cur.own[i] != po
            if raided:
                mem.raids[lv] += 1
            if po == me:
                mem.my_cellturns[lv] += 1
                mem.my_exp_raids[lv] += pt[lv]
                if raided:
                    mem.my_raids[lv] += 1

    # --- scores: income at end of turn t + raid bonuses ---
    for i in range(cur.size):
        o = cur.own[i]
        if o:
            mem.scores[o] = mem.scores.get(o, 0.0) + cur.val[i] * cur.lvl[i]
            po = prev.own[i]
            if po and po != o:
                mem.scores[o] += 8 * cur.val[i] * prev.lvl[i]

    if mem.last_action in MOVES:
        _learn_dirs(mem, prev, cur, mem.last_action)


# ---------------------------------------------------------------------------
# 5. Planner
# ---------------------------------------------------------------------------
class Ctx(object):
    pass


def _neighbors(mem, idx):
    n = mem.n
    r, c = divmod(idx, n)
    out = []
    for a in MOVES:
        dr, dc = mem.dmap[a]
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((a, rr * n + cc))
    return out


def _bfs(mem, start, blocked):
    dist = [-1] * (mem.n * mem.n)
    dist[start] = 0
    q = [start]
    for x in q:
        dx = dist[x] + 1
        for _, y in _neighbors(mem, x):
            if dist[y] < 0 and not blocked[y]:
                dist[y] = dx
                q.append(y)
    return dist


def _rho(mem, o):
    st = mem.stats.get(o)
    if st is None:
        return P["rho_prior"]
    n0 = P["rho_prior_n"]
    return (st["sh"] + P["rho_prior"] * n0) / (st["opp"] + n0)


def _q_stay(mem, r):
    st = mem.stats.get(r)
    n0 = P["q_prior_n"]
    if st is None:
        return P["q_prior"]
    return (st["rv_stay"] + P["q_prior"] * n0) / (st["rv_opp"] + n0)


def _build_ctx(mem, snap):
    c = Ctx()
    c.mem, c.s = mem, snap
    c.t = mem.turn
    c.R = max(1, TOTAL_TURNS - c.t + 1)
    c.me = mem.me
    c.rivals = [b for b in snap.bots if b != c.me]
    c.mypos = snap.pos[c.me]
    size = snap.size

    # frozen_until[b]: last turn on which b is still frozen (t-1 if not frozen)
    c.frozen_until = {}
    for b in snap.bots:
        if snap.frozen(b):
            run = max(1, mem.frz_run.get(b, 1))
            c.frozen_until[b] = c.t + max(0, 2 - run)
        else:
            c.frozen_until[b] = c.t - 1

    # distances (cells occupied by other bots are closed)
    occ = [False] * size
    for b, p in snap.pos.items():
        occ[p] = True
    blocked = list(occ)
    blocked[c.mypos] = False
    c.dme = _bfs(mem, c.mypos, blocked)
    c.dr = {}
    for b in c.rivals:
        bl = list(occ)
        bl[snap.pos[b]] = False
        d = _bfs(mem, snap.pos[b], bl)
        extra = max(0, c.frozen_until[b] - c.t + 1)
        if extra:
            d = [x + extra if x >= 0 else x for x in d]
        c.dr[b] = d

    # income + Elo weights
    income = dict((b, 0) for b in mem.ids)
    for i in range(size):
        o = snap.own[i]
        if o:
            income[o] = income.get(o, 0) + snap.val[i] * snap.lvl[i]
    c.income = income
    proj = dict((b, mem.scores.get(b, 0.0) + income.get(b, 0) * c.R) for b in snap.bots)
    c.proj = proj
    sigma = P["sigma_base"] + P["sigma_per_turn"] * c.R
    sens = {}
    for r in c.rivals:
        z = (proj[c.me] - proj[r]) / sigma
        z = max(-30.0, min(30.0, z))
        p = 1.0 / (1.0 + math.exp(-z))
        sens[r] = p * (1.0 - p)
    tot = sum(sens.values())
    mix = P["w_uniform_mix"]
    nr = max(1, len(c.rivals))
    c.w = {}
    for r in c.rivals:
        base = sens[r] / tot if tot > 1e-12 else 1.0 / nr
        c.w[r] = (1.0 - mix) * base + mix / nr
    c.w[0] = 0.0
    c.rho = dict((r, _rho(mem, r)) for r in c.rivals)
    n0 = P["lam_prior_n"]
    c.lam1 = (mem.raids[1] + P["lam1_prior"] * n0) / (mem.cellturns[1] + n0)
    c.lam2 = (mem.raids[2] + P["lam2_prior"] * n0) / (mem.cellturns[2] + n0)
    fz_haz = mem.fz_prog and P["fz_hazard"]
    lam2_prior = P["lam2_prior_fz"] if fz_haz else P["lam2_prior"]
    c.lam2_mine = (mem.my_raids[2] + lam2_prior * n0) / (mem.my_cellturns[2] + n0)
    if fz_haz:
        c.lam2 = max(c.lam2, c.lam2_mine)
    c.w_avg = 1.0 / nr
    # F2: phase-dependent hazard (profile x learned per-game scale)
    c.phase = P["hazard_mode"] == 1
    if c.phase:
        hn = P["h_prior_n"]
        s1 = (mem.raids[1] + hn) / (mem.exp_raids[1] + hn) * P["h1_mult"]
        s2 = (mem.raids[2] + hn) / (mem.exp_raids[2] + hn) * P["h2_mult"]
        s2m = (mem.my_raids[2] + hn) / (mem.my_exp_raids[2] + hn) * P["h2_mult"]
        c.lt1, c.se1 = _life_table(1, s1)
        c.lt2 = _life_table(2, s2)[0]
        c.lt2m = _life_table(2, s2m)[0]
    # exposure: cells near a rival bot get raided sooner
    k, r0 = P["expo_k"], P["expo_r"]
    c.expo = [1.0] * size
    if k:
        for x in range(size):
            md = 99
            for b in c.rivals:
                d = c.dr[b][x]
                if 0 <= d < md:
                    md = d
            c.expo[x] = min(3.0, max(0.3, 1.0 + k * (r0 - md) / r0))
    return c


def _life(G, lam):
    """Expected number of scoring turns over G turns for a cell raided at rate lam."""
    if G <= 0:
        return 0.0
    if not P["use_life"] or lam <= 1e-9:
        return float(G)
    return (1.0 - (1.0 - lam) ** G) / lam


def _life_table(lv, scale):
    """Phase hazard. L[a] = expected scoring turns of a cell completed on turn a
    (scores a..80, raided at rate scale*profile(u) on later turns).
    S[a] = probability it survives to the end."""
    prof = _PROF[lv]
    n = TOTAL_TURNS
    L = [0.0] * (n + 2)
    S = [1.0] * (n + 2)
    L[n] = 1.0
    for a in range(n - 1, 0, -1):
        q = 1.0 - min(0.95, scale * prof[a + 1])
        L[a] = 1.0 + q * L[a + 1]
        S[a] = q * S[a + 1]
    L[0], S[0] = L[1], S[1]
    return L, S


def _lt(tab, G):
    """Table value for a plan leaving G scoring turns (completion turn 81 - G)."""
    if G <= 0:
        return 0.0
    a = TOTAL_TURNS + 1 - G
    if a <= 1:
        return tab[1]
    a0 = int(a)
    f = a - a0
    return tab[a0] * (1.0 - f) + tab[a0 + 1] * f


def _react_rate(mem, o, bk):
    """Posterior P(owner o shields) in reaction bucket bk, and observations."""
    pri = (P["r1_prior"], P["r20_prior"], P["r2k_prior"])[bk]
    rc = mem.react.get(o)
    if rc is None:
        return pri, 0
    n, sh = rc[bk]
    n0 = P["react_n"]
    return (sh + pri * n0) / (n + n0), n


def _react_raid(c, x, o, kind, d, prog, stays, start_turn, timed):
    """F1 raid model (default rules). Returns (P(raid finishes), expected extra turns),
    or None for a vetoed L2 raid.

    Stays fall on turns u0 .. u0+stays-1. A shield on a stay catches us: progress
    is lost and we are frozen 2 turns. After an L1 catch the retake on turn +3
    always beats the 4-turn cooldown; an L2 raid caught once is dead."""
    mem = c.mem
    u0 = start_turn + d
    if kind == RAID1:
        r, _ = _react_rate(mem, o, 0)
        pc = r if (not timed or _shield_avail(c, o, u0)) else 0.0
        return 1.0, P["raid_caught_delay"] * pc
    re, _ = _react_rate(mem, o, 1)
    rl, nl = _react_rate(mem, o, 2)
    bo = mem.backoff.get((o, x))
    if bo is not None and c.t - bo <= P["backoff_turns"]:
        rl = max(rl, P["backoff_h"])
        nl = max(nl, P["react_veto_n"])
    p_ok = 1.0
    delay = 0.0
    for j in range(stays):
        if timed and not _shield_avail(c, o, u0 + j):
            continue
        p = prog + j
        if p > 0 and timed and rl >= P["react_veto"] and nl >= P["react_veto_n"]:
            return None                       # owner reliably shields this stay
        h = re if p <= 0 else rl
        delay += p_ok * h * (j + 3 - stays)   # caught at stay j: j+1 stays + 2 frozen
        p_ok *= 1.0 - h
    return p_ok, delay


def _fz_probe(c):
    """First free turn after being caught on rival o's L2 cell x (caught on turn
    tc, frozen tc+1 and tc+2, now tc+3). o cannot shield before tc+5, so one more
    stay is safe. Under the literal rules it costs a turn. Under the reading where
    frozen stays build progress, it completes the take (and the tracker then
    detects that reading), where the back-off alone would walk away."""
    mem, s = c.mem, c.s
    if mem.fz_prog or not P["react_model"] or not P["fz_probe"]:
        return False
    x = c.mypos
    o = s.own[x]
    if not o or o == c.me or s.home[x] or s.lvl[x] != 2:
        return False
    tc = mem.backoff.get((o, x))
    return tc is not None and c.t == tc + 3 and not _shield_avail(c, o, c.t)


def _shield_avail(c, o, u):
    mem = c.mem
    return u > mem.last_shield.get(o, -100) + 4 and u > c.frozen_until.get(o, c.t - 1)


def _kind_of(c, x, own, lvl):
    """Plan kind for cell x given an ownership view, or None."""
    s = c.s
    o = own[x]
    if o == c.me:
        if lvl[x] == 1 and (not s.home[x] or P["home_upgrade"]):
            return UPGRADE
        return None
    if s.home[x]:
        return None
    if o == 0:
        return NEUTRAL
    return RAID1 if lvl[x] == 1 else RAID2


def _stays_needed(kind):
    return 1 if kind in (NEUTRAL, RAID1) else 3


def _race_keep(c, x, d, arrival_turn):
    """Probability we, not a rival, end up taking neutral cell x."""
    if d <= 0:
        return 1.0
    v = c.s.val[x]
    scale = P["race_value_scale"] + (1.0 - P["race_value_scale"]) * v / 3.0
    keep = 1.0
    for b in c.rivals:
        db = c.dr[b][x]
        if db < 0:
            continue
        if db < d:
            p = P["race_ahead"]
        elif db == d:
            p = P["race_tie_win"] if _first_in_order(c.mem, arrival_turn, c.me, b) else P["race_tie_lose"]
        else:
            continue
        keep *= 1.0 - p * scale
    return keep


def _evaluate(c, x, kind, d, prog, start_turn, own, lvl, timed=True):
    """Return (utility, turns) for doing `kind` on cell x, d moves away,
    starting at absolute turn start_turn with `prog` stays already done."""
    s = c.s
    v = s.val[x]
    o = own[x]
    stays = max(1, _stays_needed(kind) - prog)
    k = d + stays
    R = TOTAL_TURNS - start_turn + 1
    if k > R:
        return None
    keep = 1.0
    delay = 0.0
    if kind == NEUTRAL:
        keep = _race_keep(c, x, d + (start_turn - c.t), start_turn + d - 1)
    elif kind in (RAID1, RAID2) and P["react_model"] and not c.mem.fz_prog:
        rr = _react_raid(c, x, o, kind, d, prog, stays, start_turn, timed)
        if rr is None:
            return None
        keep, delay = rr
    elif kind in (RAID1, RAID2):
        rho = c.rho.get(o, P["rho_prior"])
        if timed:
            n_av = 0
            for u in range(start_turn + d, start_turn + d + stays):
                if _shield_avail(c, o, u):
                    n_av += 1
        else:
            n_av = stays
        pc = 1.0 - (1.0 - rho) ** n_av if n_av else 0.0
        if c.mem.fz_prog:
            # frozen stays still count: a catch only costs the protected stay
            delay = (1.0 if kind == RAID1 else 2.0) * pc
        elif kind == RAID1:
            delay = P["raid_caught_delay"] * pc
        else:
            keep = 1.0 - pc
            delay = P["raid2_caught_delay"] * pc
    k_eff = k + delay
    G = R - k_eff + 1
    if G <= 0:
        return None
    w = c.w.get(o, 0.0)
    phase = c.phase
    life1 = _lt(c.lt1, G) if phase else _life(G, c.lam1 * c.expo[x])
    if kind == NEUTRAL:
        u = v * life1
    elif kind == RAID1:
        u = 8 * v + v * life1 + w * v * life1
    elif kind == RAID2:
        u = 16 * v + v * life1 + w * 2 * v * (_lt(c.lt2, G) if phase else _life(G, c.lam2))
    else:
        # level 2 lasts longer (we defend it with shields) and denies raid bonuses
        u = 2 * v * (_lt(c.lt2m, G) if phase else _life(G, c.lam2_mine)) - v * life1
        if P["use_life"]:
            p_raided = (1.0 - _lt(c.se1, G)) if phase else (1.0 - (1.0 - c.lam1) ** G)
            u += c.w_avg * 8 * v * p_raided
        rad = P["upg_def_radius"]
        md = min([c.dr[b][x] for b in c.rivals if c.dr[b][x] >= 0] or [99])
        if md <= rad:
            u += P["upg_def_frac"] * v * G * (rad + 1 - md) / (rad + 1.0)
    return u * keep, k_eff


def _apply(c, own, lvl, x, kind):
    own2 = list(own)
    lvl2 = list(lvl)
    if kind == UPGRADE:
        lvl2[x] = 2
    else:
        own2[x] = c.me
        lvl2[x] = 1
    return own2, lvl2


def _chain(c, x, kind, u_acc, k_acc, own, lvl, t0, depth, deadline):
    """Best rate (total utility / total turns) of chains that start with the
    given (already accumulated) plans and continue up to `depth` more plans."""
    s = c.s
    n = s.n
    own2, lvl2 = _apply(c, own, lvl, x, kind)
    t2 = t0 + int(math.ceil(k_acc))
    if t2 > TOTAL_TURNS:
        return u_acc / k_acc
    best = u_acc / k_acc
    nxt = []
    for y in range(s.size):
        kind2 = _kind_of(c, y, own2, lvl2)
        if kind2 is None:
            continue
        if y != x and s.occ[y] and y != c.mypos:
            continue
        d2 = 0 if y == x else _manh(n, x, y)
        ev2 = _evaluate(c, y, kind2, d2, 0, t2, own2, lvl2, timed=False)
        if ev2 is None:
            continue
        u2, k2 = ev2
        r2 = (u_acc + u2) / (k_acc + k2)
        if r2 > best:
            best = r2
        nxt.append((r2, u2, k2, y, kind2))
    if depth > 1 and nxt and time.perf_counter() < deadline:
        nxt.sort(reverse=True)
        for r2, u2, k2, y, kind2 in nxt[:int(P["chain_top2"])]:
            r3 = _chain(c, y, kind2, u_acc + u2, k_acc + k2, own2, lvl2, t0, depth - 1, deadline)
            if r3 > best:
                best = r3
    return best


def _plan(c, deadline):
    s, mem = c.s, c.mem
    own, lvl = s.own, s.lvl
    me = c.me
    first = []
    for x in range(s.size):
        if x == c.mypos:
            kind = _kind_of(c, x, own, lvl)
            if kind is None:
                continue
            ev = _evaluate(c, x, kind, 0, mem.prog.get(me, 0), c.t, own, lvl)
        else:
            d = c.dme[x]
            if d < 0 or s.occ[x]:
                continue
            kind = _kind_of(c, x, own, lvl)
            if kind is None:
                continue
            ev = _evaluate(c, x, kind, d, 0, c.t, own, lvl)
        if ev is None:
            continue
        u, k = ev
        if u <= 0:
            continue
        first.append((u / k, u, k, x, kind))
    if not first:
        return None
    first.sort(reverse=True)

    # bait: moving off a shield-ready owner's land can make it backfire
    here_owner = own[c.mypos]
    bait = 0.0
    if here_owner and here_owner != me and _shield_avail(c, here_owner, c.t):
        bait = c.rho.get(here_owner, P["rho_prior"]) * P["bait_value"] * c.w.get(here_owner, 0.0) * 3.0

    ranked = []
    top = first[:int(P["chain_top"])]
    for rate, u1, k1, x, kind in top:
        score = rate
        if time.perf_counter() < deadline:
            if P["chain_depth"] > 1:
                score = max(score, _chain(c, x, kind, u1, k1, own, lvl, c.t,
                                          int(P["chain_depth"]) - 1, deadline))
        if x != c.mypos:
            score += bait / max(1.0, k1)
        if mem.target is not None and mem.target == (x, kind):
            score *= 1.0 + P["hysteresis"]
        ranked.append((score, x, kind, u1, k1))
    ranked.sort(key=lambda p: p[0], reverse=True)
    return ranked


def _step_toward(c, target):
    """First move on a shortest path to target, avoiding contests we'd lose."""
    s, mem, me = c.s, c.mem, c.me
    occ = [False] * s.size
    for b, p in s.pos.items():
        occ[p] = True
    bl = list(occ)
    bl[c.mypos] = False
    bl[target] = False
    dt = _bfs(mem, target, bl)
    here = dt[c.mypos]
    best = None
    for a, y in _neighbors(mem, c.mypos):
        if a not in s.legal or occ[y]:
            continue
        if dt[y] < 0 or (here >= 0 and dt[y] >= here):
            continue
        risk = 0.0
        for b in c.rivals:
            if c.frozen_until[b] >= c.t:
                continue
            pb = s.pos[b]
            if _manh(s.n, pb, y) != 1:
                continue
            if s.own[y] == me:
                continue
            if s.own[y] == b or not _first_in_order(mem, c.t, me, b):
                risk += P["contest_block_p"]
        score = -risk + 0.01 * s.val[y] + (0.02 if s.own[y] != me else 0.0)
        if best is None or score > best[0]:
            best = (score, a)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# 6. Shield decision
# ---------------------------------------------------------------------------
def _shield_value(c, plan):
    s, mem = c.s, c.mem
    me = c.me
    if "SHIELD" not in s.legal or not _shield_avail(c, me, c.t):
        return None                      # rulebook 2.6: no shield for 4 turns after one
    # what one turn of our own plan is worth, and what shielding throws away
    my_prog = mem.prog.get(me, 0)
    if plan is None:
        my_rate, lost_turns = 0.0, 1.0
    else:
        u1, k1 = plan[3], plan[4]
        if plan[1] == c.mypos:
            my_rate = u1 / max(1.0, k1 + my_prog)
            lost_turns = 1.0 + my_prog
        else:
            my_rate = u1 / max(1.0, k1)
            lost_turns = 1.0
    ev = 0.0
    p_none = 1.0
    any_threat = False
    for r in c.rivals:
        if c.frozen_until[r] >= c.t:
            continue                     # already frozen: can't be caught
        x = s.pos[r]
        if s.own[x] != me:
            continue
        any_threat = True
        v = s.val[x]
        G = c.R
        w = c.w.get(r, 0.0)
        prog = mem.prog.get(r, 0)
        if s.home[x]:
            q = _q_stay(mem, r) * 0.5
            saved = 0.0
        elif s.lvl[x] == 1:
            q = _q_stay(mem, r)
            full = v * G + w * (8 * v + v * G)
            saved = 3 * v + P["shield_giveup"] * full
        else:
            full = 2 * v * G + w * (16 * v + v * G)
            if prog >= 2:
                q = max(_q_stay(mem, r), P["q_committed"])
                saved = P["shield_recatch"] * full
            elif prog >= 1:
                q = max(_q_stay(mem, r), P["q_committed"])
                saved = P["shield_early_frac"] * full
            else:
                q = _q_stay(mem, r)
                saved = 0.25 * P["shield_early_frac"] * full
        freeze = w * 3.0 * P["rival_rate"]
        if mem.fz_prog and P["fz_shield"] and not s.home[x]:
            # a caught raider keeps building progress while frozen: we only
            # delay the take by the protected turn(s)
            saved = v * s.lvl[x] * (1.0 if s.lvl[x] == 1 else 3.0)
            freeze = w * P["rival_rate"]
        ev += q * (saved + freeze)
        p_none *= 1.0 - q
    if not any_threat:
        return None
    cost = my_rate * lost_turns
    cost += p_none * P["backfire_turns"] * my_rate
    cost += P["cooldown_cost"]
    return ev - cost - P["shield_margin"]


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
def _log_exc():
    if _DEBUG:
        import traceback
        traceback.print_exc(file=sys.stderr)


def _choose(state, t0):
    global _MEM
    snap = Snap(state)
    if _is_new_game(_MEM, snap):
        _MEM = Memory(snap)
    mem = _MEM
    mem.turn += 1
    try:
        mem.turn = min(TOTAL_TURNS, mem.turn + _skipped_turns(mem, snap))
    except Exception:
        _log_exc()
    try:
        if mem.prev is not None:
            _update(mem, mem.prev, snap)
        _token_update(mem, mem.turn, snap.me, snap.token)
        for b, h in snap.homes.items():
            mem.homes[b] = h
    except Exception:
        _log_exc()

    action = "STAY"
    try:
        action = _decide(mem, snap, t0)
    except Exception:
        _log_exc()
        action = "STAY"
    if action not in snap.legal:
        action = "STAY"
    mem.prev = snap
    mem.last_action = action
    mem.last_shield_legal = "SHIELD" in snap.legal
    return snap.legal_map.get(action, snap.legal_map.get("STAY", "STAY"))


def _decide(mem, snap, t0):
    action = "STAY"
    if snap.frozen(snap.me):
        return action                    # rulebook 2.6: a frozen bot can only STAY
    if snap.me in snap.pos and len(snap.legal) > 1:
        budget = P["budget_ms"] / 1000.0
        deadline = t0 + budget * 0.8
        c = _build_ctx(mem, snap)
        ranked = _plan(c, deadline) or []
        plan = None
        step = None
        for cand in ranked:                     # best plan we can actually start
            if cand[1] == c.mypos:
                plan, step = cand, "STAY"
                break
            st = _step_toward(c, cand[1])
            if st:
                plan, step = cand, st
                break
        sv = _shield_value(c, plan)
        if sv is not None and sv > 0:
            action = "SHIELD"
            mem.target = None
        elif _fz_probe(c):
            action = "STAY"
            mem.target = None
        elif plan is not None:
            mem.target = (plan[1], plan[2])
            action = step
        if _DEBUG:
            sys.stderr.write("ZRDBG %s\n" % repr({
                "turn": mem.turn, "me": mem.me, "act": action, "plan": plan and plan[1:3],
                "shield_ev": sv, "scores": dict(mem.scores), "prog": dict(mem.prog),
                "last_shield": dict(mem.last_shield), "inferred": mem.inferred,
                "rho": c.rho, "w": c.w}))
    return action


def act(state): # Do not change or modify this function signature
    t0 = time.perf_counter()
    try:
        return _choose(state, t0)
    except Exception:
        _log_exc()
        try:
            legal = list(state.legal_actions)
            for a in legal:
                if str(a).upper() == "STAY":
                    return a
            return legal[0]
        except Exception:
            return "STAY"
