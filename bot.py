# this is my 67 bot
"""econ - exact-economics Zone Rush bot.

Every turn: list every task open to me (take a neutral cell, upgrade one of my
cells, raid a rival cell), compute the points it adds by the end of the game
and the turns it costs (travel included), search short chains of tasks and do
the first task of the chain with the best points-per-turn.  Shield exactly when
a rival's tracked stay-progress would complete a capture this turn.
All bookkeeping (turn, scores, progress, cooldowns, freezes) is tracked in
module globals and re-derived from the board after a restart.
"""

import time as _time

_TOTAL_TURNS = 50          # turns per game (was 80)

_S = None
_FIX_RESUME = True         # restart-safe turn (board progress + token snap), non-shield freeze bookkeeping

# tuning
_GAMMA = 0.5        # weight of rate vs. total in plan value
_Q1 = (1.0, 0.75, 0.75, 0.6)   # chance weight that a level-1 cell of base b stays mine
_Q2 = 0.95           # same for a level-2 cell (shield-defended)
_KEEP = 0.7          # chance weight that a raided L1 cell's income stays mine
_DENY = 0.3          # weight of points a raid takes away from the victim
_BEAM = (7, 4, 3)    # beam widths per depth
_TOPN = 26           # candidates kept after the first-level ranking
_NODE_CAP = 60       # hard cap on expanded plan nodes


def act(state):
    legal = None
    try:
        legal = list(state.legal_actions)
        a = _decide(state, legal)
        if a in legal:
            return a
    except Exception:
        pass
    try:
        if legal and "STAY" not in legal:
            return legal[0]
    except Exception:
        pass
    return "STAY"


def _read(state):
    occ = [0] * 64
    own = [0] * 64
    lvl = [1] * 64
    home = [0] * 64
    base = [1] * 64
    pos = [-1] * 5
    frozen = [False] * 5
    i = 0
    for row in state.board:
        for cell in row:
            if i >= 64:
                break
            o = cell.occupant
            if o and 1 <= o <= 4:
                occ[i] = o
                pos[o] = i
                if cell.is_frozen:
                    frozen[o] = True
            w = cell.owner
            if w and 1 <= w <= 4:
                own[i] = w
            if cell.is_upgraded:
                lvl[i] = 2
            if cell.is_home:
                home[i] = 1
            b = cell.base_points
            base[i] = b if isinstance(b, int) and b > 0 else 1
            i += 1
    return occ, own, lvl, home, base, pos, frozen


_INC_TURN = ((8, 8), (14, 13), (20, 19), (25, 26), (29, 31), (35, 39), (40, 47),
             (45, 58), (50, 68), (54, 76))


# (cells + upgrades of all bots, turn), mean of example and random maps
_RS_TAB = ((0, 1), (4, 5), (11, 10), (19, 15), (27, 20), (34, 25), (42, 30), (49, 35), (55, 40), (60, 45), (66, 50))


def _estimate_turn(own, lvl, home, pos, base=None):
    work = [0] * 5
    for i in range(64):
        o = own[i]
        if o:
            if not home[i]:
                work[o] += 2
            if lvl[i] == 2:
                work[o] += 3
    best = max(work)
    for s in range(1, 5):
        p = pos[s]
        if p < 0:
            continue
        for i in range(64):
            if home[i] and own[i] == s:
                d = abs(i // 8 - p // 8) + abs(i % 8 - p % 8)
                if d > best:
                    best = d
    if _FIX_RESUME:
        # map-independent: cells + upgrades of all four bots (fit on 50-turn games, +-1.3 turns)
        cnt = [0] * 5
        for i in range(64):
            o = own[i]
            if o:
                if not home[i]:
                    cnt[o] += 1
                if lvl[i] == 2:
                    cnt[o] += 1
        x = sum(cnt)
        est = 1
        for k in range(len(_RS_TAB) - 1):
            x0, t0 = _RS_TAB[k]
            x1, t1 = _RS_TAB[k + 1]
            if x >= x0:
                est = int(t0 + (t1 - t0) * min(1.0, (x - x0) / float(x1 - x0)) + 0.5)
        if est > best + 1:
            best = est - 1
        return max(1, min(_TOTAL_TURNS, best + 1))
    if base is not None:
        inc = [0] * 5
        for i in range(64):
            if own[i]:
                inc[own[i]] += base[i] * lvl[i]
        top = max(inc)
        est = 1
        for a, b in _INC_TURN:
            if top >= a:
                est = b
        if est > best:
            best = est
    return max(1, min(_TOTAL_TURNS - 2, best + 1))


def _init(me, own, lvl, home, base, pos, frozen):
    pristine = True
    for i in range(64):
        if (own[i] and not home[i]) or lvl[i] == 2:
            pristine = False
            break
    if pristine:
        for s in range(1, 5):
            p = pos[s]
            if p >= 0 and not home[p]:
                pristine = False
    t = 1 if pristine else _estimate_turn(own, lvl, home, pos, base)
    score = [0.0] * 5
    if t > 1:
        for i in range(64):
            if own[i]:
                score[own[i]] += base[i] * lvl[i] * (t - 1) * 0.55
    return {
        "turn": t, "pos": pos, "frozen": frozen, "own": own, "lvl": lvl,
        "prog": [0] * 5, "shield_ready": [1] * 5, "frozen_until": [0] * 5,
        "score": score, "last_act": "STAY", "bait": [0] * 5,
        "shield_use": [0] * 5, "noshield": [0] * 5, "targets": [],
        "caught_me": [0] * 5, "l1free": [0] * 5, "l1catch": [0] * 5,
        "dance_fail": [0] * 5, "danced": None,
        "pristine": pristine, "dirty": not pristine, "resync": not pristine,
        "fzc": [None] * 5,
    }


def _update(S, me, own, lvl, home, base, pos, frozen):
    S["turn"] += 1
    t = S["turn"]
    last = t - 1
    ppos, pfr, pown, plvl = S["pos"], S["frozen"], S["own"], S["lvl"]
    prog = S["prog"]
    my_shield = (S["last_act"] == "SHIELD")
    shielded = [False] * 5
    if my_shield:
        shielded[me] = True
        S["shield_ready"][me] = last + 5
    newly = [False] * 5
    for x in range(1, 5):
        if pos[x] < 0:
            continue
        if frozen[x] and not pfr[x]:
            newly[x] = True
            moved = (pos[x] != ppos[x])
            S["frozen_until"][x] = last + (3 if moved else 2)
            if moved:
                continue
            o = pown[pos[x]]
            if x == me:
                if not my_shield and o and o != me:
                    shielded[o] = True          # I was caught by o's shield
                    S["caught_me"][o] += 1
                    if plvl[pos[x]] == 1 and prog[x] == 0:
                        S["l1catch"][o] += 1
                continue
            if o and o != x:
                if o == me:
                    if not my_shield or (frozen[me] and not pfr[me]):
                        shielded[x] = True      # x backfired on my land
                elif frozen[o] and not pfr[o] and pos[o] == ppos[o]:
                    shielded[x] = True          # both shielded, both backfired
                    shielded[o] = True
                else:
                    shielded[o] = True          # caught by o's shield
                    if plvl[pos[x]] == 1 and prog[x] == 0:
                        S["l1catch"][o] += 1
            else:
                shielded[x] = True              # backfire
    for x in range(1, 5):
        if x != me and shielded[x]:
            S["shield_ready"][x] = last + 5
            S["shield_use"][x] += 1
    if _FIX_RESUME:
        # a freeze that outlasts a shield freeze (2 turns) was a fault / not-ready freeze: undo the shield credit
        fzc = S["fzc"]
        fu = S["frozen_until"]
        for x in range(1, 5):
            if pos[x] < 0:
                continue
            if not frozen[x]:
                fzc[x] = None
                continue
            if newly[x]:
                fzc[x] = None
                if pos[x] == ppos[x]:
                    o = pown[pos[x]]
                    l1 = 1 if (plvl[pos[x]] == 1 and prog[x] == 0) else 0
                    if x == me:
                        if not my_shield and o and o != me:
                            fzc[x] = (o, l1, 1)
                    elif o and o != x:
                        if o == me:
                            if shielded[x]:
                                fzc[x] = (x, 0, 0)
                        elif not (frozen[o] and not pfr[o] and pos[o] == ppos[o]):
                            fzc[x] = (o, l1, 0)
                    else:
                        fzc[x] = (x, 0, 0)
            elif t > fu[x]:
                c = fzc[x]
                if c is not None and fu[x] > 0:
                    y = c[0]
                    if y != me:
                        if S["shield_use"][y] > 0:
                            S["shield_use"][y] -= 1
                        if S["shield_ready"][y] == fu[x] + 3:
                            S["shield_ready"][y] = 1
                    if c[1] and S["l1catch"][y] > 0:
                        S["l1catch"][y] -= 1
                    if c[2] and S["caught_me"][y] > 0:
                        S["caught_me"][y] -= 1
                    fzc[x] = None
                fu[x] = t
    # bait detection: I shielded and it backfired
    if my_shield and newly[me]:
        for x in S["targets"]:
            S["bait"][x] += 1
    S["targets"] = []
    # progress
    for x in range(1, 5):
        p = pos[x]
        if p < 0:
            continue
        if p != ppos[x] or pfr[x] or newly[x] or shielded[x]:
            prog[x] = 0
            continue
        o_prev = pown[p]
        if o_prev == x:
            if plvl[p] == 1:
                prog[x] += 1
                if prog[x] >= 3 or lvl[p] == 2:
                    prog[x] = 0
            else:
                prog[x] = 0
        elif home[p]:
            prog[x] = 0
        elif o_prev and shielded[o_prev]:
            pass
        else:
            prog[x] += 1
            if own[p] == x:
                prog[x] = 0
            elif prog[x] > 2:
                prog[x] = 2
    # scores, raid bonuses, unshielded L2 losses
    score = S["score"]
    for i in range(64):
        o = own[i]
        if o:
            score[o] += base[i] * lvl[i]
            po = pown[i]
            if po and po != o:
                score[o] += 8 * base[i] * plvl[i]
                if S["shield_ready"][po] <= last and not pfr[po]:
                    if plvl[i] == 2:
                        S["noshield"][po] += 1
                    else:
                        S["l1free"][po] += 1
    S["pos"], S["frozen"], S["own"], S["lvl"] = pos, frozen, own, lvl


def _plan(cands, mr, mc, T):
    """Search chains of up to 3 tasks. Returns list of (value, rate, cand index)
    for first tasks, best first."""
    n_c = len(cands)
    gamma = _GAMMA
    first_best = {}
    nodes = [0]
    deadline = _time.perf_counter() + 0.020

    def rec(pr, pc, tau, used, depth, G, L, first, pool):
        lst = []
        for j in pool:
            if used >> j & 1:
                continue
            cd = cands[j]
            req = cd[6]
            if req >= 0 and not (used >> req & 1):
                continue
            d = abs(cd[0] - pr) + abs(cd[1] - pc)
            xd = cd[4]
            if depth == 0 and d == 0:
                xd = cd[8]
            n = T - tau - d - xd
            if n < 1:
                continue
            g = cd[2] + cd[3] * n
            if g <= 0:
                continue
            dur = d + 1 + xd + cd[5]
            lst.append((g / dur, g, dur, j, d + 1 + xd))
        if not lst:
            return
        lst.sort(reverse=True)
        if depth == 0:
            lst = lst[:_TOPN]
            keep = set(x[3] for x in lst)
            # paired upgrades of kept takes stay available
            pool2 = [j for j in range(n_c) if j in keep or cands[j][6] in keep]
        else:
            pool2 = pool
        width = _BEAM[depth]
        k = 0
        for rate, g, dur, j, adv in lst:
            if k >= width:
                break
            k += 1
            G2 = G + g
            L2 = L + dur
            V = G2 * (1.0 - gamma) + gamma * T * G2 / L2
            f = j if depth == 0 else first
            cur = first_best.get(f)
            if cur is None or V > cur[0]:
                first_best[f] = (V, G2 / L2 if depth else rate)
            if depth + 1 < 3 and nodes[0] < _NODE_CAP and _time.perf_counter() < deadline:
                nodes[0] += 1
                cd = cands[j]
                rec(cd[0], cd[1], tau + adv, used | (1 << j), depth + 1, G2, L2, f, pool2)

    rec(mr, mc, 0, 0, 0, 0.0, 0.0, -1, list(range(n_c)))
    out = [(v[0], v[1], f) for f, v in first_best.items()]
    out.sort(reverse=True)
    return out


def _decide(state, legal):
    global _S
    me = state.bot_id
    occ, own, lvl, home, base, pos, frozen = _read(state)
    S = _S
    if S is None:
        S = _S = _init(me, own, lvl, home, base, pos, frozen)
    else:
        pristine = True
        for i in range(64):
            if (own[i] and not home[i]) or lvl[i] == 2:
                pristine = False
                break
        if pristine:
            for s in range(1, 5):
                p = pos[s]
                if p >= 0 and not home[p]:
                    pristine = False
        if not pristine:
            S["dirty"] = True
        if pristine and S["turn"] > 6 and (S["dirty"] or not _FIX_RESUME):
            S = _S = _init(me, own, lvl, home, base, pos, frozen)
        else:
            _update(S, me, own, lvl, home, base, pos, frozen)
    if _FIX_RESUME and S["resync"] and state.is_priority_token:
        # seat k holds the token on turns k, k+4, ...: snap the estimate (ties go earlier)
        S["resync"] = False
        d = (me - S["turn"]) % 4
        if d >= 2:
            d -= 4
        if S["turn"] + d < 1:
            d += 4
        if d:
            S["turn"] += d
            for x in range(1, 5):
                if S["shield_ready"][x] > 1:
                    S["shield_ready"][x] += d
                if S["frozen_until"][x] > 0:
                    S["frozen_until"][x] += d
            if S["danced"] is not None:
                S["danced"] = (S["danced"][0] + d, S["danced"][1])
    S["last_act"] = "STAY"
    mypos = pos[me]
    if mypos < 0:
        return "STAY"
    if frozen[me] or len(legal) <= 1:
        return "STAY"
    t = S["turn"]
    T = _TOTAL_TURNS + 1 - t
    if t > _TOTAL_TURNS + 4:
        T = 81 - t          # game runs past 50 turns: assume the old 80-turn length
    if T < 1:
        T = 1
    prog = S["prog"]
    myprog = prog[me]
    mr, mc = mypos // 8, mypos % 8
    rivals = [x for x in range(1, 5) if x != me and pos[x] >= 0]
    rpos = [(pos[x] // 8, pos[x] % 8) for x in rivals if not frozen[x]]

    # ---------------------------------------------------------- candidates
    score = S["score"]
    cands = []
    for i in range(64):
        o = own[i]
        b = base[i]
        r, c = i // 8, i % 8
        d0 = abs(r - mr) + abs(c - mc)
        oc = occ[i]
        occf = 1.0
        if oc and oc != me:
            if d0 <= 1:
                continue
            if o == oc or o == 0:
                continue
            occf = 0.6
        if o == 0:
            mind = 99
            for (rr, rc) in rpos:
                dd = abs(rr - r) + abs(rc - c)
                if dd < mind:
                    mind = dd
            if mind < d0:
                f = 0.55 if mind <= 1 else 0.85
            elif mind == d0:
                f = 0.9
            else:
                f = 1.0
            q1 = _Q1[b] if b < 4 else 0.5
            cands.append((r, c, 0.0, b * q1 * f * occf, 0, 0.0, -1, i, 0))
            j = len(cands) - 1
            if b >= 2:
                cands.append((r, c, 0.0, b * (2.0 * _Q2 - q1) * f, 2, 0.0, j, i, 2))
        elif o == me:
            if lvl[i] == 1:
                xd0 = 2
                if i == mypos:
                    xd0 = max(0, 2 - myprog)
                q1 = 1.0 if home[i] else (_Q1[b] if b < 4 else 0.5)
                q2 = 1.0 if home[i] else _Q2
                cands.append((r, c, 0.0, b * (2.0 * q2 - q1), 2, 0.0, -1, i, xd0))
        else:
            if home[i]:
                continue
            # leader bonus on denial
            deny = _DENY
            if score[o] >= score[me]:
                deny += 0.2
            if lvl[i] == 1:
                q1 = _Q1[b] if b < 4 else 0.5
                stay_turn = t + d0
                pc = 0.0
                if S["shield_ready"][o] <= stay_turn and not (frozen[o] and S["frozen_until"][o] >= stay_turn):
                    cm = S["l1catch"][o]
                    pc = (cm + 0.6) / (cm + 0.5 * S["l1free"][o] + 1.0)
                    if pc > 0.92:
                        pc = 0.92
                if pc > 0.0 and T - d0 - 3 < 1:
                    occf *= (1.0 - pc)
                xt = pc * 3.0
                if S["dance_fail"][o] < 1 and xt > 1.5:
                    xt = 1.5
                cands.append((r, c, 8.0 * b * occf, (b * q1 + deny * b) * occf, 0, xt, -1, i, 0))
                j = len(cands) - 1
                if b >= 2:
                    cands.append((r, c, 0.0, b * (2.0 * _Q2 - q1), 2, 0.0, j, i, 2))
            else:
                xd0 = 2
                if i == mypos:
                    xd0 = max(0, 2 - myprog)
                done_turn = t + d0 + xd0
                if S["shield_ready"][o] > done_turn or (frozen[o] and S["frozen_until"][o] >= done_turn):
                    P = 0.95
                else:
                    ns = S["noshield"][o]
                    su = S["shield_use"][o]
                    if su >= 1:
                        P = 0.04
                    elif ns >= 1:
                        P = 0.75
                    else:
                        P = 0.12
                P *= occf
                cands.append((r, c, P * 16.0 * b, P * (b * _KEEP + deny * 2 * b), 2, (1.0 - P) * 2.0, -1, i, xd0))

    plans = _plan(cands, mr, mc, T) if cands else []
    best_rate = plans[0][1] if plans else 0.0

    # ---------------------------------------------------------- shield
    if "SHIELD" in legal:
        sv = 0.0
        targets = []
        for x in rivals:
            if frozen[x]:
                continue
            p = pos[x]
            if own[p] != me or home[p]:
                continue
            b = base[p]
            if lvl[p] == 2:
                px = prog[x]
                if px >= 2 or (px == 1 and S["bait"][x] >= 1):
                    sv += 2.0 * b * T + 0.5 * (16.0 * b + b * T)
                    targets.append(x)
            else:
                if T <= 3:
                    sv += b * T + 0.5 * (8.0 * b + b * T)
                else:
                    sv += 4.5 * b
                targets.append(x)
        if targets:
            pstay = 0.85
            cost = best_rate * (1.0 + myprog)
            if sv * pstay - (1.0 - pstay) * 2.0 * best_rate > cost:
                S["last_act"] = "SHIELD"
                S["targets"] = targets
                return "SHIELD"

    # ------------------------------------------- dance around a trigger-happy shielder
    dn = S.get("danced")
    if dn is not None and dn[0] == t - 1:
        if S["shield_ready"][dn[1]] <= t:
            S["dance_fail"][dn[1]] += 1
    o = own[mypos]
    if (o and o != me and not home[mypos] and lvl[mypos] == 1 and myprog == 0 and T >= 4
            and plans and cands[plans[0][2]][7] == mypos
            and S["shield_ready"][o] <= t and not frozen[o]
            and (S["l1catch"][o] + 0.6) / (S["l1catch"][o] + 0.5 * S["l1free"][o] + 1.0) >= 0.5
            and S["dance_fail"][o] < 1
            and (dn is None or dn[0] != t - 1)):
        bestm = None
        bk = -1.0
        for name, dr, dc in _MOVES:
            if name not in legal:
                continue
            nr, nc = mr + dr, mc + dc
            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            i2 = nr * 8 + nc
            if occ[i2]:
                continue
            oo = own[i2]
            if oo == o and lvl[i2] == 1 and not home[i2]:
                k = 10.0 + base[i2]
            elif oo == 0:
                k = 6.0 + base[i2]
            elif oo != me and oo != o and lvl[i2] == 1 and not home[i2]:
                k = 4.0 + base[i2]
            else:
                k = 1.0
            if k > bk:
                bk = k
                bestm = name
        if bestm is not None:
            S["danced"] = (t, o)
            return bestm

    # ---------------------------------------------------------- act on plan
    for V, rate, j in plans:
        cd = cands[j]
        tr, tc = cd[0], cd[1]
        if tr == mr and tc == mc:
            return "STAY"
        a = _step(mr, mc, tr, tc, occ, own, base, me, legal)
        if a is not None:
            return a
    return "STAY"


_MOVES = (("NORTH", -1, 0), ("EAST", 0, 1), ("SOUTH", 1, 0), ("WEST", 0, -1))


def _step(mr, mc, tr, tc, occ, own, base, me, legal):
    d0 = abs(tr - mr) + abs(tc - mc)
    best = None
    bestk = -1.0
    for name, dr, dc in _MOVES:
        if name not in legal:
            continue
        nr, nc = mr + dr, mc + dc
        if not (0 <= nr < 8 and 0 <= nc < 8):
            continue
        if abs(tr - nr) + abs(tc - nc) >= d0:
            continue
        i = nr * 8 + nc
        if occ[i]:
            continue
        o = own[i]
        if o == 0:
            k = 2.0 * base[i]
        elif o != me:
            k = 3.0 * base[i]
        else:
            k = 0.0
        if k > bestk:
            bestk = k
            best = name
    return best
