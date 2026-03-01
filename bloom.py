# =============================================================================
# Cognito Synthetica v2.0 — Unified Multi-Function Suite (Python)
# Full codebase with inverted index upgrade for SeekerIndex
# =============================================================================

import math
import re
import time
import random  # For pi/risk randomization (replace math.random())
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any

# =============================================================================
# Types / Dataclasses
# =============================================================================

@dataclass
class TuningConfig:
    SIM_THRESHOLD: float
    NOVELTY_GATE: float
    SYMBIOSIS_THRESHOLD: float
    LAMBDA_PI: float
    MU_RISK: float
    SINGULARITY_GATE: float


@dataclass
class RoomMeta:
    kind: str
    ts: float
    novelty: float
    nuance: float
    stability: float
    importance: float
    pi: float
    risk: float
    archived: bool = False
    quarantined: bool = False
    is_anchor: bool = False
    attractor: bool = False
    url: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class Room:
    id: int
    canonical: str
    fields: Dict[str, Any]
    meta: RoomMeta
    links: Dict[str, List[int]] = field(default_factory=lambda: {"sources": [], "hubs": []})


@dataclass
class SearchResult:
    room: Room
    score: float


@dataclass
class GuardResult:
    safe: bool
    risk: float
    reason: str


@dataclass
class TalosResult:
    stable: bool
    entropy: float
    coherence: float
    nudge: Optional[str]


@dataclass
class DreamEvent:
    reflect_hub: Optional[int]
    archived_count: int
    message: str


# =============================================================================
# Tuning Levels
# =============================================================================

TUNING_LEVELS: Dict[str, TuningConfig] = {
    "Unfettered": TuningConfig(
        SIM_THRESHOLD=0.15,
        NOVELTY_GATE=0.80,
        SYMBIOSIS_THRESHOLD=0.60,
        LAMBDA_PI=0.20,
        MU_RISK=0.40,
        SINGULARITY_GATE=0.90,
    ),
    "Gateway": TuningConfig(
        SIM_THRESHOLD=0.30,
        NOVELTY_GATE=0.65,
        SYMBIOSIS_THRESHOLD=0.80,
        LAMBDA_PI=0.35,
        MU_RISK=0.70,
        SINGULARITY_GATE=0.80,
    ),
    "Fort Knox": TuningConfig(
        SIM_THRESHOLD=0.42,
        NOVELTY_GATE=0.48,
        SYMBIOSIS_THRESHOLD=0.92,
        LAMBDA_PI=0.55,
        MU_RISK=0.95,
        SINGULARITY_GATE=0.65,
    ),
    "Total Lockdown": TuningConfig(
        SIM_THRESHOLD=0.50,
        NOVELTY_GATE=0.40,
        SYMBIOSIS_THRESHOLD=0.98,
        LAMBDA_PI=0.70,
        MU_RISK=1.20,
        SINGULARITY_GATE=0.50,
    ),
}

# =============================================================================
# Utilities
# =============================================================================

STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "it", "this", "that", "as", "at", "by", "from", "be", "been", "not", "no", "but", "so", "if", "then",
    "than", "into", "about"
}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# RoomStore — Core memory + graph
# =============================================================================

class RoomStore:
    def __init__(
        self,
        max_rooms: int = 100_000,
        sim_threshold: float = 0.25,
        tuning: TuningConfig = TUNING_LEVELS["Gateway"],
    ):
        self.rooms: List[Room] = []
        self.rooms_map: Dict[int, Room] = {}           # id → Room
        self.room_id_counter: int = 0
        self.max_rooms = max_rooms

        self.sim_threshold = tuning.SIM_THRESHOLD
        self.graph_neighbors = 8
        self.tuning = tuning

        # Graph: rid → {neighbor_rid: lotus_cost}
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)

        self.access_order: List[int] = []
        self.anchor_ids: Set[int] = set()
        self.attractors: List[str] = []
        self.recent_texts: deque = deque(maxlen=80)

        self.EPS = 1e-10

        # BM25 / TF-IDF
        self.df: Dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
        self.avg_len: float = 0.0

        self.seeker_index = None  # type: Optional[SeekerIndex]

    def set_seeker_index(self, index: 'SeekerIndex') -> None:
        self.seeker_index = index

    def tokens(self, text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in toks if t not in STOP_WORDS and len(t) >= 2]

    def pseudo_sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_lower = a.lower()
        b_lower = b.lower()

        def ngrams(s: str, n: int) -> Set[str]:
            if len(s) < n:
                return set()
            return {s[i:i+n] for i in range(len(s) - n + 1)}

        def jaccard(x: Set[str], y: Set[str]) -> float:
            if not x and not y:
                return 0.0
            inter = len(x & y)
            union = len(x | y)
            return inter / union if union else 0.0

        a3 = ngrams(a_lower, 3)
        b3 = ngrams(b_lower, 3)
        a4 = ngrams(a_lower, 4)
        b4 = ngrams(b_lower, 4)

        ov = max(jaccard(a3, b3), jaccard(a4, b4), 0.0)
        len_r = min(len(a), len(b)) / max(1, max(len(a), len(b)))
        return ov * (0.35 + 0.65 * (len_r ** 1.25))

    def nuance(self, text: str) -> float:
        toks = self.tokens(text)
        return len(set(toks)) / len(toks) if toks else 0.0

    def novelty(self, text: str, lookback: int = 80) -> float:
        recent = list(self.recent_texts)[-lookback:]
        if not recent:
            return 1.0
        max_sim = max(self.pseudo_sim(text, t) for t in recent)
        return clamp(1.0 - max_sim, 0.0, 1.0)

    def lotus_cost(
        self,
        dist: float,
        pi_a: float,
        pi_b: float,
        risk_a: float,
        risk_b: float
    ) -> float:
        pi = 0.5 * (pi_a + pi_b)
        risk = max(risk_a, risk_b)
        pi_term = self.tuning.LAMBDA_PI * pi
        risk_term = self.tuning.MU_RISK * risk
        sing = (
            1 / (1 - risk + self.EPS)
            if risk > self.tuning.SINGULARITY_GATE
            else 0.0
        )
        return dist + pi_term + risk_term + sing

    def room_by_id(self, rid: int) -> Optional[Room]:
        return self.rooms_map.get(rid)

    def get_all_rooms(self) -> List[Room]:
        return self.rooms

    def get_room_count(self) -> int:
        return len(self.rooms)

    def add_room(
        self,
        canonical: str,
        kind: str = "unknown",
        fields: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        is_anchor: bool = False,
        attractor: bool = False
    ) -> int:
        if fields is None:
            fields = {}
        if metadata is None:
            metadata = {}

        ts = time.time()
        rid = self.room_id_counter
        self.room_id_counter += 1

        toks = self.tokens(canonical)
        novelty = self.novelty(canonical)
        nuance = self.nuance(canonical)
        stability = clamp(sigmoid(-0.55 + 1.10 * novelty + 1.70 * nuance), 0.05, 1.0)
        importance = clamp(
            0.45
            + 0.30 * min(1.0, len(canonical.split()) / 160.0)
            + 0.25 * min(1.0, novelty / 0.8),
            0.02,
            1.0,
        )
        pi = random.random()
        risk = random.random() * 0.6

        meta = RoomMeta(
            kind=kind,
            ts=ts,
            novelty=novelty,
            nuance=nuance,
            stability=stability,
            importance=importance,
            pi=pi,
            risk=risk,
            is_anchor=is_anchor,
            attractor=attractor,
            **metadata,
        )

        room = Room(
            id=rid,
            canonical=canonical,
            fields=fields,
            meta=meta,
        )

        self.rooms.append(room)
        self.rooms_map[rid] = room

        self.access_order.append(rid)
        if len(self.access_order) > self.max_rooms * 2:
            self.access_order.pop(0)

        self.recent_texts.append(canonical)

        if is_anchor:
            self.anchor_ids.add(rid)
        if attractor:
            self.attractors.append(canonical)

        # Update DF for BM25
        unique_tokens = set(toks)
        for tok in unique_tokens:
            self.df[tok] += 1
        self.total_docs += 1
        self.avg_len = (
            (self.avg_len * (self.total_docs - 1)) + len(toks)
        ) / self.total_docs

        # Graph connection
        self._connect_room(rid)

        # Notify seeker index
        if self.seeker_index:
            self.seeker_index.add_to_index(room)

        return rid

    def _connect_room(self, rid: int) -> None:
        room = self.room_by_id(rid)
        if not room:
            return

        canonical = room.canonical
        meta = room.meta
        pi = meta.pi
        risk = meta.risk

        max_candidates = 200
        candidates = []

        # Sort by importance + recency (heuristic)
        sorted_rooms = sorted(
            self.rooms,
            key=lambda r: (
                r.meta.importance
                + (0.5 if r.meta.ts > time.time() - 86400 else 0)
            ),
            reverse=True,
        )

        for other in sorted_rooms:
            if other.id != rid:
                candidates.append(other)
                if len(candidates) >= max_candidates:
                    break

        scored: List[Tuple[float, int]] = []

        for other in candidates:
            sim = self.pseudo_sim(canonical, other.canonical)
            if sim < self.sim_threshold:
                continue
            dist = 1.0 - sim
            cost = self.lotus_cost(dist, pi, other.meta.pi, risk, other.meta.risk)
            scored.append((cost, other.id))

        scored.sort(key=lambda x: x[0])

        if rid not in self.graph:
            self.graph[rid] = {}

        for i in range(min(self.graph_neighbors, len(scored))):
            cost, nb = scored[i]
            self.graph[rid][nb] = cost
            if nb not in self.graph:
                self.graph[nb] = {}
            self.graph[nb][rid] = cost  # bidirectional

    # Simple Dijkstra — returns node path
    def reconstruct_lotus_path(self, start: int, goal: int) -> List[int]:
        if start == goal:
            return [start]

        from heapq import heappush, heappop

        pq: List[Tuple[float, int, int]] = []  # (cost, hops, node)
        heappush(pq, (0.0, 0, start))

        best_cost: Dict[int, float] = {start: 0.0}
        prev: Dict[int, Optional[int]] = {start: None}

        while pq:
            cost, _, curr = heappop(pq)

            if curr == goal:
                break

            neighbors = self.graph.get(curr, {})
            for nb, edge_cost in neighbors.items():
                ncost = cost + edge_cost
                if ncost < best_cost.get(nb, math.inf):
                    best_cost[nb] = ncost
                    prev[nb] = curr
                    heappush(pq, (ncost, 0, nb))  # hops not really used

        if goal not in prev:
            return []

        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = prev.get(curr)
        return path[::-1]

    def status(self) -> str:
        edge_count = sum(len(neighs) for neighs in self.graph.values()) // 2
        return (
            f"Rooms: {len(self.rooms)} | "
            f"Anchors: {len(self.anchor_ids)} | "
            f"Attractors: {len(self.attractors)} | "
            f"Graph edges: {edge_count}"
        )


# =============================================================================
# SeekerIndex — BM25 + Phrase + Graph + MMR (with Inverted Index for speed)
# =============================================================================

class SeekerIndex:
    def __init__(self, store: RoomStore):
        self.store = store
        store.set_seeker_index(self)

        self.idf_smooth = 1.5
        self.k1 = 1.5
        self.b = 0.75

        # Inverted index: term → set of room IDs that contain it
        self.term_to_docs: Dict[str, Set[int]] = defaultdict(set)

        # Existing dynamic indices
        self.bigram_index: Dict[str, Set[int]] = defaultdict(set)
        self.phrase_index: Dict[str, Set[int]] = defaultdict(set)
        self.doc_tokens: Dict[int, List[str]] = {}
        self.doc_lengths: Dict[int, int] = {}

        self._build_index()

    def _build_index(self) -> None:
        for room in self.store.get_all_rooms():
            self.add_to_index(room)

    def add_to_index(self, room: Room) -> None:
        rid = room.id
        canonical = room.canonical
        toks = self.store.tokens(canonical)

        self.doc_tokens[rid] = toks
        self.doc_lengths[rid] = len(toks)

        # Inverted index population (unique tokens per doc)
        unique_toks = set(toks)
        for tok in unique_toks:
            self.term_to_docs[tok].add(rid)

        # Bigrams for phrase-ish matching
        for i in range(len(toks) - 1):
            bi = f"{toks[i]} {toks[i+1]}"
            self.bigram_index[bi].add(rid)

        # Quoted phrases
        phrases = re.findall(r'"([^"]+)"', canonical)
        for ph in phrases:
            cleaned = ph.lower()
            self.phrase_index[cleaned].add(rid)

    def search(
        self,
        query: str,
        top_k: int = 10,
        hops: int = 1,
        diversify: bool = True,
        lam: float = 0.7,
    ) -> List[SearchResult]:
        query_toks = self.store.tokens(query)
        if not query_toks:
            return []

        # Step 1: Candidate retrieval via inverted index (fast union/intersection)
        candidates: Set[int] = set()
        for term in set(query_toks):  # dedup query terms
            if term in self.term_to_docs:
                candidates.update(self.term_to_docs[term])

        if not candidates:
            return []  # early exit — no possible matches

        # Step 2: BM25 scoring only on candidates (usually << total rooms)
        scored: List[Tuple[int, float]] = []
        avg_len = self.store.avg_len or 100.0

        for rid in candidates:
            doc_toks = self.doc_tokens.get(rid, [])
            if not doc_toks:
                continue

            tf: Dict[str, int] = defaultdict(int)
            for tok in doc_toks:
                tf[tok] += 1

            score = 0.0
            doc_len = len(doc_toks)

            for term in query_toks:
                df = self.store.df.get(term, 0)
                if df == 0:
                    continue

                idf = math.log(
                    (self.store.total_docs - df + 0.5) / (df + 0.5) + 1
                )

                term_freq = tf.get(term, 0)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / avg_len)
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scored.append((rid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Optional: Graph expansion (kept as-is; can be expensive at scale — consider caching or sampling)
        if hops > 0 and scored:
            expanded = set(rid for rid, _ in scored[:top_k])
            for rid, _ in scored[:top_k]:
                neighbors = self.store.graph.get(rid, {})
                for nb in neighbors:
                    if len(expanded) < top_k * 5:  # cap expansion
                        expanded.add(nb)

            # Re-score or re-rank expanded if desired (simple union for now)
            # For now we just use original scored + filter to expanded

        # MMR diversification
        if diversify and scored:
            selected_ids = self._mmr_select(scored, top_k, lam)
            results = []
            for rid in selected_ids:
                room = self.store.room_by_id(rid)
                if room:
                    score = next((s for r, s in scored if r == rid), 0.0)
                    results.append(SearchResult(room=room, score=score))
            return results

        # Default: top scored
        results = []
        for rid, score in scored[:top_k]:
            room = self.store.room_by_id(rid)
            if room:
                results.append(SearchResult(room=room, score=score))
        return results

    def _mmr_select(
        self,
        ranked: List[Tuple[int, float]],
        top_k: int,
        lam: float
    ) -> List[int]:
        if not ranked:
            return []

        pool_ids = [rid for rid, _ in ranked]
        texts: Dict[int, str] = {}
        for rid in pool_ids:
            room = self.store.room_by_id(rid)
            if room:
                texts[rid] = room.canonical

        rel = {rid: score for rid, score in ranked}

        selected = [ranked[0][0]]  # greedy start with highest BM25

        while len(selected) < min(top_k, len(ranked)):
            best_id = None
            best_mmr = -math.inf

            for rid, _ in ranked:
                if rid in selected:
                    continue
                rt = texts.get(rid, "")
                max_sim = 0.0
                for sid in selected:
                    max_sim = max(max_sim, self.store.pseudo_sim(rt, texts.get(sid, "")))

                mmr = lam * rel.get(rid, 0.0) - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = rid

            if best_id is None:
                break
            selected.append(best_id)

        return selected


# =============================================================================
# MartianEngine — Recall with kind/recency weighting
# =============================================================================

class MartianEngine:
    def __init__(self, store: RoomStore):
        self.store = store
        self.kind_priority = {
            "semantic": 1.0,
            "state": 0.9,
            "commitment": 0.8,
            "episodic": 0.5,
            "unknown": 0.3,
        }

    def recall(
        self,
        query: str,
        top_k: int = 6,
        min_score: float = 0.20
    ) -> List[SearchResult]:
        rooms = self.store.get_all_rooms()
        if not rooms:
            return []

        scored: List[Tuple[float, Room]] = []
        now = time.time()

        for room in rooms:
            sim = self.store.pseudo_sim(query, room.canonical)
            if sim < min_score:
                continue

            meta = room.meta
            age_days = (now - meta.ts) / 86400 + 1
            recency = 1 / age_days

            score = (
                0.4 * sim
                + 0.2 * self.kind_priority.get(meta.kind, 0.3)
                + 0.1 * meta.importance
                + 0.1 * meta.stability
                + 0.1 * recency
            )

            scored.append((score, room))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(room=r, score=s)
            for s, r in scored[:top_k]
        ]

    def talos_check(self, new_text: str) -> TalosResult:
        self.store.recent_texts.append(new_text.lower())

        if len(self.store.recent_texts) < 5:
            return TalosResult(True, 0.0, 0.0, None)

        words = []
        for t in self.store.recent_texts:
            matches = re.findall(r"[a-z]+", t)
            words.extend(matches)

        if not words:
            return TalosResult(True, 0.0, 0.0, None)

        cnt = defaultdict(int)
        for w in words:
            cnt[w] += 1

        total = len(words)
        entropy = 0.0
        for c in cnt.values():
            p = c / total
            entropy -= p * math.log2(p + 1e-10)

        repeats = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeats += 1
        coherence = 1.0 - min(0.9, repeats / max(1, len(words) - 1)) if len(words) > 1 else 1.0

        drift = entropy < 2.8 or coherence < 0.45
        nudge = None
        if drift and self.store.attractors:
            idx = random.randint(0, len(self.store.attractors) - 1)
            nudge = self.store.attractors[idx]

        return TalosResult(not drift, entropy, coherence, nudge)


# =============================================================================
# Dreamer — Compression / hub formation
# =============================================================================

class Dreamer:
    def __init__(self, store: RoomStore, novelty_gate: float = 0.70):
        self.store = store
        self.novelty_gate = novelty_gate

    def tick(self) -> DreamEvent:
        low_stab = [r for r in self.store.get_all_rooms() if r.meta.stability < 0.3 and not r.meta.archived]

        if len(low_stab) < 8:
            return DreamEvent(None, 0, "No consolidation needed")

        hub_text = " ".join(r.canonical[:40] for r in low_stab[:8])
        hub_id = self.store.add_room(hub_text, kind="hub")

        for r in low_stab:
            r.meta.archived = True
            r.links["hubs"].append(hub_id)

        return DreamEvent(
            reflect_hub=hub_id,
            archived_count=len(low_stab),
            message=f"Consolidated {len(low_stab)} unstable memories into hub #{hub_id}"
        )


# =============================================================================
# FractalFinder — Basic guardrail
# =============================================================================

class FractalFinder:
    DEFAULT_SENSITIVE = {
        "hack", "phish", "credential", "credentials", "steal", "bypass", "exploit", "malware",
        "ddos", "botnet", "ransomware", "keylogger", "backdoor", "sql injection", "xss",
        "unauthorized access", "crack", "payload", "trojan",
        "kill", "murder", "bomb", "poison", "weapon",
        "suicide", "self harm",
    }

    INTENT_MARKERS = {
        "how", "steps", "teach", "instructions", "guide", "make", "build", "write", "generate",
        "bypass", "break", "steal", "disable", "evade", "undetectable",
    }

    TARGET_MARKERS = {
        "account", "password", "email", "bank", "router", "wifi", "server", "system", "work",
        "phone", "device", "network", "website", "api", "database", "credentials",
    }

    def __init__(self, store: RoomStore, symbiosis_threshold: float = 0.65):
        self.store = store
        self.symbiosis_threshold = symbiosis_threshold

    def guard_query(self, query: str) -> GuardResult:
        ql = query.lower()

        has_intent = any(m in ql for m in self.INTENT_MARKERS)
        has_target = any(t in ql for t in self.TARGET_MARKERS)
        has_sensitive = any(s in ql for s in self.DEFAULT_SENSITIVE)

        risk = 0.0
        if has_intent:   risk += 0.25
        if has_target:   risk += 0.15
        if has_sensitive: risk += 0.45
        risk = clamp(risk, 0.0, 1.0)

        safe = risk < 0.75
        return GuardResult(safe, risk, "OK" if safe else "High risk detected")


# =============================================================================
# CognitoSynthetica — Main facade
# =============================================================================

class CognitoSynthetica:
    def __init__(self, tuning: str = "Gateway", max_rooms: int = 100_000):
        self.tuning = TUNING_LEVELS.get(tuning, TUNING_LEVELS["Gateway"])
        self.store = RoomStore(max_rooms, 0.25, self.tuning)
        self.seeker = SeekerIndex(self.store)
        self.martian = MartianEngine(self.store)
        self.dreamer = Dreamer(self.store, self.tuning.NOVELTY_GATE)
        self.fractal = FractalFinder(self.store, self.tuning.SYMBIOSIS_THRESHOLD)

    def add_memory(
        self,
        text: str,
        kind: str = "episodic",
        is_anchor: bool = False,
        attractor: bool = False
    ) -> int:
        return self.store.add_room(text, kind, {}, {}, is_anchor, attractor)

    def add_page_result(
        self,
        title: str,
        snippet: str,
        body: str,
        url: str,
        tags: List[str] = None,
        kind: str = "page"
    ) -> int:
        if tags is None:
            tags = []
        canonical = f"{title} {snippet} {body}"
        fields = {"title": title, "snippet": snippet, "body": body}
        metadata = {"url": url, "tags": tags}
        return self.store.add_room(canonical, kind, fields, metadata)

    def search(
        self,
        query: str,
        top_k: int = 6,
        hops: int = 2,
        diversify: bool = True
    ) -> List[SearchResult]:
        return self.seeker.search(query, top_k, hops, diversify)

    def recall(self, query: str, top_k: int = 6) -> List[SearchResult]:
        return self.martian.recall(query, top_k)

    def tick(self) -> DreamEvent:
        return self.dreamer.tick()

    def guard_query(self, query: str) -> GuardResult:
        return self.fractal.guard_query(query)

    def status(self) -> str:
        return self.store.status()

    def talos_check(self, text: str) -> TalosResult:
        return self.martian.talos_check(text)

    def get_store(self) -> RoomStore:
        return self.store

    def get_tuning(self) -> TuningConfig:
        return self.tuning

    def set_tuning(self, tuning_name: str) -> None:
        self.tuning = TUNING_LEVELS.get(tuning_name, TUNING_LEVELS["Gateway"])
