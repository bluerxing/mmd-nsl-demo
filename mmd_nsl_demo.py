#!/usr/bin/env python3
"""
MMD-NSL Minimal Working Example
================================
Corresponds to Algorithm 1 in the paper:
  Lines 1      : Initialize phi (Transformer), theta (rule weights)
  Lines 3-7    : Upper-level optimization (rule discovery via T_phi)
  Lines 8-14   : Lower-level optimization (grounding + scoring on local KGs)
  Lines 15-17  : Feedback to upper-level (posterior -> retrain T_phi)

Key Components:
  T_phi (Transformer)       ->  RuleGenerator class
  C_k = <NER_h, NER_t>      ->  triple = (rel, head_type, tail_type)
  F_rule(z_j | C_k)         ->  ground_rule_on_graph() function
  theta_j rule weights       ->  RuleScorer class
  Bilevel iteration           ->  run_algorithm1() function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from torch.distributions import Categorical
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 0. Data Definition: 3 Document-Level KGs
# ==============================================================

NER_TYPES = ["Person", "Location"]
RELATIONS = ["works_in", "born_in", "located_in", "capital_of"]
INV_RELATIONS = ["~works_in", "~born_in", "~located_in", "~capital_of"]
ALL_RELS = RELATIONS + INV_RELATIONS + ["<STOP>"]

R = len(RELATIONS)       # 4
V = 2 * R + 1            # 9 = 4 forward + 4 inverse + 1 stop token

DOCUMENTS = [
    {
        # Doc A: Alice born_in Rome, Rome located_in Italy, Alice born_in Italy (all KNOWN)
        #   => Mining discovers: for <Person,Location>+born_in, path born_in^located_in exists
        #   Also: Alice works_in CompanyX, CompanyX works_in Rome
        "title": "Doc A",
        "entities": [
            {"name": "Alice",    "type": 0},  # Person [0]
            {"name": "CompanyX", "type": 0},  # Person [1] (simplified)
            {"name": "Rome",     "type": 1},  # Location [2]
            {"name": "Italy",    "type": 1},  # Location [3]
        ],
        "triples": [
            (0, 2, 1),  # Alice    --born_in-->     Rome
            (2, 3, 2),  # Rome     --located_in-->  Italy
            (0, 3, 1),  # Alice    --born_in-->     Italy    <-- this triple
            #              makes born_in^located_in a co-occurring path!
            (0, 1, 0),  # Alice    --works_in-->    CompanyX
            (1, 2, 0),  # CompanyX --works_in-->    Rome
            (0, 2, 0),  # Alice    --works_in-->    Rome     <-- makes
            #              works_in^works_in a co-occurring path for <Pers,Loc>+works_in
        ],
        "queries": [],  # Doc A is purely for training (rule mining source)
    },
    {
        # Doc B: Bob born_in Berlin, Berlin located_in Germany, Bob born_in Germany (KNOWN)
        #   => Reinforces born_in^located_in rule in the same NER context
        "title": "Doc B",
        "entities": [
            {"name": "Bob",     "type": 0},  # Person [0]
            {"name": "Berlin",  "type": 1},  # Location [1]
            {"name": "Germany", "type": 1},  # Location [2]
        ],
        "triples": [
            (0, 1, 1),  # Bob    --born_in-->     Berlin
            (1, 2, 2),  # Berlin --located_in-->  Germany
            (0, 2, 1),  # Bob    --born_in-->     Germany  <-- reinforces rule
        ],
        "queries": [],  # also training
    },
    {
        # Doc C: Carol born_in Tokyo, Tokyo located_in Japan
        #   Query: Carol --born_in--> Japan ?
        #   The rule born_in^located_in was learned from Doc A + B,
        #   and now TRANSFERS to Doc C via shared <Person,Location> context!
        #   Grounding: Carol->Tokyo->Japan  YES!
        "title": "Doc C (TEST - cross-graph transfer)",
        "entities": [
            {"name": "Carol", "type": 0},  # Person [0]
            {"name": "Tokyo", "type": 1},  # Location [1]
            {"name": "Japan", "type": 1},  # Location [2]
        ],
        "triples": [
            (0, 1, 1),  # Carol --born_in-->     Tokyo
            (1, 2, 2),  # Tokyo --located_in-->  Japan
        ],
        "queries": [(0, 2, 1)],  # predict: Carol --born_in--> Japan ?
        # CROSS-GRAPH TRANSFER: rule born_in^located_in learned from Doc A+B
        # grounds here as Carol->Tokyo->Japan !
    },
    {
        # Doc D: Dan works_in Samsung, Samsung works_in Seoul, Seoul located_in Korea
        #   Query: Dan --works_in--> Seoul ?
        #   Rule works_in^works_in learned from Doc A transfers here!
        #   Grounding: Dan->Samsung->Seoul  YES!
        "title": "Doc D (TEST - cross-graph transfer)",
        "entities": [
            {"name": "Dan",     "type": 0},  # Person [0]
            {"name": "Samsung", "type": 0},  # Person [1] (simplified)
            {"name": "Seoul",   "type": 1},  # Location [2]
            {"name": "Korea",   "type": 1},  # Location [3]
        ],
        "triples": [
            (0, 1, 0),  # Dan     --works_in-->    Samsung
            (1, 2, 0),  # Samsung --works_in-->    Seoul
            (2, 3, 2),  # Seoul   --located_in-->  Korea
        ],
        "queries": [
            (0, 2, 0),  # predict: Dan --works_in--> Seoul ?
            # works_in^works_in: Dan->Samsung->Seoul YES!
            (0, 3, 1),  # predict: Dan --born_in--> Korea ?
            # This one should NOT ground (no born_in edge for Dan)
        ],
    },
]


def display_data():
    """Pretty-print the 3 document-level KGs."""
    print("=" * 70)
    print("DOCUMENT-LEVEL KNOWLEDGE GRAPHS")
    print("=" * 70)
    print("")
    print("NER Types: %s" % NER_TYPES)
    print("Relations: %s  (+ inverses + <STOP>)" % RELATIONS)
    print("Vocabulary size: 2R+1 = %d tokens" % V)
    print("  Token IDs: ", end="")
    for i, name in enumerate(ALL_RELS):
        print("%d=%s" % (i, name), end="  ")
    print("")

    for doc in DOCUMENTS:
        print("")
        print("--- %s ---" % doc["title"])
        print("  Entities:")
        for i, e in enumerate(doc["entities"]):
            print("    [%d] %s  (%s)" % (i, e["name"], NER_TYPES[e["type"]]))
        print("  Known triples (edges):")
        for h, t, r in doc["triples"]:
            hn = doc["entities"][h]["name"]
            tn = doc["entities"][t]["name"]
            print("    %s --%s--> %s" % (hn, RELATIONS[r], tn))

        print("  KG structure:")
        enames = [e["name"] for e in doc["entities"]]
        # Build simple visual
        for h, t, r in doc["triples"]:
            print("    %s ──%s──> %s" % (
                enames[h].ljust(10), RELATIONS[r], enames[t]))

        print("  Queries to predict:")
        for qh, qt, qr in doc["queries"]:
            print("    %s --%s--> %s  ?" % (
                enames[qh], RELATIONS[qr], enames[qt]))

    print("")
    print("  Key observation: Each doc is an INDEPENDENT KG.")
    print("  Entities are disjoint, but NER types {Person, Location} are SHARED.")
    print("  This shared NER space enables cross-graph rule transfer.")


# ==============================================================
# 1. Mine Co-occurrence Rules (Initialize prior)
# ==============================================================

def mine_rules_from_documents(documents, max_depth=2):
    """
    For each triple type (relation, head_NER, tail_NER), count co-occurring
    relational paths across ALL documents (co-occurrence mining).
    """
    rule_counter = defaultdict(Counter)

    for doc in documents:
        entities = doc["entities"]
        adj = defaultdict(lambda: defaultdict(list))
        for h, t, r in doc["triples"]:
            adj[h][t].append(r)
            adj[t][h].append(r + R)

        paths_by_depth = {1: {}}
        for h in adj:
            for t in adj[h]:
                for r in adj[h][t]:
                    paths_by_depth[1].setdefault(h, {}).setdefault(t, []).append((r,))

        for depth in range(2, max_depth + 1):
            paths_by_depth[depth] = {}
            for h in paths_by_depth[depth - 1]:
                for mid in paths_by_depth[depth - 1][h]:
                    if mid not in paths_by_depth[1]:
                        continue
                    for t in paths_by_depth[1][mid]:
                        if t == h:
                            continue
                        for pp in paths_by_depth[depth - 1][h][mid]:
                            for ep in paths_by_depth[1][mid][t]:
                                paths_by_depth[depth].setdefault(
                                    h, {}).setdefault(t, []).append(pp + ep)

        for h, t, r in doc["triples"]:
            ctx = (r, entities[h]["type"], entities[t]["type"])
            all_paths = set()
            for depth in range(1, max_depth + 1):
                if h in paths_by_depth[depth] and t in paths_by_depth[depth][h]:
                    for p in paths_by_depth[depth][h][t]:
                        all_paths.add(p)
            rule_counter[ctx].update(all_paths)

    context_rules = {}
    for ctx, counter in rule_counter.items():
        total = sum(counter.values())
        rules = list(counter.keys())
        probs = [counter[r] / total for r in rules]
        context_rules[ctx] = {"rules": rules, "probs": probs}
    return context_rules


# ==============================================================
# 2. Upper Level: T_phi -- Context-Conditioned Rule Generator
#    (Alg.1 lines 3-7)
# ==============================================================

class RuleGenerator(nn.Module):
    """
    Autoregressive Transformer T_phi.
    Models P(rule_body | C_k(h), r_head, C_k(t)).

    Encoder source (3 tokens):
        src = [ ent_emb(NER_head), rel_emb(r_head), ent_emb(NER_tail) ]
                   ^                   ^                  ^
               context C_k          rule head          context C_k

    Decoder: generates body tokens r_1^j, r_2^j, ... autoregressively.
    """

    def __init__(self, hidden=64, max_depth=2, n_rel=4, n_ent_types=2, n_layers=1):
        super().__init__()
        self.R = n_rel
        self.V = 2 * n_rel + 1
        self.max_depth = max_depth
        self.hidden = hidden

        self.rel_emb = nn.Embedding(self.V, hidden)
        self.ent_emb = nn.Embedding(n_ent_types, hidden)
        self.transformer = nn.Transformer(
            d_model=hidden, nhead=4,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=128, batch_first=False
        )
        self.proj = nn.Linear(hidden, self.V)
        self.loss_fn = nn.CrossEntropyLoss()

    def _causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward_logits(self, chains, h_types, t_types):
        B, L = chains.shape
        src = torch.stack([
            self.ent_emb(h_types),
            self.rel_emb(chains[:, 0]),
            self.ent_emb(t_types),
        ], dim=0)
        tgt_emb = self.rel_emb(chains[:, 1:]).transpose(0, 1)
        start = torch.zeros(1, B, self.hidden, device=chains.device)
        tgt_in = torch.cat([start, tgt_emb], dim=0)
        tgt_mask = self._causal_mask(L, chains.device)
        out = self.transformer(src, tgt_in, tgt_mask=tgt_mask)
        return self.proj(out.transpose(0, 1))

    def compute_loss(self, chains, h_types, t_types):
        logits = self.forward_logits(chains[:, :-1], h_types, t_types)
        targets = chains[:, 1:]
        return self.loss_fn(logits.reshape(-1, self.V), targets.reshape(-1))

    @torch.no_grad()
    def sample_rules(self, context_triples, N=20):
        """Alg.1 line 5: generate rule bodies by current phi."""
        self.eval()
        results = {}
        for (rel, ht, tt) in context_triples:
            h_types = torch.LongTensor([ht] * N)
            t_types = torch.LongTensor([tt] * N)
            rels = torch.LongTensor([rel] * N)

            src = torch.stack([
                self.ent_emb(h_types),
                self.rel_emb(rels),
                self.ent_emb(t_types),
            ], dim=0)
            tgt_in = torch.zeros(1, N, self.hidden)
            chains = []
            log_probs = torch.zeros(N)

            for step in range(self.max_depth):
                tgt_mask = self._causal_mask(step + 1, src.device)
                out = self.transformer(src, tgt_in, tgt_mask=tgt_mask)[-1]
                probs = self.proj(out).softmax(dim=-1)
                dist = Categorical(probs)
                sampled = dist.sample()
                log_probs += dist.log_prob(sampled)
                chains.append(sampled)
                tgt_in = torch.cat([
                    torch.zeros(N, 1, self.hidden),
                    self.rel_emb(torch.stack(chains, dim=-1))
                ], dim=1).transpose(0, 1)

            chains_t = torch.stack(chains, dim=-1)
            seen = {}
            for i in range(N):
                key = tuple(chains_t[i].tolist())
                if key not in seen:
                    seen[key] = log_probs[i].item()
            results[(rel, ht, tt)] = list(seen.items())
        self.train()
        return results

    @torch.no_grad()
    def sample_rules_verbose(self, rel, ht, tt, N=1):
        """
        Token-by-token generation with full probability trace.
        Shows exactly how the decoder produces each body token.
        """
        self.eval()
        h_types = torch.LongTensor([ht] * N)
        t_types = torch.LongTensor([tt] * N)
        rels = torch.LongTensor([rel] * N)

        # ---- Encoder ----
        src = torch.stack([
            self.ent_emb(h_types),   # position 0: NER_head
            self.rel_emb(rels),      # position 1: r_head
            self.ent_emb(t_types),   # position 2: NER_tail
        ], dim=0)  # [3, N, H]

        print("    Encoder source tokens:")
        print("      pos 0: ent_emb(NER_head=%s)  -> vec in R^%d" % (
            NER_TYPES[ht], self.hidden))
        print("      pos 1: rel_emb(r_head=%s)    -> vec in R^%d" % (
            RELATIONS[rel], self.hidden))
        print("      pos 2: ent_emb(NER_tail=%s)  -> vec in R^%d" % (
            NER_TYPES[tt], self.hidden))
        print("")

        # ---- Decoder: step by step ----
        tgt_in = torch.zeros(1, N, self.hidden)  # start with zero vector
        chains = []
        total_log_prob = 0.0

        for step in range(self.max_depth):
            tgt_mask = self._causal_mask(step + 1, src.device)
            out = self.transformer(src, tgt_in, tgt_mask=tgt_mask)[-1]  # [N, H]
            logits = self.proj(out)           # [N, V]
            probs = logits.softmax(dim=-1)    # [N, V]

            # Show probability distribution over all tokens
            print("    Decoder step %d:" % (step + 1))
            print("      Input: [<START>%s]" % (
                "".join(", %s" % ALL_RELS[c] for c in chains)))
            print("      Output P(token | context, history):")

            p = probs[0]
            # Sort by probability descending
            sorted_probs, sorted_ids = p.sort(descending=True)
            for rank in range(min(V, 5)):  # show top-5
                tid = sorted_ids[rank].item()
                prob_val = sorted_probs[rank].item()
                bar = "#" * int(prob_val * 40)
                marker = ""
                print("        [%d] %-14s  P=%.4f  %s %s" % (
                    tid, ALL_RELS[tid], prob_val, bar, marker))
            if V > 5:
                remaining = sum(sorted_probs[5:].tolist())
                print("        ... (remaining %d tokens total P=%.4f)" % (
                    V - 5, remaining))

            # Sample
            dist = Categorical(probs)
            sampled = dist.sample()
            chosen_id = sampled[0].item()
            chosen_prob = probs[0, chosen_id].item()
            total_log_prob += dist.log_prob(sampled)[0].item()
            chains.append(sampled)

            print("      >>> Sampled token: [%d] %s  (P=%.4f)" % (
                chosen_id, ALL_RELS[chosen_id], chosen_prob))
            print("")

            # Update decoder input
            tgt_in = torch.cat([
                torch.zeros(N, 1, self.hidden),
                self.rel_emb(torch.stack(chains, dim=-1))
            ], dim=1).transpose(0, 1)

        body = tuple(c[0].item() for c in chains)
        body_parts = [ALL_RELS[r] for r in body if r != 2 * R]
        body_display = " ^ ".join(body_parts) if body_parts else "<empty>"
        print("    Final rule body: %s" % body_display)
        print("    Full rule: %s <- %s" % (RELATIONS[rel], body_display))
        print("    Total log P(body | context) = %.4f" % total_log_prob)

        self.train()
        return body, total_log_prob


# ==============================================================
# 3. Lower Level: Rule Grounding & Scoring
#    (Alg.1 lines 8-14)
# ==============================================================

def ground_rule_on_graph(doc, rule_body, head_idx, tail_idx):
    """
    F_rule(z_j | C_k): Ground rule body on local KG via path search.
    Corresponds to Eq.4: product of e_path along body, max over graph paths.
    Uses max-product DP over transition probabilities on the local KG.
    Here simplified to binary 0/1 for clarity.
    """
    adj = defaultdict(lambda: defaultdict(set))
    for h, t, r in doc["triples"]:
        adj[h][t].add(r)
        adj[t][h].add(r + R)

    current = {head_idx}
    for rel_id in rule_body:
        next_set = set()
        for node in current:
            for neighbor in adj[node]:
                if rel_id in adj[node][neighbor]:
                    next_set.add(neighbor)
        current = next_set
        if not current:
            return 0.0
    return 1.0 if tail_idx in current else 0.0


class RuleScorer(nn.Module):
    """
    Learns theta_j (rule weights) and bias per NER context.
    Score(h,t,r) = sum_j theta_j * F_rule(z_j) + bias
    Corresponds to Alg.1 line 11-12 optimization.
    """
    def __init__(self, context_rules):
        super().__init__()
        self.context_rules = context_rules
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        for ctx, rules in context_rules.items():
            key = "%d_%d_%d" % (ctx[0], ctx[1], ctx[2])
            self.weights[key] = nn.Parameter(torch.randn(len(rules)) * 0.1)
            self.biases[key] = nn.Parameter(torch.zeros(1))

    def score_query(self, doc, query_h, query_t, query_r):
        h_type = doc["entities"][query_h]["type"]
        t_type = doc["entities"][query_t]["type"]
        ctx = (query_r, h_type, t_type)
        key = "%d_%d_%d" % (ctx[0], ctx[1], ctx[2])
        if ctx not in self.context_rules:
            return self.biases.get(key, torch.zeros(1))
        rules = self.context_rules[ctx]
        gs = torch.tensor([ground_rule_on_graph(doc, rb, query_h, query_t)
                           for rb, _ in rules])
        return (self.weights[key] * gs).sum() + self.biases[key]


# ==============================================================
# 4. Helpers
# ==============================================================

def body_str(rule_body):
    parts = [ALL_RELS[r] for r in rule_body if r != 2 * R]
    return " ^ ".join(parts) if parts else "<empty>"


def prepare_training_data(context_rules, max_depth=2):
    data = []
    STOP = 2 * R
    for (rel, ht, tt), info in context_rules.items():
        for rule_body, prob in zip(info["rules"], info["probs"]):
            n_copies = max(1, int(prob * 20))
            padded = list(rule_body) + [STOP] * (max_depth - len(rule_body))
            chain = [rel] + padded
            for _ in range(n_copies):
                data.append((chain, ht, tt))
    return data


# ==============================================================
# 5. Main: Algorithm 1 -- Bilevel Optimization
# ==============================================================

def run_algorithm1(documents, n_iters=2, n_samples=20):
    STOP = 2 * R
    max_depth = 2

    # ============================================================
    # Display data
    # ============================================================
    display_data()

    # ============================================================
    # Alg.1 Line 1: Initialize parameters phi, theta
    # ============================================================
    print("")
    print("=" * 70)
    print("Alg.1 Line 1: Initialize -- Mine co-occurrence rules for prior")
    print("=" * 70)
    initial_rules = mine_rules_from_documents(documents, max_depth=max_depth)

    for ctx, info in initial_rules.items():
        print("")
        print("  Context C_%d = <%s, %s>,  r_head = %s" % (
            list(initial_rules.keys()).index(ctx),
            NER_TYPES[ctx[1]], NER_TYPES[ctx[2]], RELATIONS[ctx[0]]))
        for rule, prob in sorted(zip(info["rules"], info["probs"]),
                                 key=lambda x: -x[1])[:5]:
            print("    P(body = %s) = %.3f" % (body_str(rule), prob))

    # Pretrain T_phi on co-occurrence distribution
    print("")
    print("=" * 70)
    print("Alg.1 Line 1: Pretrain Transformer T_phi")
    print("  Encoder src = [ent_emb(NER_h), rel_emb(r_head), ent_emb(NER_t)]")
    print("  Decoder generates body tokens autoregressively")
    print("=" * 70)
    generator = RuleGenerator(hidden=64, max_depth=max_depth,
                              n_rel=R, n_ent_types=len(NER_TYPES))
    train_data = prepare_training_data(initial_rules, max_depth=max_depth)
    opt_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)

    for epoch in range(30):
        total_loss = 0
        for chain, ht, tt in train_data:
            loss = generator.compute_loss(
                torch.LongTensor([chain]),
                torch.LongTensor([ht]),
                torch.LongTensor([tt]))
            opt_gen.zero_grad(); loss.backward(); opt_gen.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print("  Epoch %d: avg cross-entropy loss = %.4f" % (
                epoch + 1, total_loss / len(train_data)))

    # ============================================================
    # Demonstrate token-by-token generation BEFORE EM
    # ============================================================
    print("")
    print("=" * 70)
    print("TOKEN-BY-TOKEN AUTOREGRESSIVE GENERATION (before EM)")
    print("  Showing how T_phi generates a rule body step by step")
    print("=" * 70)

    context_triples = list(initial_rules.keys())

    for ctx in context_triples:
        rel, ht, tt = ctx
        print("")
        print("  ---- Context: <%s, %s>,  r_head = %s ----" % (
            NER_TYPES[ht], NER_TYPES[tt], RELATIONS[rel]))
        print("")
        generator.sample_rules_verbose(rel, ht, tt, N=1)
        print("")

    # ============================================================
    # Alg.1 Line 2: for each iteration do
    # ============================================================
    for iteration in range(1, n_iters + 1):
        print("")
        print("*" * 70)
        print("Alg.1 Line 2: ITERATION %d" % iteration)
        print("*" * 70)

        # ========================================================
        # Alg.1 Lines 3-7: Upper-level optimization
        #   for each context C_k:
        #     Line 5: Solve Eq.11.a, generate rule bodies r_1^j ^ ... ^ r_L^j
        #     Line 6: Compute context distribution pi_k(phi)
        # ========================================================
        print("")
        print("  " + "-" * 60)
        print("  Alg.1 Lines 3-7: Upper-level optimization")
        print("  For each context C_k, generate rule bodies via T_phi")
        print("  and compute context distribution pi_k(phi)")
        print("  " + "-" * 60)

        sampled = generator.sample_rules(context_triples, N=n_samples)

        for ctx, rules_list in sampled.items():
            rn = RELATIONS[ctx[0]]
            hn = NER_TYPES[ctx[1]]
            tn = NER_TYPES[ctx[2]]
            print("")
            print("    C_k = <%s, %s>,  r_head = %s" % (hn, tn, rn))
            print("    pi_k(phi) = {")
            total = len(rules_list)
            for rb, lp in sorted(rules_list, key=lambda x: -x[1]):
                print("      z_j = %-25s  log pi = %+.3f" % (body_str(rb), lp))
            print("    }")

        # Show ONE detailed token-by-token trace for this iteration
        print("")
        print("  Detailed token-by-token trace (1 sample for 1st context):")
        first_ctx = context_triples[0]
        generator.sample_rules_verbose(first_ctx[0], first_ctx[1], first_ctx[2])

        # ========================================================
        # Alg.1 Lines 8-14: Lower-level optimization
        #   for each context C_k:
        #     Line 10: Process local KG evidence -> F_rule(z_j | C_k)
        #     Line 11: Maximize log-likelihood treating pi_k(phi) as fixed
        #     Line 12: Update theta*(phi) as posterior rule weights
        #     Line 13: Update path embeddings e_path
        # ========================================================
        print("")
        print("  " + "-" * 60)
        print("  Alg.1 Lines 8-14: Lower-level optimization")
        print("  Ground rules on local KGs, learn theta_j")
        print("  " + "-" * 60)

        scorer = RuleScorer(sampled)
        opt_scorer = torch.optim.Adam(scorer.parameters(), lr=0.05)

        # Line 10: Show grounding examples
        print("")
        print("    Line 10: F_rule(z_j | C_k) -- grounding on local KGs:")
        for doc in documents:
            for qh, qt, qr in doc["queries"]:
                h_type = doc["entities"][qh]["type"]
                t_type = doc["entities"][qt]["type"]
                ctx = (qr, h_type, t_type)
                if ctx not in sampled:
                    continue
                hn = doc["entities"][qh]["name"]
                tn = doc["entities"][qt]["name"]
                print("")
                print("      %s: query %s --%s--> %s" % (
                    doc["title"], hn, RELATIONS[qr], tn))
                for rb, lp in sampled[ctx]:
                    gs = ground_rule_on_graph(doc, rb, qh, qt)
                    mark = "YES" if gs > 0 else "no"
                    # Show path trace
                    if gs > 0:
                        print("        F_rule(%s) = %.1f  [%s]" % (
                            body_str(rb), gs, mark))
                        # trace the grounding path
                        adj = defaultdict(lambda: defaultdict(set))
                        for h, t, r in doc["triples"]:
                            adj[h][t].add(r)
                            adj[t][h].add(r + R)
                        current = {qh}
                        path_desc = doc["entities"][qh]["name"]
                        for ri, rel_id in enumerate(rb):
                            if rel_id == 2 * R:
                                break
                            next_set = set()
                            for node in current:
                                for nb in adj[node]:
                                    if rel_id in adj[node][nb]:
                                        next_set.add(nb)
                            if next_set:
                                mid = list(next_set)[0]
                                path_desc += " --%s--> %s" % (
                                    ALL_RELS[rel_id], doc["entities"][mid]["name"])
                            current = next_set
                        print("          path: %s" % path_desc)
                    else:
                        print("        F_rule(%s) = %.1f  [%s]" % (
                            body_str(rb), gs, mark))

        # Line 11-12: Train scorer (maximize log-likelihood)
        for epoch in range(30):
            total_loss = 0
            for doc in documents:
                for qh, qt, qr in doc["queries"]:
                    logit = scorer.score_query(doc, qh, qt, qr)
                    loss = F.binary_cross_entropy_with_logits(logit, torch.tensor([1.0]))
                    opt_scorer.zero_grad(); loss.backward(); opt_scorer.step()
                    total_loss += loss.item()

        # Also train on known triples as positive examples
        for epoch in range(30):
            for doc in documents:
                for h, t, r in doc["triples"]:
                    h_type = doc["entities"][h]["type"]
                    t_type = doc["entities"][t]["type"]
                    ctx = (r, h_type, t_type)
                    if ctx not in sampled:
                        continue
                    logit = scorer.score_query(doc, h, t, r)
                    loss = F.binary_cross_entropy_with_logits(logit, torch.tensor([1.0]))
                    opt_scorer.zero_grad(); loss.backward(); opt_scorer.step()

        print("")
        print("    Lines 11-12: theta*(phi) learned weights:")
        for ctx in context_triples:
            key = "%d_%d_%d" % (ctx[0], ctx[1], ctx[2])
            if key in scorer.weights:
                rn = RELATIONS[ctx[0]]
                hn = NER_TYPES[ctx[1]]
                tn = NER_TYPES[ctx[2]]
                w = scorer.weights[key].data
                rules_list = sampled[ctx]
                print("      <%s, %s> + %s:" % (hn, tn, rn))
                for (rb, _), wi in sorted(zip(rules_list, w.tolist()),
                                          key=lambda x: -x[1]):
                    print("        theta=%+.3f  z_j = %s" % (wi, body_str(rb)))

        # ========================================================
        # Alg.1 Lines 15-17: Feedback to upper-level
        #   Line 16: Use theta*(phi) to draw new training samples
        #   Line 17: Update phi to improve future rule generation
        # ========================================================
        print("")
        print("  " + "-" * 60)
        print("  Alg.1 Lines 15-17: Feedback to upper-level")
        print("  Posterior combines prior pi_k with theta* from lower level")
        print("  -> retrain T_phi on posterior-weighted samples")
        print("  " + "-" * 60)

        posterior_counter = defaultdict(Counter)
        for doc in documents:
            for h, t, r in doc["triples"]:
                h_type = doc["entities"][h]["type"]
                t_type = doc["entities"][t]["type"]
                ctx = (r, h_type, t_type)
                if ctx not in sampled:
                    continue
                key = "%d_%d_%d" % (ctx[0], ctx[1], ctx[2])
                rules_list = sampled[ctx]
                H_values = []
                for (rb, log_prior), wi in zip(rules_list, scorer.weights[key].data):
                    gs = ground_rule_on_graph(doc, rb, h, t)
                    H = gs * wi.item() * 0.5 + log_prior
                    H_values.append(H)
                if H_values:
                    probs = F.softmax(torch.tensor(H_values), dim=0)
                    chosen = Categorical(probs).sample().item()
                    posterior_counter[ctx][rules_list[chosen][0]] += 1

        print("")
        print("    Line 16: Posterior samples (rules reinforced by grounding):")
        for ctx, counter in posterior_counter.items():
            rn = RELATIONS[ctx[0]]
            hn = NER_TYPES[ctx[1]]
            tn = NER_TYPES[ctx[2]]
            total = sum(counter.values())
            print("      <%s, %s> + %s:" % (hn, tn, rn))
            for rb, count in counter.most_common():
                print("        %s : %d/%d (%.0f%%)" % (
                    body_str(rb), count, total, 100 * count / total))

        # Line 17: Update phi
        new_train_data = []
        for ctx, counter in posterior_counter.items():
            total = sum(counter.values())
            for rb, count in counter.items():
                padded = list(rb) + [STOP] * (max_depth - len(rb))
                chain = [ctx[0]] + padded
                n_copies = max(1, int(count / total * 10))
                for _ in range(n_copies):
                    new_train_data.append((chain, ctx[1], ctx[2]))

        combined = train_data + new_train_data
        for epoch in range(10):
            total_loss = 0
            for chain, ht, tt in combined:
                loss = generator.compute_loss(
                    torch.LongTensor([chain]),
                    torch.LongTensor([ht]),
                    torch.LongTensor([tt]))
                opt_gen.zero_grad(); loss.backward(); opt_gen.step()
                total_loss += loss.item()
        print("")
        print("    Line 17: T_phi updated. Avg loss = %.4f" % (
            total_loss / max(1, len(combined))))

    # ============================================================
    # Alg.1 Line 19: Return optimized phi, theta
    # ============================================================
    print("")
    print("=" * 70)
    print("Alg.1 Line 19: Return optimized phi, theta")
    print("=" * 70)

    # Final token-by-token demo to show improvement
    print("")
    print("  Token-by-token generation AFTER bilevel optimization:")
    for ctx in context_triples:
        rel, ht, tt = ctx
        print("")
        print("  ---- Context: <%s, %s>,  r_head = %s ----" % (
            NER_TYPES[ht], NER_TYPES[tt], RELATIONS[rel]))
        print("")
        generator.sample_rules_verbose(rel, ht, tt, N=1)

    # Final prediction
    print("")
    print("=" * 70)
    print("FINAL: Link Prediction using learned rules")
    print("=" * 70)

    final_rules = generator.sample_rules(context_triples, N=30)
    final_scorer = RuleScorer(final_rules)
    opt_f = torch.optim.Adam(final_scorer.parameters(), lr=0.05)
    for _ in range(50):
        for doc in documents:
            for qh, qt, qr in doc["queries"]:
                logit = final_scorer.score_query(doc, qh, qt, qr)
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor([1.0]))
                opt_f.zero_grad(); loss.backward(); opt_f.step()
            for h, t, r in doc["triples"]:
                h_type = doc["entities"][h]["type"]
                t_type = doc["entities"][t]["type"]
                ctx = (r, h_type, t_type)
                if ctx not in final_rules:
                    continue
                logit = final_scorer.score_query(doc, h, t, r)
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor([1.0]))
                opt_f.zero_grad(); loss.backward(); opt_f.step()

    for doc in documents:
        print("")
        print("  %s:" % doc["title"])
        for qh, qt, qr in doc["queries"]:
            logit = final_scorer.score_query(doc, qh, qt, qr)
            prob = torch.sigmoid(logit).item()
            hn = doc["entities"][qh]["name"]
            tn = doc["entities"][qt]["name"]
            ctx = (qr, doc["entities"][qh]["type"], doc["entities"][qt]["type"])
            print("    %s --%s--> %s  score=%.4f" % (hn, RELATIONS[qr], tn, prob))
            if ctx in final_rules:
                for rb, lp in sorted(final_rules[ctx], key=lambda x: -x[1])[:3]:
                    gs = ground_rule_on_graph(doc, rb, qh, qt)
                    key = "%d_%d_%d" % (ctx[0], ctx[1], ctx[2])
                    status = "GROUNDED" if gs > 0 else "no path"
                    print("      rule: %s <- %s  [%s]" % (
                        RELATIONS[qr], body_str(rb), status))


if __name__ == "__main__":
    run_algorithm1(DOCUMENTS, n_iters=2, n_samples=20)
