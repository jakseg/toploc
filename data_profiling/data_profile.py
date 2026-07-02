"""
Dataset profiling for TREC CAsT 2019.

Analyses topics, qrels, and the full passage collection. Prints summary stats
and writes seaborn plots to ./plots/.

Usage:
    python data_profile.py
"""

import argparse
import os
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT_TOPICS = "/home/toploc2/Datasets/conversational/CAST2019/topics/topics.tsv"
DEFAULT_QRELS = "/home/toploc2/Datasets/conversational/CAST2019/topics/qrels.qrel"
DEFAULT_COLLECTION = "/home/toploc2/Datasets/conversational/CAST2019/CAST2019collection.tsv"
DEFAULT_OUTDIR = "plots"


# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #
def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title):
    print(f"\n--- {title} ---")


def fmt_stats(values, name="values"):
    if not values:
        return f"  no {name}"
    return (
        f"  count: {len(values):,}\n"
        f"  min:   {min(values):,}\n"
        f"  max:   {max(values):,}\n"
        f"  mean:  {statistics.mean(values):.2f}\n"
        f"  median:{statistics.median(values):.2f}\n"
        f"  stdev: {statistics.pstdev(values):.2f}"
    )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_topics(path):
    """topics.tsv has format: turn_id,query (comma-separated despite .tsv)."""
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                topics[parts[0].strip()] = parts[1].strip()
    return topics


def load_qrels(path):
    """qrels.qrel has format: qid,iter,pid,score (comma-separated)."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            qid, _, pid, score = parts
            try:
                qrels[qid][pid] = int(score)
            except ValueError:
                continue
    return qrels


def stream_collection(path):
    """Yield (pid, text) from CAST2019collection.tsv (tab-separated)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                yield parts[0], parts[1]


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def analyse_topics(topics):
    section("TOPICS")
    print(f"Total turns: {len(topics):,}")

    conv_turns = defaultdict(list)
    for turn_key in topics:
        if "_" in turn_key:
            conv, _turn = turn_key.rsplit("_", 1)
            conv_turns[conv].append(turn_key)

    turns_per_conv = [len(v) for v in conv_turns.values()]
    print(f"Total conversations: {len(conv_turns)}")

    subsection("Turns per conversation")
    print(fmt_stats(turns_per_conv, "convs"))

    query_lens_chars = [len(q) for q in topics.values()]
    query_lens_words = [len(q.split()) for q in topics.values()]
    subsection("Query length (chars)")
    print(fmt_stats(query_lens_chars, "queries"))
    subsection("Query length (words)")
    print(fmt_stats(query_lens_words, "queries"))

    first_words = Counter(q.split()[0].lower() for q in topics.values() if q.split())
    subsection("Top 10 opening words")
    for w, c in first_words.most_common(10):
        print(f"  {w:<12} {c}")

    return {
        "conv_turns": conv_turns,
        "turns_per_conv": turns_per_conv,
        "query_lens_chars": query_lens_chars,
        "query_lens_words": query_lens_words,
    }


def analyse_qrels(qrels, topics):
    section("QRELS")
    n_turns_judged = len(qrels)
    total_judgments = sum(len(v) for v in qrels.values())
    print(f"Turns with judgments: {n_turns_judged:,}")
    print(f"Total (qid, pid) judgments: {total_judgments:,}")

    score_counter = Counter()
    judgments_per_turn = []
    relevant_per_turn = []
    for qid, pid_scores in qrels.items():
        judgments_per_turn.append(len(pid_scores))
        relevant_per_turn.append(sum(1 for s in pid_scores.values() if s > 0))
        for s in pid_scores.values():
            score_counter[s] += 1

    subsection("Relevance score distribution")
    for score in sorted(score_counter):
        pct = 100.0 * score_counter[score] / total_judgments
        print(f"  score {score}: {score_counter[score]:>7,}  ({pct:5.2f}%)")

    subsection("Judgments per turn (all scores)")
    print(fmt_stats(judgments_per_turn, "turns"))
    subsection("Relevant (score>0) passages per turn")
    print(fmt_stats(relevant_per_turn, "turns"))

    topic_keys = set(topics.keys()) if topics else set()
    qrel_keys = set(qrels.keys())
    subsection("Topic / qrel coverage")
    print(f"  turns in both:        {len(topic_keys & qrel_keys):,}")
    print(f"  turns only in topics: {len(topic_keys - qrel_keys):,}")
    print(f"  turns only in qrels:  {len(qrel_keys - topic_keys):,}")

    return {
        "score_counter": score_counter,
        "judgments_per_turn": judgments_per_turn,
        "relevant_per_turn": relevant_per_turn,
    }


def analyse_collection(path):
    section("COLLECTION (CAST2019collection.tsv)")
    if not os.path.exists(path):
        print(f"  not found at {path} — skipping")
        return None

    size_bytes = os.path.getsize(path)
    print(f"File size: {size_bytes / 1e9:.2f} GB")
    print("Reading full collection (this may take a few minutes)...")

    lens_chars = []
    lens_words = []
    pid_prefix_counter = Counter()
    sample_texts = []
    n = 0
    for pid, text in stream_collection(path):
        lens_chars.append(len(text))
        lens_words.append(len(text.split()))
        prefix = pid.split("_", 1)[0] if "_" in pid else "<no-underscore>"
        pid_prefix_counter[prefix] += 1
        if n < 3:
            sample_texts.append((pid, text[:160]))
        n += 1
        if n % 5_000_000 == 0:
            print(f"  ... {n:,} passages processed")

    print(f"Total passages: {n:,}")

    subsection("Sample passages (first 3)")
    for pid, snippet in sample_texts:
        suffix = "..." if len(snippet) == 160 else ""
        print(f"  [{pid}] {snippet}{suffix}")

    subsection("Passage length (chars)")
    print(fmt_stats(lens_chars, "passages"))
    subsection("Passage length (words)")
    print(fmt_stats(lens_words, "passages"))

    subsection("PID prefix distribution")
    for prefix, c in pid_prefix_counter.most_common(10):
        pct = 100.0 * c / n
        print(f"  {prefix:<20} {c:>10,}  ({pct:5.2f}%)")

    return {
        "lens_chars": lens_chars,
        "lens_words": lens_words,
        "pid_prefix_counter": pid_prefix_counter,
        "total": n,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(topics_data, qrels_data, collection_data, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("crest")

    os.makedirs(outdir, exist_ok=True)
    section(f"WRITING PLOTS TO {outdir}/")

    def save(fig, name):
        p = os.path.join(outdir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"  {p}")

    # 1. Turns per conversation
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(topics_data["turns_per_conv"],
                 bins=range(1, max(topics_data["turns_per_conv"]) + 2),
                 ax=ax, color=palette[2], edgecolor="white")
    ax.set_xlabel("Turns per conversation")
    ax.set_ylabel("Number of conversations")
    ax.set_title("Distribution of conversation lengths — TREC CAsT 2019")
    save(fig, "01_turns_per_conversation.png")

    # 2. Query length in words
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(topics_data["query_lens_words"], bins=30,
                 ax=ax, color=palette[3], edgecolor="white", kde=True)
    ax.set_xlabel("Words per query")
    ax.set_ylabel("Number of queries")
    ax.set_title("Query length distribution")
    save(fig, "02_query_length_words.png")

    # 3. Relevance score distribution
    if qrels_data["score_counter"]:
        scores = sorted(qrels_data["score_counter"])
        counts = [qrels_data["score_counter"][s] for s in scores]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=[str(s) for s in scores], y=counts,
                    ax=ax, palette="crest", edgecolor="white")
        ax.set_xlabel("Relevance score")
        ax.set_ylabel("Number of judgments")
        ax.set_title("Qrels: relevance score distribution")
        for i, c in enumerate(counts):
            ax.text(i, c, f"{c:,}", ha="center", va="bottom")
        save(fig, "03_relevance_scores.png")

    # 4. Judgments per turn
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(qrels_data["judgments_per_turn"], bins=30,
                 ax=ax, color=palette[4], edgecolor="white", kde=True)
    ax.set_xlabel("Judgments per turn")
    ax.set_ylabel("Number of turns")
    ax.set_title("Distribution of qrel judgments per turn")
    save(fig, "04_judgments_per_turn.png")

    # 5. Relevant (score>0) passages per turn
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(qrels_data["relevant_per_turn"], bins=30,
                 ax=ax, color=palette[1], edgecolor="white", kde=True)
    ax.set_xlabel("Relevant passages per turn (score > 0)")
    ax.set_ylabel("Number of turns")
    ax.set_title("Relevant passages per turn")
    save(fig, "05_relevant_per_turn.png")

    # 6. Passage length (words) — collection
    if collection_data:
        fig, ax = plt.subplots(figsize=(9, 5))
        # Clip to 99th percentile for readability
        import numpy as np
        words = np.asarray(collection_data["lens_words"])
        upper = float(np.quantile(words, 0.99))
        sns.histplot(words[words <= upper], bins=80,
                     ax=ax, color=palette[5], edgecolor="white")
        ax.set_xlabel("Words per passage (99th-percentile clip)")
        ax.set_ylabel("Number of passages")
        ax.set_title(f"Passage length distribution — {collection_data['total']:,} passages")
        save(fig, "06_passage_length_words.png")

        # 7. PID prefix breakdown
        prefixes = collection_data["pid_prefix_counter"].most_common(10)
        if prefixes:
            labels = [p for p, _ in prefixes]
            counts = [c for _, c in prefixes]
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(x=labels, y=counts, ax=ax,
                        palette="crest", edgecolor="white")
            ax.set_xlabel("PID prefix")
            ax.set_ylabel("Number of passages")
            ax.set_title("Collection split by PID prefix")
            for i, c in enumerate(counts):
                ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=10)
            save(fig, "07_pid_prefix.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default=DEFAULT_TOPICS)
    ap.add_argument("--qrels", default=DEFAULT_QRELS)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if not os.path.exists(args.topics):
        print(f"ERROR: topics not found at {args.topics}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.qrels):
        print(f"ERROR: qrels not found at {args.qrels}", file=sys.stderr)
        sys.exit(1)

    section("DATA PROFILE — TREC CAsT 2019")
    print(f"topics:     {args.topics}")
    print(f"qrels:      {args.qrels}")
    print(f"collection: {args.collection}")
    print(f"outdir:     {args.outdir}")

    topics = load_topics(args.topics)
    qrels = load_qrels(args.qrels)

    topics_data = analyse_topics(topics)
    qrels_data = analyse_qrels(qrels, topics)
    collection_data = analyse_collection(args.collection)

    make_plots(topics_data, qrels_data, collection_data, args.outdir)

    section("DONE")


if __name__ == "__main__":
    main()
