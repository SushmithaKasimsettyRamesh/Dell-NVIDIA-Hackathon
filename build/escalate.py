"""The interrupt. When the panel splits, message a human and wait for their ruling.

    python3 build/escalate.py --send 3          push the 3 most interesting escalations
    python3 build/escalate.py --wait 180        block for a human decision, 180s

This is what makes it an agent rather than a script (STACK_INTEGRATION.md): a Python
loop that runs and exits does not satisfy "always-on business agent". An agent that
stops, asks a person, and blocks on the answer does.

The static HTML queue is the fallback and it cannot fail. This is the better demo.
Credentials are read from ~/.gb10_telegram_{token,chatid} and never logged.

Requires api.telegram.org in the OpenShell allow-list -- it is the ONE outbound
destination the policy permits, which is also why the deny log is worth showing:
everything else the agent reached for was refused.
"""
import json, os, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
API = "https://api.telegram.org/bot%s/%s"


def creds():
    t = open(os.path.expanduser("~/.gb10_telegram_token")).read().strip()
    c = open(os.path.expanduser("~/.gb10_telegram_chatid")).read().strip()
    return t, c


def call(method, payload, timeout=25):
    tok, _ = creds()
    req = urllib.request.Request(API % (tok, method),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def money(x):
    return "$%s" % format(int(round(x)), ",d")


def card(wid, a, f, q, lit, kind):
    why = ("The first reviewer cleared this and the second flagged it"
           if kind == "audit" else "The two reviewers disagree")
    return (
        "\U0001F6A9 *Escalation — %s*\n"
        "%s, opened %d days ago\n\n"
        "*%d cash deposits* totalling %s over 14 days\n"
        "%s to %s, %d branch(es), %d state(s)\n"
        "%.2f× this account's normal volume\n"
        "%d%% of inbound funds remained\n\n"
        "*Intent* (Qwen): %s\n_%s_\n\n"
        "*Literal* (Nemotron): %s\n_%s_\n\n"
        "%s. Your call."
        % (wid, a["sector"], a["opened_days_ago"],
           f["cash_deposit_count"], money(f["cash_deposit_total"]),
           money(f["min_deposit_amount"]), money(f["max_deposit_amount"]),
           f["distinct_branches"], f["distinct_states"],
           f["vs_own_history_ratio"], int(round(100 * f["end_balance_ratio"])),
           q["verdict"].upper(), _one(q.get("rationale", "")),
           lit["verdict"].upper(), _one(lit.get("rationale", "")), why))


def _one(s):
    s = (s or "").strip()
    i = s.find(". ")
    return (s[:i + 1] if 40 < i < 220 else s[:200].rstrip() + "…")


KEYBOARD = {"inline_keyboard": [[
    {"text": "\U0001F534 File report", "callback_data": "file"},
    {"text": "✅ Close", "callback_data": "close"},
    {"text": "❓ More info", "callback_data": "more"}]]}


def send(text):
    _, chat = creds()
    return call("sendMessage", {"chat_id": chat, "text": text, "parse_mode": "Markdown",
                                "reply_markup": KEYBOARD})


def drain():
    """Consume anything already queued and return the next offset.

    getUpdates with no offset returns EVERY pending update, including buttons tapped
    on earlier messages. Without this, wait() reports a stale tap as this case's
    decision -- instantly, before anyone touches the phone. It looks like it worked.
    """
    tok, _ = creds()
    try:
        d = json.loads(urllib.request.urlopen(
            API % (tok, "getUpdates") + "?timeout=1", timeout=15).read())
        r = d.get("result", [])
        return (r[-1]["update_id"] + 1) if r else None
    except Exception:
        return None


def wait(seconds=180, poll=3):
    """Block for a human ruling. This is the point -- the agent stops and asks."""
    tok, _ = creds()
    off, t0 = drain(), time.time()
    print("waiting up to %ds for a decision..." % seconds)
    while time.time() - t0 < seconds:
        url = API % (tok, "getUpdates") + "?timeout=10"
        if off:
            url += "&offset=%d" % off
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=20).read())
        except Exception:
            time.sleep(poll)
            continue
        for u in d.get("result", []):
            off = u["update_id"] + 1
            cq = u.get("callback_query")
            if cq:
                choice = cq["data"]
                who = cq["from"].get("first_name", "reviewer")
                call("answerCallbackQuery", {"callback_query_id": cq["id"],
                                             "text": "Recorded: %s" % choice})
                print("\nDECISION: %s  (by %s, after %.0fs)"
                      % (choice.upper(), who, time.time() - t0))
                rec = {"decision": choice, "by": who, "seconds": round(time.time() - t0, 1)}
                with open(os.path.join(OUT, "decisions.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                return rec
        time.sleep(poll)
    print("no decision within %ds -- the case stays in the queue" % seconds)
    return None


def main():
    import corpus as C, facts as F
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    n = int(sys.argv[sys.argv.index("--send") + 1]) if "--send" in sys.argv else 0
    if n:
        rows = [json.loads(l) for l in open(os.path.join(OUT, "panel.jsonl"))]
        I = dict((r["window_id"], r) for r in rows if r["stage"] == "intent" and r["verdict"])
        L = dict((r["window_id"], r) for r in rows if r["stage"] == "literal" and r["verdict"])
        esc = [(w, I[w], L[w], "split" if I[w]["verdict"] == "suspicious" else "audit")
               for w in L if w in I and I[w]["verdict"] != L[w]["verdict"]]
        esc.sort(key=lambda e: e[3] != "audit")        # audit-caught first, most interesting
        c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01).build()
        wins = dict((w["window_id"], w) for w in c.windows())
        fx = F.all_facts(c.accounts, c.txns, list(wins.values()))
        amap = dict((a["account_id"], a) for a in c.accounts)
        for wid, q, lit, kind in esc[:n]:
            a = amap[wins[wid]["account_id"]]
            r = send(card(wid, a, fx[wid], q, lit, kind))
            print("sent %s (%s) -> message %s" % (wid, kind, r["result"]["message_id"]))
    if "--wait" in sys.argv:
        wait(int(sys.argv[sys.argv.index("--wait") + 1]))


if __name__ == "__main__":
    main()
