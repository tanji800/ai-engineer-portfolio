"""STEP 6b - The core agent loop (ReAct style) with tool calling, from scratch.
An agent = while not done: model picks a tool or finishes -> run tool -> append observation -> repeat.
Offline: the "model" is a scripted policy so you can see the loop. Online: replace `policy` with an LLM
tool-calling API and keep everything else (tools registry, step limit, error handling).
Run:  python 06_agents/02_tool_calling_agent.py
"""
import ast, operator as op

def calculator(expr: str) -> str:
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("unsupported")
    return str(ev(ast.parse(expr, mode="eval").body))      # NEVER eval() untrusted input directly

def lookup(key: str) -> str:
    return {"vectors": "1000000", "dim": "768", "bytes_per_float": "4"}.get(key, "unknown")

TOOLS = {"calculator": calculator, "lookup": lookup}

def policy(history):
    """Scripted stand-in for an LLM. Returns ('tool', name, arg) or ('final', text)."""
    obs = [h for h in history if h[0] == "obs"]
    if len(obs) == 0: return ("tool", "lookup", "vectors")
    if len(obs) == 1: return ("tool", "lookup", "dim")
    if len(obs) == 2: return ("tool", "lookup", "bytes_per_float")
    if len(obs) == 3: return ("tool", "calculator", f"{obs[0][2]}*{obs[1][2]}*{obs[2][2]}/1000000000")
    return ("final", f"Raw float32 storage needs ~{obs[3][2]} GB (before index overhead).")

def run(question, max_steps=8):
    history = [("user", question)]
    for step in range(max_steps):                          # ALWAYS cap steps (runaway loops cost money)
        kind, *rest = policy(history)
        if kind == "final":
            return rest[0], history
        name, arg = rest
        try:
            result = TOOLS[name](arg)
        except Exception as e:                             # feed errors back so the model can recover
            result = f"ERROR: {e}"
        history.append(("obs", name, result)); print(f"step {step}: {name}({arg!r}) -> {result}")
    return "stopped: step limit", history

if __name__ == "__main__":
    print(run("How much memory do 1M 768-d float32 vectors need?")[0])
