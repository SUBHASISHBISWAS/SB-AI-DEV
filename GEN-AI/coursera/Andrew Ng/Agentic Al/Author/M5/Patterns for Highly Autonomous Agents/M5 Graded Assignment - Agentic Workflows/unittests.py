# tests_agents.py
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType

# ===== Neutral fixtures =====
_TOPIC = "The ensemble Kalman filter for time series forecasting"
_TASK  = "Draft a concise summary (150-250 words) explaining the core idea and typical applications."
_TASK_EDIT = "Reflect on the draft and suggest improvements in structure, clarity, and citations."

# =========================
# Test 1: planner_agent
# =========================
def test_planner_agent(learner_func):
    def g():
        function_name = "planner_agent"
        cases = []

        # 1) type check
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # 2) call and error handling
        try:
            out = learner_func(_TOPIC)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__}: {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        # 3) must return list
        t = test_case()
        if not isinstance(out, list):
            t.failed = True
            t.msg = f"{function_name} must return a list[str]"
            t.want = list
            t.got = type(out)
            return [t]
        cases.append(t)

        # 4) elements must be strings and reasonably many steps
        t = test_case()
        elem_types_ok = all(isinstance(s, str) for s in out)
        if not elem_types_ok or len(out) < 3:
            t.failed = True
            t.msg = "plan should be a list of >=3 string steps"
            t.want = "list[str] with length >= 3"
            t.got = {"length": len(out), "bad_types": [type(s) for s in out if not isinstance(s, str)]}
        cases.append(t)

        # 5) last step should mention markdown report
        t = test_case()
        last = (out[-1] if out else "").lower()
        if not any(k in last for k in ["markdown", "md"]):
            t.failed = True
            t.msg = "final step should mention generating a Markdown document"
            t.want = "mention of 'Markdown' or 'md'"
            t.got = last
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


# =========================
# Test 2: research_agent
# (function signature: research_agent(task: str, ..., return_messages: bool=False))
# =========================
def test_research_agent(learner_func):
    def g():
        function_name = "research_agent"
        cases = []

        # 1) type check
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # 2) call (default return_messages=False)
        try:
            out_text = learner_func("Find 3 key references and summarize them briefly.")
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__} (return_messages=False): {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        # 3) must return str (default mode)
        t = test_case()
        if not isinstance(out_text, str):
            t.failed = True
            t.msg = f"{function_name} must return a str when return_messages=False"
            t.want = str
            t.got = type(out_text)
            return [t]
        cases.append(t)

        # 4) non-trivial length
        t = test_case()
        if len(out_text.strip()) <= 50:
            t.failed = True
            t.msg = "output should be non-trivial (length > 50) for research summary"
            t.want = "> 50 chars"
            t.got = len(out_text.strip())
        cases.append(t)

        # 5) call with return_messages=True (tuple expected: (str, messages))
        try:
            out = learner_func("Briefly summarize two seminal papers in one paragraph.", return_messages=True)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__} (return_messages=True): {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        t = test_case()
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], str) and isinstance(out[1], list)):
            t.failed = True
            t.msg = f"{function_name} must return (str, messages_list) when return_messages=True"
            t.want = "(str, list)"
            t.got = type(out)
            return [t]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


# =========================
# Test 3: writer_agent
# =========================
def test_writer_agent(learner_func):
    def g():
        function_name = "writer_agent"
        cases = []

        # 1) type check
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # 2) call and error handling
        try:
            out = learner_func(_TASK)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__}: {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        # 3) must return str
        t = test_case()
        if not isinstance(out, str):
            t.failed = True
            t.msg = f"{function_name} must return a str"
            t.want = str
            t.got = type(out)
            return [t]
        cases.append(t)

        # 4) non-trivial length
        t = test_case()
        if len(out.strip()) <= 50:
            t.failed = True
            t.msg = "draft should be non-trivial (length > 50)"
            t.want = "> 50 chars"
            t.got = len(out.strip())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


# =========================
# Test 4: editor_agent
# =========================
def test_editor_agent(learner_func):
    def g():
        function_name = "editor_agent"
        cases = []

        # 1) type check
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # 2) call and error handling
        try:
            out = learner_func(_TASK_EDIT)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{function_name} raised {type(e).__name__}: {e}"
            t.want = "no exception"
            t.got = str(e)
            return [t]

        # 3) must return str
        t = test_case()
        if not isinstance(out, str):
            t.failed = True
            t.msg = f"{function_name} must return a str"
            t.want = str
            t.got = type(out)
            return [t]
        cases.append(t)

        # 4) non-trivial length
        t = test_case()
        if len(out.strip()) <= 50:
            t.failed = True
            t.msg = "editor output should be non-trivial (length > 50)"
            t.want = "> 50 chars"
            t.got = len(out.strip())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
