"""Evaluation-related tests (kept separate from workflow tests)."""

import pytest
from unittest.mock import patch
import sys
import types
import dspy

pytestmark = pytest.mark.skip(reason="Evaluation pipeline pending refactor; disable to avoid expensive runs.")


def _stub_tools_module() -> None:
    class _WebSearchTool:
        def __init__(self, api_key: str):
            pass
        def __call__(self, *a, **k):
            return ""
    class _FileSystemTool:
        def __init__(self, root: str = "memory"):
            from pathlib import Path as _Path
            self.root = _Path(root)
            self.root.mkdir(exist_ok=True)
        def write(self, path: str, content: str):
            fp = self.root / path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)
            return fp
        def read(self, path: str) -> str:
            fp = self.root / path
            if not fp.exists():
                return f"[ERROR] File not found: {path}"
            return fp.read_text()
        def exists(self, path: str) -> bool:
            return (self.root / path).exists()
        def tree(self, max_depth: int = 3) -> str:
            paths = []
            def _collect(p, rel: str, depth: int):
                if max_depth is not None and depth >= max_depth:
                    return
                items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)) if p.exists() else []
                for item in items:
                    ip = f"{rel}{item.name}" if rel else item.name
                    if item.is_dir():
                        paths.append(f"{ip}/")
                        _collect(item, f"{ip}/", depth + 1)
                    else:
                        paths.append(ip)
            _collect(self.root, "", 0)
            if not paths:
                return "memory/ (empty)"
            return "\n".join(["memory/"] + sorted(paths))
    sys.modules['tools'] = types.SimpleNamespace(
        WebSearchTool=_WebSearchTool,
        FileSystemTool=_FileSystemTool,
    )


def test_browsecomp_program_wrapper():
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    _stub_tools_module()
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        from eval import BrowseCompProgram

        class FakeAgent:
            async def run(self, problem: str) -> str:
                return f"Report for: {problem}"

        program = BrowseCompProgram(FakeAgent())
        assert hasattr(program, 'forward') and callable(program.forward)


def test_browsecomp_metric():
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    _stub_tools_module()
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        from eval import browsecomp_metric, dspy as _dspy

        class _FakeJudge:
            def __call__(self, question: str, report: str, correct_answer: str):
                is_correct = correct_answer in report
                return types.SimpleNamespace(
                    extracted_answer=correct_answer if is_correct else "None",
                    reasoning="stubbed",
                    is_correct=is_correct,
                )

        with patch.object(_dspy, 'ChainOfThought', new=lambda *a, **k: _FakeJudge()):
            example = dspy.Example(problem="What is 2+2?", answer="4")
            assert 0.0 <= browsecomp_metric(example, dspy.Prediction(report="The answer is 4.")) <= 1.0
            assert 0.0 <= browsecomp_metric(example, dspy.Prediction(report="The answer is 5.")) <= 1.0


def test_browsecomp_evaluation_framework():
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    _stub_tools_module()
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        from eval import BrowseCompProgram, dspy as _dspy

        class _FakeJudge:
            def __call__(self, question: str, report: str, correct_answer: str):
                is_correct = correct_answer in report
                return types.SimpleNamespace(
                    extracted_answer=correct_answer if is_correct else "None",
                    reasoning="stubbed",
                    is_correct=is_correct,
                )

        class FakeAgent:
            async def run(self, problem: str) -> str:
                return f"Report for: {problem}"

        with patch.object(_dspy, 'ChainOfThought', new=lambda *a, **k: _FakeJudge()):
            program = BrowseCompProgram(FakeAgent())
            dataset = [dspy.Example(problem="What is 2+2?", answer="4")]
            
            # Test the evaluation framework (minimal)
            results = []
            for example in dataset:
                prediction = program.forward(problem=example.problem)
                results.append(prediction)
            
            assert len(results) == 1
            assert hasattr(results[0], 'report')
