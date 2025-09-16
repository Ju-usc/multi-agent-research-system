"""Evaluation-related tests (kept separate from workflow tests)."""

import pytest
from unittest.mock import patch
import sys
import types
import dspy


def _stub_tools_module() -> None:
    class _WebSearchTool:
        def __init__(self, api_key: str):
            pass
        async def __call__(self, *a, **k):
            return ""
    class _FileSystemTool:
        def __init__(self, fs):
            self.fs = fs
        def tree(self, max_depth: int = 3) -> str:
            return self.fs.tree(max_depth)
        def read(self, path: str) -> str:
            return ""
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