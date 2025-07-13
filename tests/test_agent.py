import pytest
import dspy
import warnings
import logging
from agent import LeadAgent  # Assume importable

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

@pytest.mark.asyncio
async def test_lead_agent_basic_run():
    agent = LeadAgent()  # Init with defaults/mocks if needed
    query = "Test simple research query"
    result = await agent.run(query)
    print(result)
    assert isinstance(result, str), "Result should be a string report"
    assert len(agent.steps_trace) > 0, "Steps trace should have entries"
    assert any('Planned' in step['action'] for step in agent.steps_trace), "Should have a planning step"
