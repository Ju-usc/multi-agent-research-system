"""
Multi-Agent Research System - Evaluation Pipeline
DSPy-based evaluation with LLM judges for lead agent optimization
"""

import os
import dspy
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from agent import QueryAnalysis, ResearchPlan, PlanStep

# Load environment for DSPy configuration
from dotenv import load_dotenv
load_dotenv()

# Configure DSPy with OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EVAL_MODEL = os.getenv("EVAL_MODEL", "anthropic/claude-3.5-sonnet")

# Configure evaluation LM
eval_lm = dspy.LM(
    model=EVAL_MODEL,
    api_key=OPENROUTER_API_KEY,
    api_base=OPENROUTER_BASE_URL
)
dspy.configure(lm=eval_lm)

# ---------- Judge Signatures ----------

class DelegationQualityJudge(dspy.Signature):
    """Judge if task decomposition creates clear, actionable subagent instructions.
    
    Evaluates whether the research plan provides specific objectives, clear boundaries,
    and sufficient detail for subagents to execute tasks effectively.
    """
    query: str = dspy.InputField(desc="Original user research query")
    analysis: str = dspy.InputField(desc="Query analysis as JSON string")
    plan: str = dspy.InputField(desc="Research plan as JSON string")
    delegation_score: float = dspy.OutputField(desc="Score 0.0-1.0 for instruction clarity and specificity")
    reasoning: str = dspy.OutputField(desc="Brief explanation of score")


class ResourceEfficiencyJudge(dspy.Signature):
    """Judge appropriate effort scaling based on query complexity.
    
    Evaluates if the number of planned subagents and tool budgets match
    the complexity level (simple=1 agent, medium=3-5, complex=5-10).
    """
    query: str = dspy.InputField(desc="Original user research query")
    complexity: str = dspy.InputField(desc="Assessed complexity level (simple/medium/complex)")
    plan: str = dspy.InputField(desc="Research plan as JSON string")
    efficiency_score: float = dspy.OutputField(desc="Score 0.0-1.0 for appropriate resource allocation")
    reasoning: str = dspy.OutputField(desc="Brief explanation of score")


class QueryClassificationJudge(dspy.Signature):
    """Judge accuracy of query type classification.
    
    Evaluates if the query was correctly classified as depth-first, breadth-first,
    or straightforward based on the nature of the research needed.
    """
    query: str = dspy.InputField(desc="Original user research query")
    predicted_type: str = dspy.InputField(desc="Predicted query type")
    analysis: str = dspy.InputField(desc="Query analysis reasoning")
    classification_score: float = dspy.OutputField(desc="Score 0.0-1.0 for classification accuracy")
    reasoning: str = dspy.OutputField(desc="Brief explanation of score")


class PlanCoherenceJudge(dspy.Signature):
    """Judge if the research plan would actually answer the user's query.
    
    Evaluates overall coherence between user intent, analysis, and proposed
    research approach for achieving the desired outcome.
    """
    query: str = dspy.InputField(desc="Original user research query")
    analysis: str = dspy.InputField(desc="Query analysis as JSON string")
    plan: str = dspy.InputField(desc="Research plan as JSON string")
    coherence_score: float = dspy.OutputField(desc="Score 0.0-1.0 for plan-query alignment")
    reasoning: str = dspy.OutputField(desc="Brief explanation of score")


# ---------- Evaluation Data Models ----------

class EvaluationResult(BaseModel):
    """Results from evaluating a single agent run"""
    delegation_score: float
    efficiency_score: float
    classification_score: float
    coherence_score: float
    overall_score: float
    reasoning: Dict[str, str]


class EvaluationExample(BaseModel):
    """Single evaluation example with query and agent outputs"""
    query: str
    analysis: QueryAnalysis
    plan: ResearchPlan
    expected_score: Optional[float] = None  # For training data


# ---------- Judge Modules ----------

class LeadAgentEvaluator(dspy.Module):
    """Complete evaluation module using all judges"""
    
    def __init__(self):
        super().__init__()
        self.delegation_judge = dspy.ChainOfThought(DelegationQualityJudge)
        self.efficiency_judge = dspy.ChainOfThought(ResourceEfficiencyJudge)
        self.classification_judge = dspy.ChainOfThought(QueryClassificationJudge)
        self.coherence_judge = dspy.ChainOfThought(PlanCoherenceJudge)
    
    def forward(self, example: EvaluationExample) -> EvaluationResult:
        """Evaluate a single agent output using all judges"""
        
        # Convert analysis and plan to JSON strings for judges
        analysis_str = example.analysis.model_dump_json()
        plan_str = example.plan.model_dump_json()
        
        # Run all judges
        delegation_result = self.delegation_judge(
            query=example.query,
            analysis=analysis_str,
            plan=plan_str
        )
        
        efficiency_result = self.efficiency_judge(
            query=example.query,
            complexity=example.analysis.complexity,
            plan=plan_str
        )
        
        classification_result = self.classification_judge(
            query=example.query,
            predicted_type=example.analysis.query_type,
            analysis=analysis_str
        )
        
        coherence_result = self.coherence_judge(
            query=example.query,
            analysis=analysis_str,
            plan=plan_str
        )
        
        # Calculate overall score (simple average)
        overall_score = (
            delegation_result.delegation_score +
            efficiency_result.efficiency_score +
            classification_result.classification_score +
            coherence_result.coherence_score
        ) / 4.0
        
        return EvaluationResult(
            delegation_score=delegation_result.delegation_score,
            efficiency_score=efficiency_result.efficiency_score,
            classification_score=classification_result.classification_score,
            coherence_score=coherence_result.coherence_score,
            overall_score=overall_score,
            reasoning={
                "delegation": delegation_result.reasoning,
                "efficiency": efficiency_result.reasoning,
                "classification": classification_result.reasoning,
                "coherence": coherence_result.reasoning
            }
        )


# ---------- Evaluation Functions ----------

def evaluate_agent_output(query: str, analysis: QueryAnalysis, plan: ResearchPlan) -> EvaluationResult:
    """Simple function to evaluate a single agent output"""
    evaluator = LeadAgentEvaluator()
    example = EvaluationExample(query=query, analysis=analysis, plan=plan)
    return evaluator.forward(example)


def composite_metric(example: EvaluationExample, prediction) -> float:
    """Composite metric for DSPy optimization"""
    try:
        # Create evaluation example from prediction
        eval_example = EvaluationExample(
            query=example.query,
            analysis=prediction.analysis if hasattr(prediction, 'analysis') else example.analysis,
            plan=prediction.plan if hasattr(prediction, 'plan') else example.plan
        )
        
        evaluator = LeadAgentEvaluator()
        result = evaluator.forward(eval_example)
        return result.overall_score
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0


# ---------- MIPROv2 Optimization Setup ----------

def create_optimizer(trainset: List[EvaluationExample], metric=composite_metric):
    """Create MIPROv2 optimizer for lead agent improvement"""
    from dspy.teleprompt import MIPROv2
    
    teleprompter = MIPROv2(
        metric=metric,
        auto="light",  # Start with light optimization
        num_trials=20,  # Keep trials low for testing
        init_temperature=1.0
    )
    
    return teleprompter


def optimize_agent(agent_module, trainset: List[EvaluationExample], devset: Optional[List[EvaluationExample]] = None):
    """Optimize agent using MIPROv2"""
    
    # Use devset or split trainset if no devset provided
    if devset is None:
        split_point = int(len(trainset) * 0.8)
        train_data = trainset[:split_point]
        dev_data = trainset[split_point:]
    else:
        train_data = trainset
        dev_data = devset
    
    # Create optimizer
    optimizer = create_optimizer(train_data)
    
    # Optimize
    print(f"Optimizing agent with {len(train_data)} training examples...")
    optimized_agent = optimizer.compile(
        agent_module,
        trainset=train_data,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        requires_permission_to_run=False
    )
    
    # Evaluate on dev set
    print(f"Evaluating on {len(dev_data)} dev examples...")
    total_score = 0.0
    for example in dev_data:
        try:
            prediction = optimized_agent(query=example.query)
            score = composite_metric(example, prediction)
            total_score += score
        except Exception as e:
            print(f"Evaluation error: {e}")
    
    avg_score = total_score / len(dev_data) if dev_data else 0.0
    print(f"Average score: {avg_score:.3f}")
    
    return optimized_agent, avg_score


# ---------- Sample Data Generation ----------

def create_sample_evaluation_data() -> List[EvaluationExample]:
    """Create sample evaluation data for testing"""
    
    # Sample queries of different types and complexities
    sample_data = [
        {
            "query": "What is the population of Tokyo?",
            "analysis": QueryAnalysis(
                query_type="straightforward",
                complexity="simple",
                main_concepts=["population", "Tokyo"],
                key_entities=["Tokyo"],
                relationships=["city-population"],
                answer_format="single fact"
            ),
            "plan": ResearchPlan(steps=[
                PlanStep(id=1, description="Search for current Tokyo population", budget_calls=3)
            ])
        },
        {
            "query": "Compare the top 3 cloud providers",
            "analysis": QueryAnalysis(
                query_type="breadth_first",
                complexity="medium",
                main_concepts=["cloud providers", "comparison"],
                key_entities=["AWS", "Azure", "GCP"],
                relationships=["provider-features", "provider-pricing"],
                answer_format="comparative analysis"
            ),
            "plan": ResearchPlan(steps=[
                PlanStep(id=1, description="Research AWS features and pricing", budget_calls=5),
                PlanStep(id=2, description="Research Azure features and pricing", budget_calls=5),
                PlanStep(id=3, description="Research GCP features and pricing", budget_calls=5)
            ])
        },
        {
            "query": "What are the best approaches to building AI finance agents in 2025?",
            "analysis": QueryAnalysis(
                query_type="depth_first",
                complexity="complex",
                main_concepts=["AI finance agents", "development approaches", "2025 trends"],
                key_entities=["finance", "AI agents", "development"],
                relationships=["technology-finance", "agent-architecture"],
                answer_format="detailed report with multiple perspectives"
            ),
            "plan": ResearchPlan(steps=[
                PlanStep(id=1, description="Research current AI finance technologies", budget_calls=7),
                PlanStep(id=2, description="Analyze regulatory considerations", budget_calls=6),
                PlanStep(id=3, description="Study successful implementations", budget_calls=6),
                PlanStep(id=4, description="Examine future trends and predictions", budget_calls=5)
            ])
        }
    ]
    
    return [
        EvaluationExample(
            query=item["query"],
            analysis=item["analysis"],
            plan=item["plan"]
        )
        for item in sample_data
    ]


if __name__ == "__main__":
    # Quick test
    sample_data = create_sample_evaluation_data()
    evaluator = LeadAgentEvaluator()
    
    print("Testing evaluation pipeline...")
    for i, example in enumerate(sample_data):
        print(f"\nExample {i+1}: {example.query}")
        result = evaluator.forward(example)
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Delegation: {result.delegation_score:.3f}")
        print(f"Efficiency: {result.efficiency_score:.3f}")
        print(f"Classification: {result.classification_score:.3f}")
        print(f"Coherence: {result.coherence_score:.3f}")