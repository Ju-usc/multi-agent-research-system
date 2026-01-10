"""
BrowseComp Evaluation Module

Evaluates the multi-agent research system on BrowseComp using DSPy's built-in evaluation framework.
"""

import time
import logging

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from agent import Agent
from models import BrowseCompJudge, LLMJudgeAnswer
from config import (
    ModelConfig,
    GRADER_MODEL,
    GRADER_MAX_TOKENS,
    OPTIMIZER_MODEL,
    OPTIMIZER_MAX_TOKENS,
    OPENROUTER_API_KEY,
)
from dataset import BrowseCompDataset
from utils import (
    create_model_cli_parser,
    start_cleanup_watchdog,
    create_isolated_workspace,
)

logger = logging.getLogger(__name__)

# Model pricing per 1M tokens
# Add your model here if using --lead/--sub with a custom model
LM_PRICING = {
    "openrouter/x-ai/grok-4.1-fast": {"input": 0.20, "output": 0.50},
    "openrouter/deepseek/deepseek-v3.2": {"input": 0.24, "output": 0.38},
    "openrouter/google/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "openrouter/google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "openrouter/openai/gpt-oss-120b": {"input": 0.039, "output": 0.19},
    "openrouter/qwen/qwen3-235b-a22b-2507": {"input": 0.071, "output": 0.463},
    "openrouter/openai/gpt-5.2": {"input": 1.75, "output": 14.0, "cached_input": 0.175},
}

class MultiAgentResearchSystem(dspy.Module):
    """
    DSPy program wrapper for Agent to make it compatible with dspy.Evaluate.

    Agent is created as module attribute so GEPA can discover and optimize tools.
    Uses deepcopy() for thread-safe parallel evaluation.
    """

    def __init__(self, config: ModelConfig | None = None, *, log_dir: str = "logs/eval"):
        super().__init__()
        self.agent = Agent(config=config, work_dir="memory_eval/default", log_dir=log_dir)

    def forward(self, problem: str) -> dspy.Prediction:
        work_dir = create_isolated_workspace()

        try:
            # Use DSPy's built-in deepcopy to preserve optimized tool descriptions
            agent = self.agent.deepcopy()
            agent.reset_workspace(work_dir)

            start_time = time.perf_counter()
            prediction = agent(problem)
            
            prediction.report = prediction.answer
            prediction.elapsed_seconds = time.perf_counter() - start_time
            prediction.tool_cost_usd = agent.search_tool.total_cost_usd + agent.fetch_tool.total_cost_usd
            
            return prediction
        finally:
            pass  # Keep workspace for debugging (was: cleanup_workspace(work_dir))

class BrowseCompEvaluator:
    """Encapsulates BrowseComp evaluation with proper state management."""

    QUERY_TIMEOUT_SECONDS = 600  # 10 min, matches OpenAI Deep Research
    
    def __init__(self, args):
        self.args = args
        self.reflection_lm = None  # Initialized lazily if optimization requested
        self.total_cost_accumulated = 0.0
        
        # Initialize grader LM once for all evaluations (major efficiency improvement)
        self.grader_lm = dspy.LM(
            model=GRADER_MODEL,
            temperature=1.0,  # Required for GPT-5 reasoning models
            max_tokens=GRADER_MAX_TOKENS,
            api_key=OPENROUTER_API_KEY,
        )
        self.judge = dspy.ChainOfThought(BrowseCompJudge)

        # Initialize reflection LM for GEPA optimization if needed
        if args.optimize:
            self.reflection_lm = dspy.LM(
                model=OPTIMIZER_MODEL,
                temperature=1.0,  # Higher temp for creative prompt mutations
                max_tokens=OPTIMIZER_MAX_TOKENS,
                api_key=OPENROUTER_API_KEY,
            )
    
    def calculate_lm_cost(self, usage: dict) -> float:
        """Calculate LM cost with accurate input/output/cached token pricing.
        
        Pricing in LM_PRICING is per 1M tokens (industry standard).
        Formula: (tokens / 1,000,000) * price_per_1M = cost in USD
        """
        total_cost = 0.0
        
        for model_name, stats in usage.items():
            pricing = LM_PRICING[model_name]
            
            prompt_tokens = stats["prompt_tokens"]
            completion_tokens = stats["completion_tokens"]
            prompt_details = stats.get("prompt_tokens_details") or {}
            cached_tokens = prompt_details.get("cached_tokens", 0)
            non_cached_input = prompt_tokens - cached_tokens
            
            input_cost = (non_cached_input / 1_000_000) * pricing["input"]
            cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing["input"])
            output_cost = (completion_tokens / 1_000_000) * pricing["output"]
            
            total_cost += input_cost + cached_cost + output_cost
        
        return total_cost
    
    def grade_prediction(self, example: dspy.Example, pred: dspy.Prediction) -> LLMJudgeAnswer:
        """Grade prediction using grader LM."""
        # Skip grader for timeouts
        if pred.elapsed_seconds > self.QUERY_TIMEOUT_SECONDS:
            return LLMJudgeAnswer(
                is_correct=False,
                extracted_answer="TIMEOUT - NOT GRADED",
                reasoning=f"Exceeded {self.QUERY_TIMEOUT_SECONDS}s limit ({int(pred.elapsed_seconds)}s). Agent's answer: {pred.answer}"
            )
        
        try:
            with dspy.context(lm=self.grader_lm):
                result = self.judge(
                    question=example.problem,
                    report=pred.report,
                    correct_answer=example.answer
                )
            return result.answer
        except Exception as e:
            logger.error(f"Grading error: {e}")
            return LLMJudgeAnswer(
                is_correct=False,
                extracted_answer="Error",
                reasoning="Grading failed"
            )

    def metric(self, example, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
        """Unified metric for dspy.Evaluate and GEPA."""
        llm_grading = self.grade_prediction(example, pred)
        
        accuracy = 1.0 if llm_grading.is_correct else 0.0
        extracted_answer = llm_grading.extracted_answer
        reasoning = llm_grading.reasoning
        
        usage = pred.get_lm_usage() or {}
        total_cost = self.calculate_lm_cost(usage) + pred.tool_cost_usd
        self.total_cost_accumulated += total_cost
        
        # Composite score: accuracy / (1 + cost) - rewards correct + cheap
        composite_score = accuracy / (1 + total_cost)
        
        pred.metrics = {
            "accuracy": accuracy,
            "composite_score": composite_score,
            "total_cost_usd": total_cost,
            "elapsed_seconds": pred.elapsed_seconds,
        }
        
        feedback = (
            f"Score: {composite_score:.4f} (accuracy / (1 + cost_usd))\n"
            f"Accuracy: {accuracy:.0f}/1 | Cost (Token + Tool): ${total_cost:.4f}\n"
            f"Agent Answer: {pred.answer}\n"
            f"Grader Extracted Answer: {extracted_answer}\n"
            f"Ground Truth: {example.answer}\n"
            f"Grader Reasoning: {reasoning}\n"
            f"---\n"
            f"OPTIMIZATION GUIDANCE: Score = accuracy / (1 + cost). "
            f"If correct, optimize for cost efficiency. "
            f"If wrong, prioritize accuracy first."
        )
        
        return ScoreWithFeedback(score=composite_score, feedback=feedback)
    
    def optimize_with_gepa(self, program: MultiAgentResearchSystem, train: list, val: list) -> MultiAgentResearchSystem:
        """Run GEPA optimization on program."""
        optimizer = GEPA(
            metric=self.metric,
            reflection_lm=self.reflection_lm,
            max_full_evals=self.args.max_full_evals,
            num_threads=self.args.num_threads,
            track_stats=True,
            track_best_outputs=True,
            enable_tool_optimization=True,
            component_selector="all",
        )
        
        return optimizer.compile(student=program, trainset=train, valset=val)
    
    def run(self, program: MultiAgentResearchSystem, examples: list) -> tuple:
        """Run evaluation and return (result, predictions)."""
        predictions_dict = {}
        
        def metric_with_capture(example, pred, trace=None):
            result = self.metric(example, pred, trace)
            predictions_dict[example.problem] = pred
            return result
        
        evaluator = dspy.Evaluate(
            devset=examples,
            metric=metric_with_capture,
            num_threads=self.args.num_threads,
            display_progress=True,
            display_table=5,
            max_errors=10,
        )
        
        result = evaluator(program)
        predictions = self._extract_predictions(predictions_dict, examples)
        return result, predictions
    
    def _extract_predictions(self, predictions_dict: dict, examples: list) -> list:
        """Extract predictions in correct order, handling missing ones."""
        predictions = []
        for i, ex in enumerate(examples):
            pred = predictions_dict.get(ex.problem)
            if pred is None:
                logger.warning(f"Missing prediction for example {i}, creating placeholder")
                pred = dspy.Prediction(answer="ERROR", report="ERROR")
                pred.metrics = {"accuracy": 0.0, "elapsed_seconds": 0, "total_cost_usd": 0}
            predictions.append(pred)
        return predictions

def _parse_args():
    parser = create_model_cli_parser("Run BrowseComp evaluation")
    parser.add_argument("--offline", action="store_true", help="Use BrowseComp-Plus with local corpus (requires MCP server)")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of dataset examples (split 50/50 train/val)")
    parser.add_argument("--num-threads", type=int, default=5, help="Parallel evaluation threads")
    parser.add_argument("--optimize", action="store_true", help="Run GEPA optimization")
    parser.add_argument("--max-full-evals", type=int, default=4, help="Max full evaluation passes for GEPA (~iterations)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")
    parser.add_argument("--log-dir", default="logs/eval", help="Directory for agent execution logs (JSONL).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    print("🔍 BrowseComp Evaluation")
    print("=" * 50)
    print(f"⚖️  Grader model: {GRADER_MODEL} (fixed for consistency)")
    print("=" * 50)

    # Build config from CLI args
    config = ModelConfig(lead=args.lead, sub=args.sub, offline=args.offline)
    print(f"🤖 Models: lead={config.lead}, sub={config.sub}")

    if dspy.settings.lm is None:
        dspy.configure(
            lm=dspy.LM(
                model=config.lead,
                temperature=config.temperature,
                max_tokens=config.lead_max_tokens,
                api_key=OPENROUTER_API_KEY,
            ),
            adapter=ChatAdapter(),
        )

    dspy.settings.configure(track_usage=True)

    # Initialize evaluator with grader and optimizer LMs
    evaluator = BrowseCompEvaluator(args)

    # Load dataset based on --offline flag
    if args.offline:
        from browsecompplus import BrowseCompPlusDataset
        dataset = BrowseCompPlusDataset(
            "../BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
            num_examples=args.num_examples,
            seed=args.seed,
        )
        print("📚 Using BrowseComp-Plus (offline corpus)")
    else:
        dataset = BrowseCompDataset(num_examples=args.num_examples, seed=args.seed)

    examples = dataset.load()
    print(f"📚 Loaded {len(examples)} examples")

    # Create agent program
    program = MultiAgentResearchSystem(config=config, log_dir=args.log_dir)

    if args.optimize:
        train, val = dataset.split(train_size=0.5)
        
        logger.info("GEPA optimization starting")
        print(f"\n🧬 GEPA Optimization (max_full_evals={args.max_full_evals})")
        print(f"🤖 Reflection model: {OPTIMIZER_MODEL}")
        print(f"📊 Train: {len(train)}, Val: {len(val)}")
        
        optimized_program = evaluator.optimize_with_gepa(program, train, val)
        
        # Save optimized program immediately (before any potential crash)
        optimized_program.save("optimized_program.json")
        logger.info("Saved optimized program to optimized_program.json")
        
        # Start watchdog before accessing results (litellm may hang)
        start_cleanup_watchdog()
        
        results = optimized_program.detailed_results
        baseline_score = results.val_aggregate_scores[0]
        best_score = results.val_aggregate_scores[results.best_idx]
        
        # Derive accuracy from composite: if score > 0, accuracy = 1
        # Note: val_subscores[candidate_idx] is a dict {instance_id: score}, iterate over values
        baseline_accuracy = sum(1 for s in results.val_subscores[0].values() if s > 0) / len(val)
        best_accuracy = sum(1 for s in results.val_subscores[results.best_idx].values() if s > 0) / len(val)
        
        print("\n" + "=" * 50)
        print(f"📈 Baseline: accuracy={baseline_accuracy:.0%}, composite={baseline_score:.4f}")
        print(f"📈 Best:     accuracy={best_accuracy:.0%}, composite={best_score:.4f}")
        print(f"📊 Examples: {len(examples)} (train={len(train)}, val={len(val)})")
        print(f"🧬 Candidates: {len(results.candidates)}")
        print(f"🔄 Metric calls: {results.total_metric_calls}")
        
        print(f"💰 Total cost: ${evaluator.total_cost_accumulated:.2f}")
        logger.info(f"GEPA complete: baseline_acc={baseline_accuracy:.0%}, best_acc={best_accuracy:.0%}, candidates={len(results.candidates)}, cost=${evaluator.total_cost_accumulated:.2f}")
        
        # Compare baseline vs optimized prompts
        baseline_program = results.candidates[0]
        optimized_program = results.candidates[results.best_idx]
        
        def get_all_tools(program):
            """Get all tools from program: {name: tool}"""
            tools = {}
            for name, tool in program.agent.lead_agent.tools.items():
                tools[f"leadagent.{name}"] = tool
            for tool in program.agent.subagent_tool._tools:
                tools[f"subagent.{tool.name}"] = tool
            return tools
        
        # Compare predictor instructions
        print("\n📝 Predictor Instructions:")
        for (predictor_name, baseline_program_predictor), (_, optimized_program_predictor) in zip(
            baseline_program.named_predictors(), optimized_program.named_predictors()
        ):
            baseline_program_instructions = baseline_program_predictor.signature.instructions
            optimized_program_instructions = optimized_program_predictor.signature.instructions
            marker = "✨" if baseline_program_instructions != optimized_program_instructions else ""
            
            print(f"\n{marker} {predictor_name}:")
            print(f"  BASELINE:\n    {baseline_program_instructions}")
            print(f"  OPTIMIZED:\n    {optimized_program_instructions}")
            
            logger.info(f"{predictor_name} BASELINE: {baseline_program_instructions}")
            logger.info(f"{predictor_name} OPTIMIZED: {optimized_program_instructions}")
        
        # Compare tool descriptions
        baseline_program_tools = get_all_tools(baseline_program)
        optimized_program_tools = get_all_tools(optimized_program)
        print("\n🔧 Tool Descriptions:")
        for tool_name, optimized_program_tool in optimized_program_tools.items():
            baseline_program_tool = baseline_program_tools[tool_name]
            baseline_program_tool_description = baseline_program_tool.desc
            optimized_program_tool_description = optimized_program_tool.desc
            marker = "✨" if baseline_program_tool_description != optimized_program_tool_description else ""
            
            print(f"\n{marker} {tool_name}:")
            print(f"  BASELINE:\n    {baseline_program_tool_description}")
            print(f"  OPTIMIZED:\n    {optimized_program_tool_description}")
            
            logger.info(f"{tool_name} BASELINE: {baseline_program_tool_description}")
            logger.info(f"{tool_name} OPTIMIZED: {optimized_program_tool_description}")
    else:
        # Evaluation only (no optimization)
        print("🚀 Evaluating...")
        result, predictions = evaluator.run(program, examples)
        
        # Calculate total cost
        total_run_cost = sum(p.metrics.get("total_cost_usd", 0) for p in predictions)
        
        print("\n" + "=" * 50)
        print(f"📈 Score: {result.score:.4f}")
        print(f"💰 Total cost: ${total_run_cost:.2f}")
        logger.info(f"Total run cost: ${total_run_cost:.2f}")
        print(f"📊 Examples: {len(examples)}")

    start_cleanup_watchdog()


if __name__ == "__main__":
    main()
