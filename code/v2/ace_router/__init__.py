"""Training-free TRACE-Writer v1 implementation for physGen v2."""

from .trace_compile import CompiledTrace, compile_trace_plan
from .trace_config import TraceWriterConfig
from .demo_inputs import DemoInputError, DemoTraceInput, parse_sample_ids, resolve_demo_inputs
from .trace_masks import fixed5_temporal_weights, flatten_stage_gates
from .trace_model_adapter import TraceModelProxy
from .trace_pipeline import PreparedTrace, prepare_trace_contexts
from .trace_runtime import RoutingRuntime, build_routing_runtime
from .trace_schema import FIXED_STAGE_IDS, CausalStage, TracePlan, load_trace_plan
from .trace_validation import ValidationIssue, ValidationReport, validate_trace_plan
from .trace_writer import trace_block_forward
from .upstream_guard import capture_upstream, verify_expected_upstream

__all__ = [
    "CausalStage",
    "CompiledTrace",
    "DemoInputError",
    "DemoTraceInput",
    "FIXED_STAGE_IDS",
    "TracePlan",
    "TraceModelProxy",
    "TraceWriterConfig",
    "ValidationIssue",
    "ValidationReport",
    "RoutingRuntime",
    "PreparedTrace",
    "build_routing_runtime",
    "compile_trace_plan",
    "capture_upstream",
    "fixed5_temporal_weights",
    "flatten_stage_gates",
    "load_trace_plan",
    "prepare_trace_contexts",
    "parse_sample_ids",
    "resolve_demo_inputs",
    "trace_block_forward",
    "validate_trace_plan",
    "verify_expected_upstream",
]
