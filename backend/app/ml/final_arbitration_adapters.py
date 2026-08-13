from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.ml.final_decision_router import (
    FinalDeadliftProbeContext,
    FinalDecisionState,
    FinalPushPressProbeContext,
    YoloDeadliftRecoveryContext,
    apply_final_deadlift_probe_result,
    apply_final_push_press_probe_result,
    apply_yolo_deadlift_recovery_result,
    final_state_from_decision,
    plan_final_deadlift_probe,
    plan_final_push_press_probe,
    plan_yolo_deadlift_recovery,
)


@dataclass
class FinalArbitrationProbeAdapters:
    biomechanics: list[dict[str, Any]]
    forced_exercise_label: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    wrist_overhead_ratio: float
    explosive_score: float
    bodyweight_debug: dict[str, Any]
    bar_debug: dict[str, Any]
    use_yolo_tracking: bool
    summarize_biomechanics: Callable[[list[dict[str, Any]]], dict[str, Any]]
    analyze_push_press_reps: Callable[[list[dict[str, Any]], str], Any]
    analyze_deadlift_reps: Callable[[list[dict[str, Any]]], Any]
    analyze_yolo_deadlift_reps: Callable[[list[dict[str, Any]]], list[Any]]
    analyze_squat_reps: Callable[[list[dict[str, Any]], str], Any]
    yolo_deadlift_probe_reps: list[Any] = field(default_factory=list)

    def push_press_probe(self, state: FinalDecisionState) -> FinalDecisionState:
        context = FinalPushPressProbeContext(
            forced_exercise_label=self.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=self.raw_label,
            base_confidence=self.base_confidence,
            bio_label=self.bio_label,
            bio_confidence=self.bio_confidence,
            squat_label=self.squat_label,
            squat_confidence=self.squat_confidence,
            bodyweight_router_label=self.bodyweight_router_label,
            bodyweight_router_confidence=self.bodyweight_router_confidence,
            olympic_pred=self.olympic_pred,
            olympic_confidence=self.olympic_confidence,
            explosive_score=self.explosive_score,
            bar_debug=self.bar_debug,
        )
        plan = plan_final_push_press_probe(context)
        probe_reps = []

        if plan.should_probe:
            try:
                probe_result = self.analyze_push_press_reps(
                    self.biomechanics,
                    "push_press",
                )
                probe_reps = (
                    probe_result[0]
                    if isinstance(probe_result, tuple)
                    else probe_result
                ) or []
            except Exception:
                probe_reps = []

        return final_state_from_decision(
            apply_final_push_press_probe_result(
                context,
                probe_reps=probe_reps,
                minimum_rep_count=plan.minimum_rep_count,
                recovery_reason=plan.recovery_reason,
            )
        )

    def deadlift_probe(self, state: FinalDecisionState) -> FinalDecisionState:
        try:
            summary = self.summarize_biomechanics(self.biomechanics) or {}
        except Exception:
            summary = {}

        knee_range = max(
            0.0,
            float(summary.get("max_knee_angle", 0.0) or 0.0)
            - float(summary.get("min_knee_angle", 0.0) or 0.0),
        )
        hip_range = max(
            0.0,
            float(summary.get("max_hip_angle", 0.0) or 0.0)
            - float(summary.get("min_hip_angle", 0.0) or 0.0),
        )
        torso_range = max(
            0.0,
            float(summary.get("max_torso_angle", 0.0) or 0.0)
            - float(summary.get("min_torso_angle", 0.0) or 0.0),
        )

        context = FinalDeadliftProbeContext(
            forced_exercise_label=self.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=self.raw_label,
            base_confidence=self.base_confidence,
            bio_label=self.bio_label,
            bio_confidence=self.bio_confidence,
            squat_label=self.squat_label,
            squat_confidence=self.squat_confidence,
            router_v6_label=self.router_v6_label,
            router_v6_confidence=self.router_v6_confidence,
            wrist_overhead_ratio=self.wrist_overhead_ratio,
            explosive_score=self.explosive_score,
            deadlift_knee_range=knee_range,
            deadlift_hip_range=hip_range,
            deadlift_torso_range=torso_range,
            bodyweight_debug=self.bodyweight_debug,
            bar_debug=self.bar_debug,
        )
        plan = plan_final_deadlift_probe(context)
        probe_reps = []

        if plan.should_probe:
            try:
                probe_result = self.analyze_deadlift_reps(self.biomechanics)
                probe_reps = (
                    probe_result[0]
                    if isinstance(probe_result, tuple)
                    else probe_result
                ) or []
            except Exception:
                probe_reps = []

        decision = apply_final_deadlift_probe_result(
            context,
            probe_reps=probe_reps,
            recovery_reason=plan.recovery_reason,
        )
        return final_state_from_decision(decision)

    def yolo_deadlift_recovery(
        self,
        state: FinalDecisionState,
    ) -> FinalDecisionState:
        self.yolo_deadlift_probe_reps = []
        squat_probe_reps = []
        context = YoloDeadliftRecoveryContext(
            use_yolo_tracking=self.use_yolo_tracking,
            forced_exercise_label=self.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=self.raw_label,
            bio_label=self.bio_label,
            olympic_confidence=self.olympic_confidence,
        )
        plan = plan_yolo_deadlift_recovery(context)

        if plan.should_probe:
            try:
                self.yolo_deadlift_probe_reps = self.analyze_yolo_deadlift_reps(
                    self.biomechanics
                )
                squat_probe_reps, _ = self.analyze_squat_reps(
                    self.biomechanics,
                    plan.squat_probe_label,
                )
            except Exception as exc:
                print(f"YOLO deadlift recovery skipped: {exc}")

        return final_state_from_decision(
            apply_yolo_deadlift_recovery_result(
                context,
                deadlift_probe_reps=self.yolo_deadlift_probe_reps,
                squat_probe_reps=squat_probe_reps,
            )
        )
