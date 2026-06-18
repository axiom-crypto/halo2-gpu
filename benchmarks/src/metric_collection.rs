use std::ffi::OsStr;

use metrics_tracing_context::MetricsLayer;
use metrics_util::debugging::DebuggingRecorder;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

#[cfg(feature = "metrics")]
use {
    dashmap::DashMap,
    metrics_tracing_context::TracingContextLayer,
    metrics_util::layers::Layer as MetricsRecorderLayer,
    std::{sync::Arc, time::Instant},
    tracing::{
        field::{Field, Visit},
        Id, Subscriber,
    },
    tracing_subscriber::{registry::LookupSpan, Layer},
};

#[cfg(feature = "nvtx")]
use openvm_stark_sdk::nvtx_tracing::NvtxLayer;

/// Run a function with metric collection enabled.
///
/// Halo2-gpu emits timing spans as `halo2_section` with a `phase` field. The
/// local timing layer preserves the SDK metric format while rewriting that
/// `phase` label to the full halo2 parent path.
pub fn run_with_metric_collection<R>(
    output_path_envar: impl AsRef<OsStr>,
    f: impl FnOnce() -> R,
) -> R {
    let file = std::env::var(output_path_envar).map(|path| std::fs::File::create(path).unwrap());

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,p3_=warn"));
    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());
    #[cfg(feature = "metrics")]
    let subscriber = subscriber.with(NestedTimingMetricsLayer::new());
    #[cfg(feature = "nvtx")]
    let subscriber = subscriber.with(NvtxLayer::new(Default::default()));
    tracing::subscriber::set_global_default(subscriber).unwrap();

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    #[cfg(feature = "metrics")]
    {
        let recorder = TracingContextLayer::all().layer(recorder);
        metrics::set_global_recorder(recorder).unwrap();
    }

    let res = f();

    if let Ok(file) = file {
        serde_json::to_writer_pretty(
            &file,
            &openvm_stark_sdk::bench::serialize_metric_snapshot(snapshotter.snapshot()),
        )
        .unwrap();
    }

    res
}

#[cfg(feature = "metrics")]
#[derive(Clone, Default)]
struct NestedTimingMetricsLayer {
    span_timings: Arc<DashMap<Id, SpanTiming>>,
}

#[cfg(feature = "metrics")]
#[derive(Debug)]
struct SpanTiming {
    name: String,
    start_time: Instant,
    labels: Vec<(String, String)>,
    local_phase: Option<String>,
}

#[cfg(feature = "metrics")]
struct ReturnValueVisitor {
    has_return: bool,
}

#[cfg(feature = "metrics")]
impl Visit for ReturnValueVisitor {
    fn record_debug(&mut self, field: &Field, _value: &dyn std::fmt::Debug) {
        if field.name() == "return" {
            self.has_return = true;
        }
    }

    fn record_i64(&mut self, _field: &Field, _value: i64) {}
    fn record_u64(&mut self, _field: &Field, _value: u64) {}
    fn record_bool(&mut self, _field: &Field, _value: bool) {}
    fn record_str(&mut self, _field: &Field, _value: &str) {}
}

#[cfg(feature = "metrics")]
#[derive(Default)]
struct LabelVisitor {
    labels: Vec<(String, String)>,
    phase: Option<String>,
}

#[cfg(feature = "metrics")]
impl Visit for LabelVisitor {
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
    fn record_i64(&mut self, _field: &Field, _value: i64) {}
    fn record_u64(&mut self, _field: &Field, _value: u64) {}
    fn record_bool(&mut self, _field: &Field, _value: bool) {}

    fn record_str(&mut self, field: &Field, value: &str) {
        let key = field.name().to_string();
        let value = value.to_string();
        if key == "phase" {
            self.phase = Some(value.clone());
        }
        self.labels.push((key, value));
    }
}

#[cfg(feature = "metrics")]
impl NestedTimingMetricsLayer {
    fn new() -> Self {
        Self::default()
    }

    fn emit_metric(name: &str, duration_ms: f64, labels: &[(String, String)]) {
        let metric_name = format!("{name}_time_ms");
        let labels = labels
            .iter()
            .map(|(k, v)| metrics::Label::new(k.clone(), v.clone()))
            .collect::<Vec<_>>();
        metrics::gauge!(metric_name, labels).set(duration_ms);
    }
}

#[cfg(feature = "metrics")]
impl<S> Layer<S> for NestedTimingMetricsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else { return };
        let metadata = span.metadata();
        if metadata.level() > &tracing::Level::INFO {
            return;
        }

        let mut label_visitor = LabelVisitor::default();
        attrs.record(&mut label_visitor);
        let mut labels = label_visitor.labels;

        let local_phase = label_visitor.phase;
        if metadata.name() == "halo2_section" {
            if let Some(phase) = local_phase.as_ref() {
                let mut parent_phases = Vec::new();
                let mut parent = span.parent();
                while let Some(parent_span) = parent {
                    if let Some(timing) = self.span_timings.get(&parent_span.id()) {
                        if let Some(parent_phase) = timing.local_phase.as_ref() {
                            parent_phases.push(parent_phase.clone());
                        }
                    }
                    parent = parent_span.parent();
                }
                parent_phases.reverse();
                parent_phases.push(phase.clone());

                if let Some((_, label_phase)) = labels.iter_mut().find(|(key, _)| key == "phase") {
                    *label_phase = parent_phases.join(".");
                }
            }
        }

        self.span_timings.insert(
            id.clone(),
            SpanTiming {
                name: metadata.name().to_string(),
                start_time: Instant::now(),
                labels,
                local_phase,
            },
        );
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = ReturnValueVisitor { has_return: false };
        event.record(&mut visitor);

        if visitor.has_return {
            if let Some(span) = ctx.event_span(event) {
                let span_id = span.id();
                if let Some((_, timing)) = self.span_timings.remove(&span_id) {
                    let duration_ms = timing.start_time.elapsed().as_millis() as f64;
                    Self::emit_metric(&timing.name, duration_ms, &timing.labels);
                }
            }
        }
    }

    fn on_close(&self, id: Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some((_, timing)) = self.span_timings.remove(&id) {
            let duration_ms = timing.start_time.elapsed().as_millis() as f64;
            Self::emit_metric(&timing.name, duration_ms, &timing.labels);
        }
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use std::sync::{Once, OnceLock};

    use metrics_util::debugging::Snapshotter;
    use metrics_util::MetricKind;
    use tracing::{info_span, Span};

    use super::*;

    #[test]
    fn halo2_section_phase_labels_include_all_parents() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            let subscriber = Registry::default().with(NestedTimingMetricsLayer::new());
            tracing::subscriber::with_default(subscriber, || {
                let span_a = info_span!("halo2_section", phase = "A");
                let _guard_a = span_a.enter();

                let span_b = info_span!("halo2_section", phase = "B");
                let _guard_b = span_b.enter();

                let span_c = info_span!("halo2_section", phase = "C");
                let _guard_c = span_c.enter();
            });
        });

        let mut phases = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter_map(|(ckey, _, _, _)| {
                if ckey.kind() != MetricKind::Gauge {
                    return None;
                }
                let (_, key) = ckey.into_parts();
                let (name, labels) = key.into_parts();
                if name.as_str() != "halo2_section_time_ms" {
                    return None;
                }
                labels.into_iter().find_map(|label| {
                    let (key, value) = label.into_parts();
                    (key.as_ref() == "phase").then(|| value.to_string())
                })
            })
            .collect::<Vec<_>>();

        phases.sort();
        assert_eq!(phases, ["A", "A.B", "A.B.C"]);
    }

    // Cross-thread tests share one process-wide subscriber + recorder because
    // `set_global_default` / `set_global_recorder` are one-shot, and a spawned
    // thread cannot see thread-local `with_default` / `with_local_recorder`
    // overrides set on another thread.
    fn ensure_global_metrics() -> &'static Snapshotter {
        static SNAPSHOTTER: OnceLock<Snapshotter> = OnceLock::new();
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            let layered = TracingContextLayer::all().layer(recorder);
            metrics::set_global_recorder(layered)
                .expect("global metrics recorder must be installable for this test");
            SNAPSHOTTER.set(snapshotter).ok();

            let subscriber = Registry::default()
                .with(MetricsLayer::new())
                .with(NestedTimingMetricsLayer::new());
            tracing::subscriber::set_global_default(subscriber)
                .expect("global tracing subscriber must be installable for this test");
        });
        SNAPSHOTTER.get().expect("snapshotter installed by INIT")
    }

    #[test]
    fn group_label_survives_thread_scope() {
        let snapshotter = ensure_global_metrics();

        const GROUP: &str = "test_thread_scope_group";
        const PHASE: &str = "test_thread_scope_inner_phase";

        let outer = info_span!("test_outer", group = GROUP);
        let _enter = outer.enter();
        let parent = Span::current();

        std::thread::scope(|s| {
            s.spawn(|| {
                parent.in_scope(|| {
                    let _section = info_span!("halo2_section", phase = PHASE).entered();
                });
            });
        });

        let found = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .any(|(ckey, _, _, _)| {
                if ckey.kind() != MetricKind::Gauge {
                    return false;
                }
                let (_, key) = ckey.into_parts();
                let (name, labels) = key.into_parts();
                if name.as_str() != "halo2_section_time_ms" {
                    return false;
                }
                let mut phase_match = false;
                let mut group_match = false;
                for label in labels {
                    let (k, v) = label.into_parts();
                    if k.as_ref() == "phase" && v.as_ref() == PHASE {
                        phase_match = true;
                    }
                    if k.as_ref() == "group" && v.as_ref() == GROUP {
                        group_match = true;
                    }
                }
                phase_match && group_match
            });
        assert!(
            found,
            "expected halo2_section_time_ms metric with phase={PHASE} to carry group={GROUP}",
        );
    }
}
