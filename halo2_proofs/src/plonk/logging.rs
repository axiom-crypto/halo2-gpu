// Adapted from ark-std's perf_trace.rs.

#[allow(dead_code, unused_imports)]
#[macro_use]
pub mod inner {
    pub use colored::Colorize;
    pub use lazy_static::lazy_static;
    pub use std::{
        collections::HashMap,
        format,
        string::{String, ToString},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
        thread::ThreadId,
        time::Instant,
    };

    pub const PAD_CHAR: &str = "·";

    pub struct TimerInfo {
        pub msg: String,
        pub time: Instant,
    }

    pub struct NestedEvent {
        pub root_msg: String,
        pub num_indent: AtomicUsize,
    }

    lazy_static! {
        pub static ref NUM_INDENTS: Arc<Mutex<HashMap<ThreadId, NestedEvent>>> =
            Arc::new(Mutex::new(HashMap::new()));
    }

    #[macro_export]
    macro_rules! start_timer {
        ($msg:expr) => {{
            use $crate::plonk::logging::inner::{
                compute_indent, AtomicUsize, Colorize, Instant, NestedEvent, Ordering, ToString,
                NUM_INDENTS,
            };

            let msg = $msg();
            let thread_id = std::thread::current().id();
            let start_info = "Start:".yellow().bold();
            let (root_msg, indent_amount) = {
                let mut num_indents = NUM_INDENTS.lock().unwrap();
                let num_indent_opt = num_indents.get_mut(&thread_id);
                if num_indent_opt.is_none() {
                    num_indents.insert(
                        thread_id,
                        NestedEvent { root_msg: msg.to_string(), num_indent: AtomicUsize::new(0) },
                    );
                }
                let event = num_indents.get_mut(&thread_id).unwrap();
                let num_indent = &event.num_indent;
                let indent_amount = 2 * num_indent.fetch_add(0, Ordering::Relaxed);
                if indent_amount == 0 {
                    event.root_msg = msg.to_string();
                }

                num_indent.fetch_add(1, Ordering::Relaxed);

                (event.root_msg.clone(), indent_amount)
            };
            let indent = compute_indent(indent_amount);

            log::debug!("[{}] {}{:8} {}", root_msg, indent, start_info, msg);
            $crate::plonk::logging::inner::TimerInfo { msg: msg.to_string(), time: Instant::now() }
        }};
    }

    #[macro_export]
    macro_rules! end_timer {
        ($time:expr) => {{
            end_timer!($time, || "");
        }};
        ($time:expr, $msg:expr) => {{
            use $crate::plonk::logging::inner::{
                compute_indent, format, Colorize, Ordering, NUM_INDENTS,
            };

            let time = $time.time;
            let final_time = time.elapsed();
            let final_time = {
                let secs = final_time.as_secs();
                let millis = final_time.subsec_millis();
                let micros = final_time.subsec_micros() % 1000;
                let nanos = final_time.subsec_nanos() % 1000;
                if secs != 0 {
                    format!("{}.{:03}s", secs, millis).bold()
                } else if millis > 0 {
                    format!("{}.{:03}ms", millis, micros).bold()
                } else if micros > 0 {
                    format!("{}.{:03}µs", micros, nanos).bold()
                } else {
                    format!("{}ns", final_time.subsec_nanos()).bold()
                }
            };

            let end_info = "End:".green().bold();
            let message = format!("{} {}", $time.msg, $msg());
            let (root_msg, indent_amount) = {
                let num_indents = NUM_INDENTS.lock().unwrap();
                let thread_id = std::thread::current().id();
                let event = num_indents.get(&thread_id).unwrap();
                let num_indent = &event.num_indent;
                num_indent.fetch_sub(1, Ordering::Relaxed);
                let indent_amount = 2 * num_indent.fetch_add(0, Ordering::Relaxed);

                (event.root_msg.clone(), indent_amount)
            };
            let indent = compute_indent(indent_amount);

            log::debug!(
                "[{}] {}{:8} {:.<pad$}{}",
                root_msg,
                indent,
                end_info,
                message,
                final_time,
                pad = 75 - indent_amount
            );
        }};
    }

    pub fn compute_indent(indent_amount: usize) -> String {
        let mut indent = String::new();
        for _ in 0..indent_amount {
            indent.push_str(&PAD_CHAR.white());
        }
        indent
    }
}
