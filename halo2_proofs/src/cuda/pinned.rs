//! Page-locked (pinned) host-buffer pool for staging host-to-device copies.
//!
//! A pool hit returns a registered, all-zero buffer that copies H2D at full
//! PCIe bandwidth; a miss falls back to a fresh pageable allocation. Returned
//! buffers go to a background cleaner that registers them once
//! (`cudaHostRegister`) and re-zeroes only the written prefix, keeping both
//! costs off the caller's critical path.
//!
//! Async-safety: a `cudaMemcpyAsync` from pinned memory returns with the DMA
//! still in flight, so a returned buffer must not be reused until enqueued work
//! has drained. The cleaner therefore device-synchronizes before touching any
//! buffer, which lets callers `give_back` immediately after enqueueing the copy.

use std::{
    collections::{BTreeMap, HashSet},
    ffi::c_void,
    sync::{mpsc, Mutex, OnceLock},
};

use openvm_cuda_common::{error::CudaError, stream::device_synchronize};

use crate::cuda::utils::ensure_current_device_matches_ctx;

extern "C" {
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
}

/// A returned buffer together with its dirty-prefix length.
type ReturnedBuffer = (Vec<u8>, usize);

/// Page-locks `len` bytes at `ptr`. Returns `false` (buffer stays pageable) on
/// failure.
pub fn register_region(ptr: *mut u8, len: usize) -> bool {
    // SAFETY: [ptr, ptr+len) is a live allocation owned by the caller.
    let rc = unsafe { cudaHostRegister(ptr as *mut c_void, len, 0) };
    if rc != 0 {
        tracing::debug!(
            "cudaHostRegister failed: {}; buffer stays pageable",
            CudaError::new(rc)
        );
        return false;
    }
    true
}

/// Registered, all-zero buffers ready for reuse, keyed by allocation size.
fn pool() -> &'static Mutex<BTreeMap<usize, Vec<Vec<u8>>>> {
    static POOL: OnceLock<Mutex<BTreeMap<usize, Vec<Vec<u8>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Base pointers of buffers whose `cudaHostRegister` succeeded.
fn registered() -> &'static Mutex<HashSet<usize>> {
    static REGISTERED: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    REGISTERED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Cleaner thread: registers (first cycle) and re-zeroes returned buffers off
/// the critical path, then makes them available to [`take`].
fn cleaner() -> &'static Mutex<mpsc::Sender<ReturnedBuffer>> {
    static TX: OnceLock<Mutex<mpsc::Sender<ReturnedBuffer>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ReturnedBuffer>();
        std::thread::Builder::new()
            .name("pinned-cleaner".into())
            .spawn(move || {
                // Bind this fresh thread to the CUDA ctx device before any
                // CUDA call; unbound it would default to logical device 0.
                if ensure_current_device_matches_ctx().is_err() {
                    tracing::debug!("pinned-cleaner: device bind failed; pool disabled");
                    return;
                }
                while let Ok(first) = rx.recv() {
                    // Coalesce a burst of returns behind one device sync.
                    let mut batch = vec![first];
                    while batch.len() < 64 {
                        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(next) => batch.push(next),
                            Err(_) => break,
                        }
                    }
                    // Wait for the in-flight H2D copies reading these buffers
                    // to drain before touching their contents.
                    if let Err(e) = device_synchronize() {
                        tracing::debug!("cudaDeviceSynchronize failed: {e}; dropping batch");
                        continue;
                    }
                    for (mut buffer, dirty_len) in batch {
                        let ptr = buffer.as_mut_ptr();
                        let is_new = !registered().lock().unwrap().contains(&(ptr as usize));
                        if is_new {
                            if !register_region(ptr, buffer.len()) {
                                continue;
                            }
                            registered().lock().unwrap().insert(ptr as usize);
                        }
                        let dirty_len = dirty_len.min(buffer.len());
                        buffer[..dirty_len].fill(0);
                        pool()
                            .lock()
                            .unwrap()
                            .entry(buffer.len())
                            .or_default()
                            .push(buffer);
                    }
                }
            })
            .expect("failed to spawn pinned-cleaner thread");
        Mutex::new(tx)
    })
}

/// Returns an all-zero buffer of at least `min_size` bytes (rounded up to the
/// next power of two), page-locked if it came from the pool.
pub fn take(min_size: usize) -> Vec<u8> {
    let size = min_size.next_power_of_two();
    if let Some(buffer) = pool()
        .lock()
        .unwrap()
        .get_mut(&size)
        .and_then(|bufs| bufs.pop())
    {
        debug_assert_eq!(buffer.len(), size);
        return buffer;
    }
    // Pool miss: a plain pageable allocation (registered on give_back for reuse).
    vec![0u8; size]
}

/// Hands `buffer` back to the cleaner for registration, re-zeroing, and reuse.
/// `dirty_len` is an upper bound on the written prefix; the rest must be zero.
pub fn give_back(buffer: Vec<u8>, dirty_len: usize) {
    if buffer.is_empty() || !buffer.len().is_power_of_two() {
        return; // not a pool-shaped buffer; drop normally
    }
    // The cleaner's receiver never drops before teardown; a send error there
    // just means we drop the buffer, which is fine.
    let _ = cleaner().lock().unwrap().send((buffer, dirty_len));
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn wait_for_pooled(size: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while pool()
            .lock()
            .unwrap()
            .get(&size)
            .is_none_or(|bufs| bufs.is_empty())
        {
            assert!(Instant::now() < deadline, "size-{size} buffer never pooled");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn take_rounds_up_to_next_power_of_two_and_zero_fills() {
        for (min_size, expected) in [(1, 1), (3, 4), (1024, 1024), (1025, 2048)] {
            let buf = take(min_size);
            assert_eq!(buf.len(), expected, "take({min_size})");
            assert!(buf.iter().all(|&b| b == 0), "take({min_size}) not zeroed");
        }
    }

    #[test]
    fn round_trip_recycles_registered_rezeroed_buffer() {
        const SIZE: usize = 1 << 13;
        let mut buf = take(SIZE);
        let ptr = buf.as_ptr() as usize;
        buf.fill(0xAB);
        give_back(buf, usize::MAX); // oversized dirty_len must clamp, not panic
        wait_for_pooled(SIZE);
        assert!(
            registered().lock().unwrap().contains(&ptr),
            "pooled buffer was not page-locked"
        );
        let buf = take(SIZE);
        assert_eq!(buf.as_ptr() as usize, ptr, "pool hit should reuse the alloc");
        assert!(buf.iter().all(|&b| b == 0), "recycled buffer not re-zeroed");
    }
}
