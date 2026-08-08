use crate::metal::{
    BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline, Device,
    Fence, LastFence, ResidencySet,
};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;

fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer, MetalKernelError> {
    command_queue.commandBuffer().map(CommandBuffer::new).ok_or(
        MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()),
    )
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: MutexGuard<'a, EntryState>,
}

impl AsRef<ComputeCommandEncoder> for CommandsGuard<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.guard.current_encoder.as_ref().unwrap()
    }
}

impl CommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.as_ref().set_compute_pipeline_state(pipeline);
    }

    #[cfg(feature = "debug-labels")]
    #[must_use = "the debug group is popped when the returned guard is dropped"]
    pub fn debug_group(&self, label: &str) -> crate::metal::DebugGroupGuard<'_> {
        self.as_ref().debug_group(label)
    }
}

/// RAII guard for blit command encoder operations.
pub struct BlitCommandsGuard<'a> {
    _guard: MutexGuard<'a, EntryState>,
    state: BlitCommandEncoder,
}

impl<'a> AsRef<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_ref(&self) -> &BlitCommandEncoder {
        &self.state
    }
}

impl<'a> AsMut<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_mut(&mut self) -> &mut BlitCommandEncoder {
        &mut self.state
    }
}

impl BlitCommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.as_mut()
            .copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size)
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.as_mut().fill_buffer(buffer, range, value);
    }
}

impl Drop for BlitCommandsGuard<'_> {
    fn drop(&mut self) {
        self.as_ref().end_encoding();
    }
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
    current_encoder: Option<ComputeCommandEncoder>,
}

impl EntryState {
    pub fn new(cb: CommandBuffer) -> EntryState {
        EntryState {
            current: cb,
            in_flight: vec![],
            current_encoder: None,
        }
    }
}

pub struct Commands {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    device: Device,
    /// Fence of the immediately preceding encoder (compute or blit), if any. Encoders are
    /// created strictly serially — guarded by `state` — so each new encoder only needs to wait
    /// on this one fence: fence transitivity (each encoder already waited on its own
    /// predecessor before running) makes that equivalent to waiting on every earlier write,
    /// without per-buffer bookkeeping. Lives on `Commands` (not `EntryState`) so it carries
    /// across `commit_swap_locked` command-buffer boundaries.
    last_fence: LastFence,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: &ResidencySet,
    ) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        if let Some(raw) = residency_set.raw() {
            command_queue.addResidencySet(raw);
        }

        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue)?;

        Ok(Self {
            state: Mutex::new(EntryState::new(cb)),
            compute_count: AtomicUsize::new(0),
            command_queue,
            compute_per_buffer,
            device,
            last_fence: Arc::new(Mutex::new(None)),
        })
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        if state_guard.current_encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device));
            let enc = state_guard.current.compute_command_encoder(&fence);
            // Wait for the previous encoder's fence before the first dispatch. Because
            // encoders are created strictly serially, this single wait transitively covers
            // every earlier write. Using HazardTrackingModeUntracked implies that Metal does
            // not automatically flush GPU caches at encoder or command buffer boundaries.
            {
                let guard = self.last_fence.lock().unwrap();
                if let Some(prev) = guard.as_ref() {
                    enc.wait_for_fence(prev);
                }
            }
            state_guard.current_encoder = Some(enc);
        }

        Ok(CommandsGuard { guard: state_guard })
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // End compute encoder before starting blit.
        if let Some(enc) = state_guard.current_encoder.take() {
            self.end_encoding(enc);
        }

        let fence = Arc::new(Fence::new(&self.device));
        let encoder = state_guard
            .current
            .blit_command_encoder(&fence, &self.last_fence);

        // Wait for the previous encoder's fence before any blit commands execute (same
        // chained-fence reasoning as `command_encoder` above). Required for
        // HazardTrackingModeUntracked: GPU caches are not auto-flushed.
        {
            let guard = self.last_fence.lock().unwrap();
            if let Some(prev) = guard.as_ref() {
                encoder.wait_for_fence(prev);
            }
        }

        Ok(BlitCommandsGuard {
            _guard: state_guard,
            state: encoder,
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        let to_wait = {
            let mut state = self.state.lock()?;
            if self.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&mut state, 0)?;
            }
            std::mem::take(&mut state.in_flight)
        };

        // Wait only on the last CB. Metal executes CBs in queue order, so all earlier
        // CBs are guaranteed complete when the last one is. Calling waitUntilCompleted on
        // each CB individually pays OS notification latency (~1-2ms) N times unnecessarily.
        if let Some(last) = to_wait.last() {
            Self::ensure_completed(last)?;
        }
        // Check earlier CBs for errors (no need to block — they're already done).
        for cb in &to_wait[..to_wait.len().saturating_sub(1)] {
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
        }

        // Deliberately NOT resetting last_fence here. Everything queued before `to_wait` was
        // snapshotted is now GPU-complete (ensure_completed + Metal's FIFO CB ordering), so
        // waiting on it again is a no-op — an already-signaled fence is satisfied immediately.
        // Resetting to None would have to happen after `to_wait` is taken, i.e. outside the
        // `state` lock: a concurrent thread could acquire `state`, encode + end an encoder,
        // and publish its fence as the new last_fence in that window, which a reset here would
        // then discard, breaking the chain for whoever comes after. Leaving the stale fence in
        // place is strictly conservative (worst case one harmless extra wait) and race-free.

        Ok(())
    }

    /// Commit the current command buffer and wait on that specific buffer, for CPU readbacks.
    /// [`Self::wait_until_completed`] waits on the last in-flight buffer, which a concurrent
    /// `flush_and_wait` on another thread may already have taken, returning before our work ran.
    pub fn flush_and_wait_current(&self) -> Result<(), MetalKernelError> {
        let cb = {
            let mut state = self.state.lock()?;
            self.commit_swap_locked(&mut state, 0)?;
            state.in_flight.last().cloned()
        };
        if let Some(cb) = cb {
            Self::ensure_completed(&cb)?;
            // queue is FIFO: everything committed before cb is done too
            let mut state = self.state.lock()?;
            state
                .in_flight
                .retain(|c| c.status() != MTLCommandBufferStatus::Completed);
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.state.lock()?;
        if self.compute_count.load(Ordering::Acquire) > 0 {
            self.commit_swap_locked(&mut state, 0)?;
        }
        Ok(())
    }

    /// Commit the current command buffer WITHOUT waiting, returning a clone of the just-committed
    /// buffer as a waitable handle. Unlike [`Self::wait_until_completed`] /
    /// [`Self::flush_and_wait_current`], which both block on `state.in_flight`'s LAST entry at
    /// CALL time (the queue tail — which by the time of a later wait may include far more work
    /// than the caller's own), the returned handle lets the caller wait on THIS SPECIFIC buffer
    /// later, once its own GPU work is actually needed: `CommandBuffer::wait_until_completed` on
    /// an already-completed buffer returns immediately (Apple's documented behavior), so a wait
    /// deferred past enough other encoded work is effectively free. Mirrors
    /// `flush_and_wait_current`'s unconditional commit (always commits, even if nothing new was
    /// encoded since the last flush) so the returned handle always denotes "everything encoded up
    /// to this call" — never a stale, already-superseded buffer.
    pub fn flush_returning_handle(&self) -> Result<CommandBuffer, MetalKernelError> {
        let mut state = self.state.lock()?;
        self.commit_swap_locked(&mut state, 0)?;
        // commit_swap_locked unconditionally pushes the just-committed buffer, so this is always
        // populated — the same invariant `flush_and_wait_current` relies on via `.cloned()`.
        Ok(state
            .in_flight
            .last()
            .cloned()
            .expect("commit_swap_locked always pushes exactly one buffer"))
    }

    fn commit_swap_locked(
        &self,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }

        match state.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                state.current.commit();
            }
            _ => {}
        }
        let new_cb = create_command_buffer(&self.command_queue)?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        self.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn end_encoding(&self, encoder: ComputeCommandEncoder) {
        use objc2_metal::MTLCommandEncoder as _;
        use objc2_metal::MTLComputeCommandEncoder as _;

        // Signal this encoder's completion fence, chain it as the wait target for whatever
        // encoder comes next (compute or blit), and end encoding.
        encoder.raw.updateFence(encoder.fence.raw());
        *self.last_fence.lock().unwrap() = Some(Arc::clone(&encoder.fence));
        encoder.raw.endEncoding();
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
