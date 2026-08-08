#![cfg(feature = "metal")]

use candle_core::{Device, Result, Tensor};

// readbacks must wait on the command buffer holding their own blit, not shared device state
#[test]
fn concurrent_readback() -> Result<()> {
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope.spawn(move || {
                for iter in 0..100usize {
                    let value = (thread * 1000 + iter) as f64;
                    let a = Tensor::full(value as f32, (64, 64), &device).unwrap();
                    let b = a.affine(2.0, 1.0).unwrap();
                    let values = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let expected = (2.0 * value + 1.0) as f32;
                    assert!(
                        values.iter().all(|&x| x == expected),
                        "thread {thread} iter {iter}: expected {expected}, got {:?}",
                        &values[..4]
                    );
                }
            });
        }
    });
    Ok(())
}

// The async readback (`metal_readback_async`) must return the exact same bytes as an immediate
// `to_vec1`, whether or not GPU work is encoded between committing the readback and waiting on
// it — the scenario the encode/execute overlap pipeline depends on (issue the readback, encode
// the next step's work, wait on the readback later). A regression here would silently return
// stale or wrong values rather than crash (the risk class the design's own Risks section names).
#[test]
fn async_readback_matches_sync_with_intervening_work() -> Result<()> {
    let device = Device::new_metal(0)?;
    for iter in 0..50usize {
        let value = iter as f32;

        // Baseline: compute the same tensor and read it back immediately (existing sync path).
        let a = Tensor::full(value, (64, 64), &device)?;
        let expected: Vec<f32> = a.affine(3.0, 1.0)?.flatten_all()?.to_vec1()?;

        // Async: commit the readback, then encode a batch of UNRELATED GPU work that is never
        // itself read back before finally waiting on the pending readback — mirrors "submit N+1
        // before reading N."
        let a2 = Tensor::full(value, (64, 64), &device)?;
        let b2 = a2.affine(3.0, 1.0)?.flatten_all()?;
        let pending = b2.metal_readback_async::<f32>()?;
        for _ in 0..8 {
            let noise = Tensor::rand(-1f32, 1f32, (256, 256), &device)?;
            let _ = noise.affine(2.0, 0.5)?; // encode-only; deliberately never read back here
        }
        let got = pending.wait_and_read();
        assert_eq!(
            got, expected,
            "iter {iter}: async readback diverged from the sync baseline"
        );
    }
    Ok(())
}

// `wait_and_read_into` exists so a caller with a fixed-size, per-tick readback (StreamObserver's
// logits materialization) can reuse one buffer's allocation across calls instead of paying a
// fresh allocation every time — a naive implementation could either (a) leave a stale tail when a
// later read is shorter than an earlier one, since `out` isn't truncated by a plain
// `extend_from_slice`, or (b) silently stop reusing the buffer (allocate-and-reassign instead of
// clear-and-extend), which would be correct but defeat the entire point. This test catches both:
// content must match the sync baseline at every size, `out.len()` must track the CURRENT read
// exactly even after a larger previous one, and the buffer's pointer must stay stable across
// repeated same-size calls (proof of in-place reuse, not reassignment).
#[test]
fn async_readback_into_reused_buffer_no_stale_tail() -> Result<()> {
    let device = Device::new_metal(0)?;

    let expected_at = |value: f32, n: usize| -> Result<Vec<f32>> {
        Tensor::full(value, n, &device)?.affine(3.0, 1.0)?.to_vec1()
    };
    let pending_at = |value: f32, n: usize| -> Result<candle_core::MetalPendingReadback<f32>> {
        Tensor::full(value, n, &device)?
            .affine(3.0, 1.0)?
            .metal_readback_async::<f32>()
    };

    let mut out: Vec<f32> = vec![-999.0; 400]; // pre-filled, LARGER than every read below

    // A larger stale buffer must be truncated to exactly the new, smaller count — not just
    // overwritten in its first N elements while old sentinel values linger past that point.
    let n1 = 256;
    let expected1 = expected_at(1.0, n1)?;
    pending_at(1.0, n1)?.wait_and_read_into(&mut out);
    assert_eq!(out.len(), n1, "did not shrink from the pre-filled 400 elements");
    assert_eq!(out, expected1, "content diverged from the sync baseline");

    // Now read something SMALLER still, reusing the same (now len-256) buffer: must shrink again,
    // with nothing left over from the previous 256-element read.
    let n2 = 64;
    let expected2 = expected_at(2.0, n2)?;
    pending_at(2.0, n2)?.wait_and_read_into(&mut out);
    assert_eq!(out.len(), n2, "did not shrink from the prior 256-element read");
    assert_eq!(out, expected2, "content diverged from the sync baseline");
    assert!(
        !out.contains(&-999.0) && !out.contains(&expected1[0]),
        "stale tail: a value from an earlier, longer read survived into a shorter one"
    );

    // Repeated same-size calls must reuse the buffer's own allocation (pointer-stable), not
    // reassign a freshly allocated Vec each time — that would be correct but silently defeat the
    // reuse this method exists for.
    let n3 = 64;
    pending_at(3.0, n3)?.wait_and_read_into(&mut out);
    let ptr_after_first_n3 = out.as_ptr();
    for iter in 0..5 {
        pending_at(3.0 + iter as f32, n3)?.wait_and_read_into(&mut out);
        assert_eq!(
            out.as_ptr(),
            ptr_after_first_n3,
            "iter {iter}: buffer reallocated on a same-size call — reuse was not in-place"
        );
        assert_eq!(out, expected_at(3.0 + iter as f32, n3)?);
    }

    Ok(())
}

// Same property under concurrency (mirrors `concurrent_readback` above): each thread's pending
// readback must resolve to ITS OWN value, not a sibling thread's, even though all threads share
// one command queue and interleave their commits and waits.
#[test]
fn concurrent_async_readback() -> Result<()> {
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope.spawn(move || {
                for iter in 0..50usize {
                    let value = (thread * 1000 + iter) as f64;
                    let a = Tensor::full(value as f32, (64, 64), &device).unwrap();
                    let b = a.affine(2.0, 1.0).unwrap().flatten_all().unwrap();
                    let pending = b.metal_readback_async::<f32>().unwrap();
                    // Give other threads' work a chance to interleave before this thread waits.
                    let noise = Tensor::rand(-1f32, 1f32, (128, 128), &device).unwrap();
                    let _ = noise.affine(1.0, 0.0).unwrap();
                    let values = pending.wait_and_read();
                    let expected = (2.0 * value + 1.0) as f32;
                    assert!(
                        values.iter().all(|&x| x == expected),
                        "thread {thread} iter {iter}: expected {expected}, got {:?}",
                        &values[..4]
                    );
                }
            });
        }
    });
    Ok(())
}

// MECHANISM CONTROL (positive control for the async readback, not a correctness test — the
// correctness tests above already establish that). `async_readback_matches_sync_with_intervening_work`
// would pass unchanged on a build where `to_cpu_async` simply called the old synchronous `to_cpu` —
// byte-equality can't distinguish the two paths, since the sync path also returns correct bytes.
// Only TIMING can: the old `to_cpu` path's readback rides the same FIFO command queue as
// everything else, so a read issued after queuing unrelated "next step" work waits for that work
// to finish (`flush_and_wait_current` waits on the queue tail). The new async-handle path commits
// its blit BEFORE that work is queued and waits on its OWN retained handle — decoupled from
// whatever gets queued after it. The two are structurally different regardless of values, and the
// margin between them is large (a heavy matmul chain vs a handle wait), so a generous, non-flaky
// threshold: PIPE_NEW (the new path) must land near PLAIN (nothing queued), not near PIPE_OLD/HEAVY
// (the old path, which structurally cannot beat the queued work's own cost). `#[ignore]`: takes
// real wall-clock measurements; run manually:
//   cargo test --release --features metal --test metal_concurrent_tests -- --ignored async_readback_decouples --nocapture
#[test]
#[ignore = "timing-sensitive mechanism control; run manually, see module comment"]
fn async_readback_decouples_from_queued_work() -> Result<()> {
    use std::time::Instant;

    const N: usize = 1024;
    const HEAVY_DEPTH: usize = 24;
    const REPS: usize = 12;

    fn chain(x: &Tensor, w: &Tensor, depth: usize) -> Result<Tensor> {
        let mut acc = x.clone();
        for _ in 0..depth {
            acc = acc.matmul(w)?;
        }
        Ok(acc)
    }

    fn median(mut v: Vec<u128>) -> u128 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    let dev = Device::new_metal(0)?;
    let w = (Tensor::ones((N, N), candle_core::DType::F32, &dev)? * (1.0 / N as f64))?;
    let x = Tensor::ones((1, N), candle_core::DType::F32, &dev)?;

    // Warm-up: get past first-call allocation/compile overhead before any timed rep.
    for _ in 0..6 {
        let y = chain(&x, &w, HEAVY_DEPTH)?;
        let _ = y.flatten_all()?.to_vec1::<f32>()?;
    }
    dev.synchronize()?;

    let mut heavy = Vec::new();
    for _ in 0..REPS {
        dev.synchronize()?;
        let t = Instant::now();
        let y = chain(&x, &w, HEAVY_DEPTH)?;
        let _ = y.flatten_all()?.to_vec1::<f32>()?;
        heavy.push(t.elapsed().as_micros());
    }

    let mut plain = Vec::new();
    for _ in 0..REPS {
        dev.synchronize()?;
        let n_out = chain(&x, &w, 1)?.flatten_all()?.contiguous()?;
        let t = Instant::now();
        let v = n_out.to_vec1::<f32>()?;
        plain.push(t.elapsed().as_micros());
        std::hint::black_box(v);
    }

    let mut pipe_new = Vec::new();
    for _ in 0..REPS {
        dev.synchronize()?;
        let n_out = chain(&x, &w, 1)?.flatten_all()?.contiguous()?;
        let want: Vec<f32> = n_out.to_vec1()?; // sync baseline for the same tensor
        dev.synchronize()?;

        let pending = n_out.metal_readback_async::<f32>()?;
        let next = chain(&x, &w, HEAVY_DEPTH)?; // "token N+1" — encoded AFTER the readback commit
        let t = Instant::now();
        let got = pending.wait_and_read();
        pipe_new.push(t.elapsed().as_micros());
        assert_eq!(got, want, "async readback returned wrong values");
        std::hint::black_box(next);
    }

    let (p, n2, h) = (median(plain), median(pipe_new), median(heavy));
    eprintln!("PLAIN median {p}us  PIPE_NEW median {n2}us  HEAVY reference {h}us");

    // Generous margin (per review discussion): PIPE_NEW should land near PLAIN, nowhere near
    // PIPE_OLD's structural floor (HEAVY's own cost, since that path must wait for the queued
    // forward). 0.25x HEAVY is comfortably below PIPE_OLD's expected ~1x HEAVY while leaving real
    // headroom over PLAIN for scheduling noise.
    assert!(
        (n2 as f64) < (p as f64) + 0.25 * (h as f64),
        "handle wait did not decouple from queued work: PIPE_NEW {n2}us, PLAIN {p}us, HEAVY {h}us \
         — the async handle may be secretly waiting on the queue tail"
    );
    Ok(())
}

#[test]
fn concurrent_quantized_data_roundtrip() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope.spawn(move || {
                for iter in 0..25usize {
                    let src = Tensor::rand(-1f32, 1f32, (256, 256), &device).unwrap();
                    let q = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
                    let bytes = q.data().unwrap();
                    let q2 = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
                    let bytes2 = q2.data().unwrap();
                    assert_eq!(
                        bytes, bytes2,
                        "thread {thread} iter {iter}: data() readback mismatch"
                    );
                }
            });
        }
    });
    Ok(())
}
