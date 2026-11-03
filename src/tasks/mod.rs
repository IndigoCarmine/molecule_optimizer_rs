//! Asynchronous job infrastructure.
//!
//! * [`Engine`] owns a dedicated rayon thread pool. CPU-heavy work never runs
//!   on the tokio runtime; jobs are awaited through a oneshot channel, so the
//!   API works from any async runtime (or synchronously via
//!   [`JobHandle::wait_blocking`]).
//! * Every job publishes a typed snapshot `S` of its current state. Readers
//!   get it lock-free at any time with [`JobHandle::snapshot`], or ask the
//!   workers for a fresh one with [`JobHandle::request_snapshot`], or
//!   follow the intermediate states as a `Stream` with [`JobHandle::stream`].
//! * Workers publish when a snapshot was requested, or periodically
//!   (default every 50 ms); between those points no copies are made.
//! * [`JobHandle::cancel`] stops the job at the next iteration boundary; jobs
//!   return their best result so far where that makes sense.

use crate::error::{Error, Result};
use arc_swap::ArcSwap;
use futures::Stream;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, oneshot, watch};
use tokio_util::sync::CancellationToken;

/// Progress counter of a job.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Progress {
    pub done: usize,
    pub total: usize,
    /// Set when the job has finished (successfully or not).
    pub finished: bool,
}

impl Progress {
    pub fn fraction(&self) -> f64 {
        if self.total == 0 { if self.finished { 1.0 } else { 0.0 } } else { self.done as f64 / self.total as f64 }
    }
}

/// A published state of a job.
#[derive(Debug, Clone)]
pub struct Snapshot<S> {
    /// Monotonically increasing version (0 = initial state).
    pub version: u64,
    /// Time since the job started.
    pub elapsed: Duration,
    pub data: S,
}

struct Shared<S> {
    data: Mutex<S>,
    published: ArcSwap<Snapshot<S>>,
    version: AtomicU64,
    request_epoch: AtomicU64,
    cancel: CancellationToken,
    notify: Notify,
    finished: AtomicBool,
    start: Instant,
    last_publish: Mutex<Instant>,
    interval: Duration,
    progress: watch::Sender<Progress>,
}

/// Worker-side handle: publishes snapshots, reports progress, checks for
/// cancellation. Cheap to clone and `Send + Sync`.
pub struct JobContext<S> {
    shared: Arc<Shared<S>>,
}

impl<S> Clone for JobContext<S> {
    fn clone(&self) -> Self {
        JobContext { shared: self.shared.clone() }
    }
}

impl<S: Clone + Send + Sync + 'static> JobContext<S> {
    /// `true` once [`JobHandle::cancel`] was called.
    pub fn is_cancelled(&self) -> bool {
        self.shared.cancel.is_cancelled()
    }
    /// Current snapshot-request epoch. A worker that remembers the epoch it
    /// last served can tell whether a new request is pending.
    pub fn request_epoch(&self) -> u64 {
        self.shared.request_epoch.load(Ordering::Acquire)
    }
    /// `true` if a snapshot request newer than `served_epoch` is pending, or
    /// the periodic publish interval has elapsed.
    pub fn should_publish(&self, served_epoch: u64) -> bool {
        if self.request_epoch() > served_epoch {
            return true;
        }
        if self.shared.interval.is_zero() {
            return false;
        }
        match self.shared.last_publish.try_lock() {
            Ok(t) => t.elapsed() >= self.shared.interval,
            Err(_) => false,
        }
    }
    /// Mutates the job state and publishes a new snapshot.
    pub fn update(&self, f: impl FnOnce(&mut S)) {
        let mut d = self.shared.data.lock().unwrap();
        f(&mut d);
        self.publish_locked(&d);
    }
    /// Mutates the job state without publishing (cheap bookkeeping).
    pub fn update_silent(&self, f: impl FnOnce(&mut S)) {
        let mut d = self.shared.data.lock().unwrap();
        f(&mut d);
    }
    fn publish_locked(&self, d: &S) {
        let v = self.shared.version.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.published.store(Arc::new(Snapshot { version: v, elapsed: self.shared.start.elapsed(), data: d.clone() }));
        if let Ok(mut t) = self.shared.last_publish.lock() {
            *t = Instant::now();
        }
        self.shared.notify.notify_waiters();
    }
    /// Publishes the current state as-is.
    pub fn publish(&self) {
        let d = self.shared.data.lock().unwrap();
        self.publish_locked(&d);
    }
    /// Reports progress.
    pub fn set_progress(&self, done: usize, total: usize) {
        self.shared.progress.send_replace(Progress { done, total, finished: false });
    }
    /// Increments the `done` counter.
    pub fn inc_progress(&self) {
        self.shared.progress.send_modify(|p| p.done += 1);
    }
    /// Cancellation token (e.g. to link child work).
    pub fn cancellation_token(&self) -> CancellationToken {
        self.shared.cancel.clone()
    }
}

/// Handle to a running job producing `R` and publishing snapshots `S`.
pub struct JobHandle<R, S> {
    shared: Arc<Shared<S>>,
    rx: oneshot::Receiver<Result<R>>,
}

impl<R, S: Clone + Send + Sync + 'static> JobHandle<R, S> {
    /// Latest published snapshot (never blocks).
    pub fn snapshot(&self) -> Arc<Snapshot<S>> {
        self.shared.published.load_full()
    }

    /// Asks the workers to publish their current state and waits until a
    /// newer snapshot is available (or the job finished, or `timeout`).
    pub async fn request_snapshot_timeout(&self, timeout: Duration) -> Arc<Snapshot<S>> {
        let v0 = self.shared.version.load(Ordering::Acquire);
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.shared.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            self.shared.request_epoch.fetch_add(1, Ordering::AcqRel);
            if self.shared.version.load(Ordering::Acquire) > v0 || self.shared.finished.load(Ordering::Acquire) {
                return self.snapshot();
            }
            match tokio::time::timeout_at(deadline, notified).await {
                Ok(()) => {
                    if self.shared.version.load(Ordering::Acquire) > v0 {
                        return self.snapshot();
                    }
                }
                Err(_) => return self.snapshot(),
            }
        }
    }

    /// [`request_snapshot_timeout`](Self::request_snapshot_timeout) with a 10 s timeout.
    pub async fn request_snapshot(&self) -> Arc<Snapshot<S>> {
        self.request_snapshot_timeout(Duration::from_secs(10)).await
    }

    /// Blocking variant of [`request_snapshot`](Self::request_snapshot) for
    /// non-async callers.
    pub fn request_snapshot_blocking(&self, timeout: Duration) -> Arc<Snapshot<S>> {
        let v0 = self.shared.version.load(Ordering::Acquire);
        self.shared.request_epoch.fetch_add(1, Ordering::AcqRel);
        let t0 = Instant::now();
        while self.shared.version.load(Ordering::Acquire) <= v0
            && !self.shared.finished.load(Ordering::Acquire)
            && t0.elapsed() < timeout
        {
            std::thread::sleep(Duration::from_micros(200));
        }
        self.snapshot()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.shared.cancel.cancel();
    }

    pub fn is_finished(&self) -> bool {
        self.shared.finished.load(Ordering::Acquire)
    }

    /// Progress watch channel.
    pub fn progress(&self) -> watch::Receiver<Progress> {
        self.shared.progress.subscribe()
    }

    /// Progress as a `Stream` (yields on every change, ends when finished).
    pub fn progress_stream(&self) -> impl Stream<Item = Progress> + Send + 'static {
        let rx = self.shared.progress.subscribe();
        futures::stream::unfold((rx, false), |(mut rx, done)| async move {
            if done {
                return None;
            }
            let p = *rx.borrow_and_update();
            if p.finished {
                return Some((p, (rx, true)));
            }
            if rx.changed().await.is_err() {
                return None;
            }
            let p = *rx.borrow_and_update();
            Some((p, (rx, p.finished)))
        })
    }

    /// Intermediate states of the job as a `Stream` of snapshot data (works
    /// for every job type: `optimize`, `optimize_batch`, `scan`,
    /// `conformer_search`, `assembly_search`, custom [`Engine::spawn`] jobs).
    /// For optimizations, [`JobHandle::frames`] yields just the [`Frame`](crate::calc::Frame)s.
    ///
    /// Same semantics as [`snapshot_stream`](Self::snapshot_stream): it
    /// yields what the workers publish (periodically every
    /// [`EngineConfig::publish_interval`], on [`request_snapshot`](Self::request_snapshot),
    /// and once at the end; with `Duration::ZERO` only requested and final
    /// states) and never makes them publish more often, so it
    /// does not slow the computation. A slow consumer skips intermediate
    /// states (latest value, no buffering). The stream ends after the final
    /// state, also when the job is cancelled or fails. Any number of streams
    /// may be open at once.
    ///
    /// ```no_run
    /// # use molecule_optimizer_rs::{calc::OptimizeOptions, ff::uff::UffBuilder, tasks::Engine, Molecule};
    /// # use futures::StreamExt;
    /// # use std::sync::Arc;
    /// # async fn f(mol: Arc<Molecule>) {
    /// let job = Engine::global().optimize(mol, Arc::new(UffBuilder::default()), OptimizeOptions::default());
    /// let mut states = job.stream();
    /// while let Some(s) = states.next().await {
    ///     println!("iter {} E = {}", s.frame.iteration, s.frame.energy);
    /// }
    /// let result = job.await.unwrap();
    /// # }
    /// ```
    pub fn stream(&self) -> futures::stream::BoxStream<'static, S>
    where
        R: Send + 'static,
    {
        use futures::StreamExt;
        self.snapshot_stream().map(|s| s.data.clone()).boxed()
    }

    /// Snapshots as a `Stream`, yielding each newly published version (with
    /// version number and elapsed time; see [`stream`](Self::stream) for the
    /// semantics). The initial state (version 0) is not yielded.
    pub fn snapshot_stream(&self) -> impl Stream<Item = Arc<Snapshot<S>>> + Send + 'static
    where
        R: Send + 'static,
    {
        let shared = self.shared.clone();
        futures::stream::unfold((shared, 0u64, false), |(shared, last, done)| async move {
            if done {
                return None;
            }
            loop {
                let outcome = {
                    let notified = shared.notify.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    // `finished` first: the final snapshot is published
                    // before it is set, so it is visible once `fin` is seen
                    let fin = shared.finished.load(Ordering::Acquire);
                    let snap = shared.published.load_full();
                    if snap.version > last || (fin && last == 0) {
                        Some(Some(snap))
                    } else if fin {
                        Some(None)
                    } else {
                        notified.await;
                        None
                    }
                };
                match outcome {
                    Some(Some(snap)) => {
                        let v = snap.version;
                        return Some((snap, (shared, v, false)));
                    }
                    Some(None) => return None,
                    None => continue,
                }
            }
        })
    }

    /// Blocks the current thread until the job finishes.
    pub fn wait_blocking(self) -> Result<R> {
        self.rx.blocking_recv().unwrap_or_else(|_| Err(Error::Job("worker dropped".into())))
    }
}

impl<R: Send + 'static, S: Send + Sync + 'static> IntoFuture for JobHandle<R, S> {
    type Output = Result<R>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<R>> + Send>>;
    fn into_future(self) -> Self::IntoFuture {
        let rx = self.rx;
        let _keep = self.shared;
        Box::pin(async move {
            let r = rx.await.unwrap_or_else(|_| Err(Error::Job("worker dropped".into())));
            drop(_keep);
            r
        })
    }
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Worker threads (`None` = number of CPUs).
    pub num_threads: Option<usize>,
    /// Periodic snapshot interval (`Duration::ZERO` = only on request).
    pub publish_interval: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig { num_threads: None, publish_interval: Duration::from_millis(50) }
    }
}

/// Runs jobs on a dedicated rayon pool.
pub struct Engine {
    pool: rayon::ThreadPool,
    config: EngineConfig,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").field("threads", &self.pool.current_num_threads()).field("config", &self.config).finish()
    }
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        let mut b = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("mm-worker-{i}"));
        if let Some(n) = config.num_threads {
            b = b.num_threads(n.max(1));
        }
        let pool = b.build().map_err(|e| Error::Job(e.to_string()))?;
        Ok(Engine { pool, config })
    }

    /// Process-wide default engine.
    pub fn global() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(|| Engine::new(EngineConfig::default()).expect("thread pool"))
    }

    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    pub fn pool(&self) -> &rayon::ThreadPool {
        &self.pool
    }

    /// Spawns a job. `f` runs on the engine's pool; any rayon parallelism
    /// inside it uses the same pool.
    pub fn spawn<R, S, F>(&self, initial: S, total: usize, f: F) -> JobHandle<R, S>
    where
        R: Send + 'static,
        S: Clone + Send + Sync + 'static,
        F: FnOnce(JobContext<S>) -> Result<R> + Send + 'static,
    {
        let (ptx, _prx) = watch::channel(Progress { done: 0, total, finished: false });
        let shared = Arc::new(Shared {
            published: ArcSwap::from_pointee(Snapshot { version: 0, elapsed: Duration::ZERO, data: initial.clone() }),
            data: Mutex::new(initial),
            version: AtomicU64::new(0),
            request_epoch: AtomicU64::new(0),
            cancel: CancellationToken::new(),
            notify: Notify::new(),
            finished: AtomicBool::new(false),
            start: Instant::now(),
            last_publish: Mutex::new(Instant::now()),
            interval: self.config.publish_interval,
            progress: ptx,
        });
        let (tx, rx) = oneshot::channel();
        let ctx = JobContext { shared: shared.clone() };
        self.pool.spawn(move || {
            let shared = ctx.shared.clone();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(ctx.clone())))
                .unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "panic".into());
                    Err(Error::Job(msg))
                });
            // final snapshot
            ctx.publish();
            shared.finished.store(true, Ordering::Release);
            shared.progress.send_modify(|p| p.finished = true);
            shared.notify.notify_waiters();
            let _ = tx.send(r);
        });
        JobHandle { shared, rx }
    }

    /// Runs `f` on the pool and waits for it (synchronous helper).
    pub fn install<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        self.pool.install(f)
    }
}

/// Helper used by calculations: turns a [`JobContext`] into an optimizer
/// observer that checks cancellation and publishes via `publish(pos, energy, iter)`.
pub struct CtxObserver<'a, S, F>
where
    F: FnMut(&JobContext<S>, &crate::optim::IterationInfo<'_>),
{
    pub ctx: &'a JobContext<S>,
    pub served_epoch: u64,
    pub on_publish: F,
}

impl<'a, S, F> CtxObserver<'a, S, F>
where
    S: Clone + Send + Sync + 'static,
    F: FnMut(&JobContext<S>, &crate::optim::IterationInfo<'_>),
{
    pub fn new(ctx: &'a JobContext<S>, on_publish: F) -> Self {
        CtxObserver { ctx, served_epoch: 0, on_publish }
    }
}

impl<'a, S, F> crate::optim::Observer for CtxObserver<'a, S, F>
where
    S: Clone + Send + Sync + 'static,
    F: FnMut(&JobContext<S>, &crate::optim::IterationInfo<'_>),
{
    fn on_iteration(&mut self, info: &crate::optim::IterationInfo<'_>) -> std::ops::ControlFlow<()> {
        if self.ctx.is_cancelled() {
            return std::ops::ControlFlow::Break(());
        }
        if self.ctx.should_publish(self.served_epoch) {
            self.served_epoch = self.ctx.request_epoch();
            (self.on_publish)(self.ctx, info);
        }
        std::ops::ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn snapshot_and_cancel() {
        let eng = Engine::new(EngineConfig { num_threads: Some(2), publish_interval: Duration::ZERO }).unwrap();
        let h = eng.spawn(0u64, 0, |ctx: JobContext<u64>| {
            let mut i = 0u64;
            let mut served = 0;
            while !ctx.is_cancelled() {
                i += 1;
                if ctx.should_publish(served) {
                    served = ctx.request_epoch();
                    ctx.update(|s| *s = i);
                }
                std::thread::sleep(Duration::from_micros(100));
            }
            Ok(i)
        });
        let mut last_v = 0;
        let mut last_val = 0;
        for _ in 0..5 {
            let s = h.request_snapshot().await;
            assert!(s.version > last_v);
            assert!(s.data >= last_val);
            last_v = s.version;
            last_val = s.data;
        }
        h.cancel();
        let r = h.await.unwrap();
        assert!(r >= last_val);
    }

    #[test]
    fn panic_becomes_error() {
        let eng = Engine::new(EngineConfig { num_threads: Some(1), ..Default::default() }).unwrap();
        let h = eng.spawn((), 0, |_ctx| -> Result<()> { panic!("boom") });
        assert!(matches!(h.wait_blocking(), Err(Error::Job(m)) if m.contains("boom")));
    }
}
