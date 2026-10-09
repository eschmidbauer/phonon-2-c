/*  threadpool.c – Fork-join pool with spinning workers. See threadpool.h.
 *
 *  Workers spin on an atomic generation counter for a short while after each job before parking on a condition
 *  variable, so a dispatch is a store, an increment and – only when someone is parked – a broadcast. The
 *  mutex-protected protocol this replaced cost 30–60 µs per dispatch with 18 threads; an encoder pass makes about
 *  1,400 dispatches per 32 s chunk, and on a 10 s clip most of the thread-time was spent waiting on it.
 */

#include "threadpool.h"
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define P2_MAX_THREADS 64

/* Relaxed-loop iterations a worker spins before parking (P2_SPIN overrides): on the order of 100 µs, longer than
 * the gaps between the dispatches of an encoder pass and far shorter than the pause between files. 0 parks at once. */
#define SPIN_BEFORE_PARK 8000

struct ThreadPool {
    int         n_total;     /* participants: N-1 workers + the calling thread */
    int         spin;        /* iterations before parking */
    pthread_t   workers[P2_MAX_THREADS - 1];

    _Atomic unsigned gen;    /* bumped per job; workers watch it */
    _Atomic int      n_done; /* workers finished with the current job */
    _Atomic int      sleepers;
    _Atomic int      shutdown;

    ParallelFn  fn;          /* the current job, published by the gen increment */
    void       *arg;

    pthread_mutex_t mtx;     /* only for parking */
    pthread_cond_t  work_cv;
    pthread_mutex_t dispatch_mtx;   /* serialises concurrent p2_parallel() callers */
};

struct WorkerCtx {
    ThreadPool *pool;
    int         tid;         /* 1..N-1 */
};

static inline void cpu_relax(void)
{
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");   /* the usual spin-wait pause on Apple silicon; `yield` is a no-op there */
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

static void *worker_main(void *arg_)
{
    struct WorkerCtx *ctx = (struct WorkerCtx *)arg_;
    ThreadPool *p = ctx->pool;
    int tid = ctx->tid;
    unsigned last_gen = 0;

    for (;;) {
        unsigned g;
        int spins = 0;
        while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == last_gen &&
               !atomic_load_explicit(&p->shutdown, memory_order_relaxed)) {
            if (++spins < p->spin) { cpu_relax(); continue; }
            /* Park. The generation is re-read under the mutex, so a dispatch between our last check and the wait
             * is not missed: the dispatcher either sees us in `sleepers` and broadcasts, or we see its increment. */
            pthread_mutex_lock(&p->mtx);
            atomic_fetch_add(&p->sleepers, 1);
            while (atomic_load_explicit(&p->gen, memory_order_acquire) == last_gen &&
                   !atomic_load_explicit(&p->shutdown, memory_order_relaxed))
                pthread_cond_wait(&p->work_cv, &p->mtx);
            atomic_fetch_sub(&p->sleepers, 1);
            pthread_mutex_unlock(&p->mtx);
            spins = 0;
        }
        if (atomic_load_explicit(&p->shutdown, memory_order_relaxed)) {
            free(ctx);
            return NULL;
        }
        last_gen = g;
        p->fn(tid, p->n_total, p->arg);
        atomic_fetch_add_explicit(&p->n_done, 1, memory_order_release);
    }
}

/* Thread count from P2_THREADS or the online CPU count. */
static int choose_n_threads(void)
{
    const char *env = getenv("P2_THREADS");
    if (env) {
        int n = atoi(env);
        if (n < 1) n = 1;
        if (n > P2_MAX_THREADS) n = P2_MAX_THREADS;
        return n;
    }
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs < 1) nprocs = 1;
    if (nprocs > P2_MAX_THREADS) nprocs = P2_MAX_THREADS;
    return (int)nprocs;
}

static ThreadPool *make_pool(int n_total)
{
    if (n_total < 1) n_total = 1;
    if (n_total > P2_MAX_THREADS) n_total = P2_MAX_THREADS;

    ThreadPool *p = (ThreadPool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_total = n_total;
    p->spin = SPIN_BEFORE_PARK;
    const char *spin_env = getenv("P2_SPIN");
    if (spin_env && *spin_env) p->spin = atoi(spin_env);
    atomic_init(&p->gen, 0);
    atomic_init(&p->n_done, 0);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->shutdown, 0);
    pthread_mutex_init(&p->mtx, NULL);
    pthread_mutex_init(&p->dispatch_mtx, NULL);
    pthread_cond_init(&p->work_cv, NULL);

    for (int i = 0; i < n_total - 1; i++) {
        struct WorkerCtx *ctx = (struct WorkerCtx *)malloc(sizeof(*ctx));
        if (!ctx) { p->n_total = i + 1; break; }
        ctx->pool = p;
        ctx->tid  = i + 1;
        if (pthread_create(&p->workers[i], NULL, worker_main, ctx) != 0) {
            fprintf(stderr, "p2_pool: pthread_create failed\n");
            free(ctx);
            p->n_total = i + 1;
            break;
        }
    }
    return p;
}

static void teardown_pool(ThreadPool *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->mtx);
    atomic_store(&p->shutdown, 1);
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mtx);

    for (int i = 0; i < p->n_total - 1; i++)
        pthread_join(p->workers[i], NULL);

    pthread_mutex_destroy(&p->mtx);
    pthread_mutex_destroy(&p->dispatch_mtx);
    pthread_cond_destroy(&p->work_cv);
    free(p);
}

/* ── Process-wide singleton ──────────────────────────────────────────────── */

static ThreadPool *g_pool = NULL;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void destroy_global_pool(void)
{
    teardown_pool(g_pool);
    g_pool = NULL;
}

static void init_global_pool(void)
{
    g_pool = make_pool(choose_n_threads());
    atexit(destroy_global_pool);
}

/* Thread-local override, for hosts that shard sessions across independent pools. */
static __thread ThreadPool *g_tls_pool = NULL;

ThreadPool *p2_pool(void)
{
    if (g_tls_pool) return g_tls_pool;
    pthread_once(&g_pool_once, init_global_pool);
    return g_pool;
}

void p2_set_pool(ThreadPool *p)
{
    g_tls_pool = p;
}

ThreadPool *p2_pool_create(int n_threads)
{
    return make_pool(n_threads);
}

void p2_pool_destroy(ThreadPool *p)
{
    teardown_pool(p);
}

int p2_pool_nthreads(ThreadPool *p)
{
    return p->n_total;
}

void p2_parallel(ThreadPool *p, ParallelFn fn, void *arg)
{
    if (p->n_total <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&p->dispatch_mtx);

    p->fn = fn;
    p->arg = arg;
    atomic_store_explicit(&p->n_done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);   /* publishes fn and arg */
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mtx);
        pthread_cond_broadcast(&p->work_cv);
        pthread_mutex_unlock(&p->mtx);
    }

    fn(0, p->n_total, arg);

    /* The calling thread is on the critical path: spin for the workers, yield only if one has been descheduled. */
    long spins = 0;
    while (atomic_load_explicit(&p->n_done, memory_order_acquire) < p->n_total - 1) {
        if (++spins > 200000) sched_yield();
        else cpu_relax();
    }

    pthread_mutex_unlock(&p->dispatch_mtx);
}
