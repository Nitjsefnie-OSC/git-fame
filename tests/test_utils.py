import pytest

from gitfame import _utils


def test_tighten():
    """Test (grid) table compression"""

    orig_tab = '''
+------------------------+-----+------+------+----------------------+
|   Author               | loc | coms | fils |     distribution     |
+========================+=====+======+======+======================+
|   Casper da Costa-Luis | 719 |   35 |   11 |    93.5/ 100/84.6    |
+------------------------+-----+------+------+----------------------+
|   Not Committed Yet    |  50 |    0 |    2 |     6.5/ 0.0/15.4    |
+------------------------+-----+------+------+----------------------+
'''

    # compress whitespace
    assert (_utils.tighten(orig_tab, max_width=80) == '''\
+----------------------+-----+------+------+----------------+
| Author               | loc | coms | fils |  distribution  |
+======================+=====+======+======+================+
| Casper da Costa-Luis | 719 |   35 |   11 | 93.5/ 100/84.6 |
| Not Committed Yet    |  50 |    0 |    2 |  6.5/ 0.0/15.4 |
+----------------------+-----+------+------+----------------+''')

    # compress first column
    assert (_utils.tighten(orig_tab, max_width=47) == '''\
+--------+-----+------+------+----------------+
| Author | loc | coms | fils |  distribution  |
+========+=====+======+======+================+
| Casper | 719 |   35 |   11 | 93.5/ 100/84.6 |
| Not Com|  50 |    0 |    2 |  6.5/ 0.0/15.4 |
+--------+-----+------+------+----------------+''')

    # too small width - no first column compression
    assert (_utils.tighten(orig_tab, max_width=35) == _utils.tighten(orig_tab))


def test_fext():
    """Test detection of file extensions"""
    assert (_utils.fext('foo/bar.baz') == 'baz')
    assert (_utils.fext('foo/.baz') == 'baz')
    assert (_utils.fext('foo/bar') == '')


def test_Max():
    """Test max with defaults"""
    assert (_utils.Max(range(10), -1) == 9)
    assert (_utils.Max(range(0), -1) == -1)


def test_integer_stats():
    """Test integer representations"""
    assert (_utils.int_cast_or_len(range(10)) == 10)
    assert (_utils.int_cast_or_len('90 foo') == 6)
    assert (_utils.int_cast_or_len('90') == 90)


def test_print():
    """Test printing of unicode"""
    _utils.print_unicode("\x81")


def test_imap_bounded():
    """Test bounded concurrent map: order preserved, concurrency capped"""
    from threading import Lock
    from time import sleep

    live, peak, lock = [0], [0], Lock()

    def work(i):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        # descending durations: item 0 is slowest, so an implementation
        # yielding as-completed (instead of in input order) fails below
        sleep(0.002 * (20-i))
        with lock:
            live[0] -= 1
        return i * 2

    assert list(_utils.imap_bounded(work, range(20), 4)) == [i * 2 for i in range(20)]
    assert peak[0] > 1, "should have run concurrently"
    assert peak[0] <= 4, "should not exceed the job cap"

    # serial fallback: no executor, same results
    assert list(_utils.imap_bounded(lambda i: i * 2, range(5), 1)) == [0, 2, 4, 6, 8]
    assert list(_utils.imap_bounded(lambda i: i * 2, [], 4)) == []


@pytest.mark.parametrize('jobs', [2, 4, 8])
def test_imap_bounded_window_edges(jobs):
    """Test input lengths around the `min(2 * jobs, jobs + 4)` window"""
    window = min(2 * jobs, jobs + 4)
    # fewer than, exactly, and more than the window
    for n in [window - 1, window, window + 1]:
        assert list(_utils.imap_bounded(lambda i: i * 2, range(n), jobs)) == [i * 2 for i in range(n)]

    # The clamp matters for jobs > 4; verify the initial lookahead size.
    consumed = [0]

    def counted_items():
        for i in range(2*jobs + 2):
            consumed[0] += 1
            yield i

    gen = _utils.imap_bounded(lambda i: i * 2, counted_items(), jobs)
    assert next(gen) == 0
    # one extra item is pulled after the first result completes but before yielding
    assert consumed[0] == window + 1
    gen.close()


def test_imap_bounded_abandoned():
    """Test abandoning the generator early doesn't consume the whole input"""
    from threading import Event, Lock

    started, lock, release = [0], Lock(), Event()

    def work(i):
        with lock:
            started[0] += 1
        # only item 0 finishes promptly; the rest block briefly so that
        # queued items can only start if the finally:/cancel is missing
        if i:
            release.wait(0.2)
        return i

    gen = _utils.imap_bounded(work, range(200), 2)
    assert next(gen) == 0
    gen.close()
    release.set()
    # two workers, so at most tasks 0-2 are in flight when the generator is
    # closed; without the cancel, the rest of the submitted window (5 items
    # total) runs too
    assert started[0] <= 3, "should not have run the whole submitted window"


def test_imap_bounded_raises():
    """Test exceptions from `func` propagate"""
    def work(i):
        if i == 3:
            raise ValueError(i)
        return i

    with pytest.raises(ValueError):
        list(_utils.imap_bounded(work, range(200), 4))
    with pytest.raises(ValueError):
        list(_utils.imap_bounded(work, range(200), 1))


SHA_A = "a" * 40
SHA_B = "b" * 40


def _blk(sha, orig, final, n, author=None, ctime="100", extra=()):
    """One `git blame --incremental` chunk block. Commit identity fields are
    emitted only the first time a commit appears, so `author=None` builds the
    bare repeat form: header, optional extras, `filename`."""
    out = [f"{sha} {orig} {final} {n}"]
    if author:
        out += [
            f"author {author}", f"author-mail <{author}@x>", "author-time 1", "author-tz +0000", f"committer {author}",
            f"committer-mail <{author}@x>", f"committer-time {ctime}", "committer-tz +0000", "summary s"]
    out += list(extra)
    out.append("filename f.py")
    return "\n".join(out)


def test_parse_blame_incremental_basic():
    """Metadata is carried across chunks that omit it"""
    out = "\n".join([_blk(SHA_A, 1, 1, 2, "alice"), _blk(SHA_B, 5, 5, 1, "bob"), _blk(SHA_A, 9, 9, 3)]) + "\n"
    assert list(_utils.parse_blame_incremental(out)) == [(2, "alice", "alice@x", "100"), (1, "bob", "bob@x", "100"),
                                                         (3, "alice", "alice@x", "100")]


def test_parse_blame_incremental_coalesces_adjacent():
    """Adjacent same-commit chunks re-join into one porcelain-style group.

    `--incremental` splits these; `--line-porcelain` reports one group, and
    the chunk count is observable as `ctimes` entries.
    """
    out = "\n".join([_blk(SHA_A, 27, 41, 1, "alice"), _blk(SHA_A, 28, 42, 1)]) + "\n"
    assert list(_utils.parse_blame_incremental(out)) == [(2, "alice", "alice@x", "100")]


def test_parse_blame_incremental_no_coalesce_when_gapped():
    """A gap in EITHER the final or the original file keeps chunks separate"""
    gap_final = "\n".join([_blk(SHA_A, 27, 41, 1, "alice"), _blk(SHA_A, 28, 43, 1)]) + "\n"
    assert len(list(_utils.parse_blame_incremental(gap_final))) == 2
    gap_orig = "\n".join([_blk(SHA_A, 27, 41, 1, "alice"), _blk(SHA_A, 30, 42, 1)]) + "\n"
    assert len(list(_utils.parse_blame_incremental(gap_orig))) == 2


def test_parse_blame_incremental_emission_order():
    """Chunks are grouped by commit in the stream, so ordering is by final line"""
    out = "\n".join([_blk(SHA_A, 10, 10, 1, "alice"), _blk(SHA_A, 11, 11, 1), _blk(SHA_B, 1, 1, 1, "bob")]) + "\n"
    assert list(_utils.parse_blame_incremental(out)) == [(1, "bob", "bob@x", "100"), (2, "alice", "alice@x", "100")]


def test_parse_blame_incremental_boundary():
    """`strip_boundary` drops boundary commits, and only when asked"""
    out = "\n".join([_blk(SHA_A, 1, 1, 5, "alice", extra=["boundary"]), _blk(SHA_B, 6, 6, 2, "bob")]) + "\n"
    assert len(list(_utils.parse_blame_incremental(out))) == 2
    assert list(_utils.parse_blame_incremental(out, strip_boundary=True)) == [(2, "bob", "bob@x", "100")]
    # a repeat chunk of a boundary commit carries no `boundary` line of its
    # own, so the flag has to be remembered with the rest of the metadata
    out2 = out + _blk(SHA_A, 20, 20, 3) + "\n"
    assert list(_utils.parse_blame_incremental(out2, strip_boundary=True)) == [(2, "bob", "bob@x", "100")]


def test_parse_blame_incremental_ignores_noise():
    """git-fame folds stderr into stdout, so junk lines must not derail parsing"""
    out = ("fatal: whatever\nwarning: something\n" + _blk(SHA_A, 1, 1, 4, "alice") + "\nnot a header at all\n")
    assert list(_utils.parse_blame_incremental(out)) == [(4, "alice", "alice@x", "100")]


def test_parse_blame_incremental_empty():
    assert list(_utils.parse_blame_incremental("")) == []
