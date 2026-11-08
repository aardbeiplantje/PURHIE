use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);

use ROCm::Inference qw(:all);
use ROCm::Inference::Store qw(store_evict store_files store_usage);

# LRU eviction keeps what a conversation still uses, also while it continues on the sequence (no store lookup
# needed): as opencode does, a main agent runs many requests, a subagent interjects on the same sequence, then the
# main agent resumes and needs its prefix back from the store. Needs a GPU and a model:
# ROCM_INFERENCE_TEST_MODEL=<gguf> (a recurrent one, such as Qwen3.5-4B, also checks the state snapshots)
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;

my $m = load_model($path);
set_swa_ring($m, 0);
kv_cache_init($m, 8192);
my $dir;

# both snapshot modes: eager (a prompt end written at every request, the previous one of the conversation deleted)
# and lazy (held in memory, written when the subagent takes the sequence)
for my $lazy (0, 1) {
    note($lazy ? 'lazy snapshots' : 'eager snapshots');
    reset($m);
    $dir = tempdir('rocm-inference-lru-XXXXXX', DIR => (-d '/dev/shm' ? '/dev/shm' : undef), CLEANUP => 1);
    store_open($m, $dir);
    store_set_lazy($m, $lazy);
    my $B = store_stats($m)->{block_tokens};

    # time passes between requests: every file so far gets 10 s older (mtime has 1 s resolution)
    sub age { for my $f (store_files($dir)) { utime $f->{mtime} - 10, $f->{mtime} - 10, $f->{path} } }
    sub ask {
        my ($text) = @_;
        my $ids = tokenize($m, $text);
        generate($m, $ids, { max_new_tokens => 2, temperature => 0, reuse_prefix => 1 });
        age();
        return ($ids, last_prompt_stats($m));
    }

    my $main = join ' ', map { "Line $_ of the main agent's long system prompt and tools." } 1 .. 150;
    my ($ids, $reused);
    for my $turn (1 .. 6) {  # the main agent: each request appends to the last (reused from the sequence)
        $main .= join ' ', "\nTurn $turn:", map { "the tool said item $_ is fine." } 1 .. 40;
        ($ids, $reused) = ask($main);
    }
    ok($reused > 0, "the main agent's turns reuse the sequence ($reused tokens on the last)");
    my $main_len = @$ids;
    my %before = map { $_->{name} => 1 } store_files($dir);
    my $rsts = grep { /\.rst\z/ } keys %before;
    my $grid = int(int($main_len / $B) / 4);  # store_open's default spacing: every 4th block at this depth
    my $recurrent = store_stats($m)->{snapshot_bytes} > 0;
    SKIP: {
        skip 'not a recurrent model: no snapshots', 2 unless $recurrent;
        if ($lazy) {
            ok($rsts <= $grid, "lazy: on disk only the spacing's $rsts snapshots (at most $grid), no prompt end");
            ok(store_stats($m)->{snapshots_held} > 0, store_stats($m)->{snapshots_held} . ' prompt ends held in memory instead');
        } else {
            ok($rsts <= $grid + 1, "the main agent keeps $rsts snapshots: the spacing's $grid and its latest prompt end");
            ok(store_stats($m)->{snapshots_dropped} > 0, store_stats($m)->{snapshots_dropped} . ' superseded prompt-end snapshots deleted');
        }
    }
    my $saved0 = store_stats($m)->{snapshots_saved};

    my $sub = join ' ', map { "Line $_ of the subagent's own prompt." } 1 .. 120;
    for my $turn (1 .. 3) {  # a subagent on the same sequence: a different prefix from the first token
        $sub .= join ' ', "\nStep $turn:", map { "found file $_." } 1 .. 30;
        ask($sub);
        if ($turn == 1 && $lazy && $recurrent) {
            my ($end) = grep { !$before{ $_->{name} } && $_->{name} =~ /\.rst\z/ } store_files($dir);
            ok(store_stats($m)->{snapshots_saved} > $saved0 && $end,
               'lazy: the main agent\'s held prompt end is written when the subagent takes the sequence');
        }
    }

    # the store over its limit: room for the subagent's files and the main conversation's chain with two snapshots,
    # not for the main agent's older snapshots (one per request: where each prompt ended)
    my $kvb = store_stats($m)->{block_bytes};
    my ($rst) = map { $_->{bytes} } grep { $_->{name} =~ /\.rst\z/ } store_files($dir);
    my $subs = 0;
    $subs += $_->{bytes} for grep { !$before{ $_->{name} } } store_files($dir);
    my $need = $subs + int($main_len / $B) * $kvb + 2 * ($rst // 0);
    my ($bytes) = store_usage($dir);
    ok($bytes > $need, sprintf('the store (%.1f MB) holds more than the main chain needs (%.1f MB)', $bytes / 2**20, $need / 2**20));
    my @gone = store_evict($dir, max_bytes => $need);
    ok(@gone > 0 || $lazy, scalar(@gone) . ' files evicted');

    # the main agent resumes: its prefix comes back from the store
    $main .= "\nTurn 7: the user asks for a summary.";
    ($ids, $reused) = ask($main);
    my $want = int(($main_len - 1) / $B) * $B - 4 * $B;  # within the snapshot spacing of where it stopped
    ok($reused >= $want, "the main agent resumes from the store: $reused of " . scalar(@$ids) . " tokens reused (at least $want)");
    store_close($m);
}

done_testing;
