#!/usr/bin/perl
# glm_chat.pl <model.gguf> [prompt] [max new tokens] [ctx]: one chat turn through the model's own template
# (ROCm::Inference::chat), streamed, with the time to the first token and the tokens a second after it. For a
# first text out of a model (GLM-5.3-Flash: INFERENCE_EXPERT_PIN_MB, INFERENCE_VERBOSE=1 for the pager's counts).
# Run from build/co (the kernels), the modules built:
#   perl -I../../perl/ROCm-Inference/blib/lib -I../../perl/ROCm-Inference/blib/arch \
#        -I../../perl-minijinja-xs/blib/lib -I../../perl-minijinja-xs/blib/arch ../../tools/glm_chat.pl <model>
use strict;
use warnings;
use Time::HiRes qw(time);
use ROCm::Inference qw(:all);

my ($path, $prompt, $max, $ctx) = @ARGV;
die "usage: $0 <model.gguf> [prompt] [max new tokens] [ctx]\n" unless $path;
$prompt //= 'Explain in three short paragraphs why the sky is blue.';
$max //= 200;
$| = 1;
binmode STDOUT, ':encoding(UTF-8)';
my $t0 = time;
my $m = load_model($path) or die "load: " . (ROCm::Inference::get_last_error() // '?') . "\n";
kv_cache_init($m, $ctx // 4096);
printf STDERR "loaded in %.1f s\n", time - $t0;
# GLM_CHAT_TURNS=N: the prompt N times, each a new conversation (the first reads the experts from disk: the
# later ones show the speed once they are in memory)
# GLM_CHAT_VARY=1: a different prompt each turn (other subjects, other experts)
my @vary = ('Write a Perl function that parses an ISO 8601 date and returns epoch seconds, with comments.',
    'Summarize the causes of the French Revolution in five bullet points.',
    'Explain how a B-tree index speeds up a database query, with a small example.',
    'Vertaal naar het Nederlands en leg de grammatica uit: "The weather has been getting colder since last week."',
    'What is the derivative of x^3 * sin(x)? Show the steps.',
    'Give a recipe for a vegetarian lasagna and a shopping list.',
    'Describe how TCP congestion control works, and what changed with BBR.');
for my $turn (1 .. ($ENV{GLM_CHAT_TURNS} // 1)) {
    my ($n, $t1, $tf) = (0, time);
    my $ask = $ENV{GLM_CHAT_VARY} && $turn > 1 ? $vary[($turn - 2) % @vary] : $prompt;
    my ($reply, $why) = chat($m, [{ role => 'user', content => $ask }], {
        max_new_tokens => $max,
        temperature    => 0,
        on_delta       => sub { my ($text, $kind) = @_; $tf //= time; ++$n; print $text if $turn == 1; return 0 },
    });
    my $t2 = time;
    print "\n" if $turn == 1;
    printf STDERR "turn %d stop: %s; first text after %.1f s, then %d pieces in %.1f s (%.2f / s)\n", $turn, $why // '?',
        ($tf // $t2) - $t1, $n, $t2 - ($tf // $t2), $n > 1 ? ($n - 1) / ($t2 - $tf) : 0;
}
