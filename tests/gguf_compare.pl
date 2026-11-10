#!/usr/bin/env perl
# gguf_compare.pl [--ignore-card] [--ulp REGEX] A.gguf B.gguf: the same metadata (keys, types, values; order ignored) and
# the same tensors (by name: type, shape and data bytes; order ignored). Exit 0 when they are, 1 with the differences
# listed otherwise. --ulp: F32 tensors whose name matches may differ by 1 ulp an element (computed with a
# transcendental: torch's exp is MKL's, not correctly rounded; ours is), the count reported.
use strict;
use warnings;

use Digest::SHA;
use ROCm::Inference::GGUF qw(gguf_read gguf_read_rows gguf_type_info);

# --ignore-card: the model-card keys convert_hf_to_gguf.py derives from README.md / the parameter count
# (general.license*, base_model*, tags, size_label, basename, ...) are not compared
my ($ignore_card, $ulp) = (0, undef);
while (@ARGV && $ARGV[0] =~ /^--/) {
    my $o = shift;
    if ($o eq '--ignore-card') { $ignore_card = 1 }
    elsif ($o eq '--ulp') { $ulp = shift }
    else { die "$0: unknown option $o\n" }
}
my $CARD = qr/^general\.(?:license|base_model|tags|size_label|basename|finetune|version|organization|author|url|repo_url|doi|uuid|description|languages|datasets|quantized_by|source)/;
die "usage: $0 [--ignore-card] [--ulp REGEX] A.gguf B.gguf\n" unless @ARGV == 2;
my ($A, $B) = map { gguf_read($_) } @ARGV;
my $bad = 0;
sub diff { print "  $_[0]\n"; ++$bad }

sub kv_str {
    my ($type, $v) = @_;
    return ref $v ? "array<$v->[0]>[" . scalar(@{$v->[1]}) . "] " . Digest::SHA::sha1_hex(join "\0", map { ref $_ ? kv_str($v->[0], $_) : $_ } @{$v->[1]}) : "$type:$v";
}
my %ka = map { $_->[0] => kv_str($_->[1], $_->[2]) } @{$A->{kv}};
my %kb = map { $_->[0] => kv_str($_->[1], $_->[2]) } @{$B->{kv}};
for my $k (sort keys %{{%ka, %kb}}) {
    next if $ignore_card && $k =~ $CARD;
    if (!exists $ka{$k} || !exists $kb{$k}) { diff("metadata $k only in " . (exists $ka{$k} ? 'A' : 'B')) }
    elsif ($ka{$k} ne $kb{$k}) { diff("metadata $k: $ka{$k} / $kb{$k}") }
}

my %ta = map { $_->{name} => $_ } @{$A->{tensors}};
my %tb = map { $_->{name} => $_ } @{$B->{tensors}};
my %fh;
sub digest {
    my ($t) = @_;
    my $f = $fh{$t->{path}} //= do { open my $h, '<:raw', $t->{path} or die "$t->{path}: $!"; $h };
    my $sha = Digest::SHA->new(256);
    my $n = 1;
    $n *= $_ for @{$t->{ne}}[1 .. $#{$t->{ne}}];
    my $chunk = int((256 << 20) / ($t->{bytes} / $n)) || 1;
    for (my $r = 0; $r < $n; $r += $chunk) { $sha->add(gguf_read_rows($f, $t, $r, $r + $chunk > $n ? $n - $r : $chunk)) }
    return $sha->hexdigest;
}
my ($same, %types, %ulp1) = (0);
for my $name (sort keys %{{%ta, %tb}}) {
    my ($a, $b) = ($ta{$name}, $tb{$name});
    if (!$a || !$b) { diff("tensor $name only in " . ($a ? 'A' : 'B')); next }
    my ($na) = gguf_type_info($a->{type});
    my ($nb) = gguf_type_info($b->{type});
    if ($a->{type} != $b->{type} || "@{$a->{ne}}" ne "@{$b->{ne}}") {
        diff("tensor $name: $na [@{$a->{ne}}] / $nb [@{$b->{ne}}]");
        next;
    }
    if (digest($a) ne digest($b)) {
        if (defined $ulp && $name =~ /$ulp/ && $na eq 'f32') {
            my @va = unpack 'l<*', gguf_read_rows($fh{$a->{path}}, $a, 0, $a->{bytes} / 4 / $a->{ne}[0]);
            my @vb = unpack 'l<*', gguf_read_rows($fh{$b->{path}}, $b, 0, $b->{bytes} / 4 / $b->{ne}[0]);
            my $off = grep { $va[$_] != $vb[$_] } 0 .. $#va;
            if (!grep { abs($va[$_] - $vb[$_]) > 1 } 0 .. $#va) { $ulp1{$name} = $off; ++$same; ++$types{$na}; next }
        }
        diff("tensor $name ($na): data differs");
        next;
    }
    ++$same;
    ++$types{$na};
}
do { my $n = 0; $n += $_ for values %ulp1; printf "%d tensors within 1 ulp, %d elements off by 1 (%s)\n", scalar keys %ulp1, $n, join(", ", sort keys %ulp1) } if %ulp1;
printf "%d tensors identical (%s), %d metadata keys; %s\n", $same,
    join(', ', map { "$types{$_} $_" } sort keys %types), scalar keys %ka, $bad ? "$bad differences" : 'no differences';
exit($bad ? 1 : 0);
