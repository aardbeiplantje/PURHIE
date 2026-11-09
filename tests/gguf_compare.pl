#!/usr/bin/env perl
# gguf_compare.pl A.gguf B.gguf: the same metadata (keys, types, values; order ignored) and the same tensors (by
# name: type, shape and data bytes; order ignored). Exit 0 when they are, 1 with the differences listed otherwise.
use strict;
use warnings;

use Digest::SHA;
use ROCm::Inference::GGUF qw(gguf_read gguf_read_rows gguf_type_info);

die "usage: $0 A.gguf B.gguf\n" unless @ARGV == 2;
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
my ($same, %types) = (0);
for my $name (sort keys %{{%ta, %tb}}) {
    my ($a, $b) = ($ta{$name}, $tb{$name});
    if (!$a || !$b) { diff("tensor $name only in " . ($a ? 'A' : 'B')); next }
    my ($na) = gguf_type_info($a->{type});
    my ($nb) = gguf_type_info($b->{type});
    if ($a->{type} != $b->{type} || "@{$a->{ne}}" ne "@{$b->{ne}}") {
        diff("tensor $name: $na [@{$a->{ne}}] / $nb [@{$b->{ne}}]");
        next;
    }
    if (digest($a) ne digest($b)) { diff("tensor $name ($na): data differs"); next }
    ++$same;
    ++$types{$na};
}
printf "%d tensors identical (%s), %d metadata keys; %s\n", $same,
    join(', ', map { "$types{$_} $_" } sort keys %types), scalar keys %ka, $bad ? "$bad differences" : 'no differences';
exit($bad ? 1 : 0);
