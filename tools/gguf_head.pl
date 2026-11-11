#!/usr/bin/env perl
# gguf_head.pl -- the first N blocks of a GGUF as a model of its own: a slice of a model too large to load, for
# checking the engine against its CPU reference on real weights.
#
#   gguf_head.pl <in.gguf> <out.gguf> <N>
#
# Keeps blk.0 .. blk.N-1 and every tensor outside the blocks (embeddings, output norm, LM head), byte for byte;
# sets <arch>.block_count to N and <arch>.nextn_predict_layers to 0 (the MTP block is left out). Per-layer
# metadata arrays stay as they are (longer than N: readers index them by layer). The slice's logits mean nothing
# as a language model; they compare one forward pass with another.

use strict;
use warnings;
use FindBin qw($Bin);
use lib "$Bin/../perl/ROCm-Inference/lib";
use ROCm::Inference::GGUF qw(gguf_read gguf_write_header);

@ARGV == 3 or die "usage: gguf_head.pl <in.gguf> <out.gguf> <N>\n";
my ($src, $dst, $n) = @ARGV;
my $g = gguf_read($src);
my ($arch) = map { $_->[2] } grep { $_->[0] eq 'general.architecture' } @{$g->{kv}};
my $seen = 0;
for my $kv (@{$g->{kv}}) {
    if ($kv->[0] eq "$arch.block_count") {
        die "$src has $kv->[2] blocks\n" if $n < 1 || $n > $kv->[2];
        $kv->[2] = $n;
        $seen = 1;
    }
    $kv->[2] = 0 if $kv->[0] eq "$arch.nextn_predict_layers";
}
die "$src: no $arch.block_count\n" unless $seen;
my @keep = grep { $_->{name} !~ /^blk\.(\d+)\./ || $1 < $n } @{$g->{tensors}};
my @out = map { {%$_} } @keep;
my ($header, $size) = gguf_write_header($g->{kv}, \@out, $g->{alignment});
open my $in, '<:raw', $src or die "$src: $!\n";
open my $fh, '>:raw', $dst or die "$dst: $!\n";
print {$fh} $header or die "$dst: $!\n";
my $base = length $header;
for my $i (0 .. $#keep) {
    my ($s, $d) = ($keep[$i], $out[$i]);
    sysseek($in, $s->{data_offset} + $s->{offset}, 0) or die "$src: seek: $!\n";
    seek($fh, $base + $d->{offset}, 0) or die "$dst: seek: $!\n";
    my $left = $s->{bytes};
    while ($left > 0) {
        my $got = sysread($in, my $buf, $left < (1 << 24) ? $left : 1 << 24);
        die "$src: tensor $s->{name} is cut short\n" unless $got;
        print {$fh} $buf or die "$dst: $!\n";
        $left -= $got;
    }
}
seek($fh, $base + $size - 1, 0) && print({$fh} "\0") if tell($fh) < $base + $size;  # the last tensor's padding
close $fh or die "$dst: $!\n";
printf "%s: %d of %d tensors, %.1f GB\n", $dst, scalar @keep, scalar @{$g->{tensors}}, ($base + $size) / 1e9;
