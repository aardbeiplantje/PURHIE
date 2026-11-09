package ROCm::Inference::SafeTensors;

# A Hugging Face checkpoint's safetensors files in plain Perl: the headers (names, dtypes, shapes, offsets) of every
# shard, tensor data read a range of rows at a time. No weights are loaded as a whole.

use strict;
use warnings;

use Carp qw(croak);
use Exporter 'import';
use File::Spec;
use JSON::PP ();

our $VERSION = '0.01';
our @EXPORT_OK = qw(st_open st_read_rows st_rows st_row_bytes);
# 1: each range read is dropped from the page cache after (ROCm::Inference::drop_cache), for a converter streaming
# a checkpoint once
our $DROP_CACHE = 0;

# Bytes per element of the dtypes a checkpoint holds
my %SIZE = (F64 => 8, F32 => 4, F16 => 2, BF16 => 2, F8_E4M3 => 1, F8_E5M2 => 1, F8_E8M0 => 1, I64 => 8, I32 => 4,
            I16 => 2, I8 => 1, U8 => 1, BOOL => 1);

sub _slurp {
    my ($path) = @_;
    open my $f, '<:raw', $path or croak "$path: $!";
    local $/;
    my $s = <$f>;
    close $f;
    return $s;
}

# st_open($dir): {dir, tensors => {name => {name, dtype, shape => [...], file, start, bytes}}, order => [names in
# file order]}. The shards are model.safetensors.index.json's, or the one model.safetensors.
sub st_open {
    my ($dir) = @_;
    my @files;
    my $index = File::Spec->catfile($dir, 'model.safetensors.index.json');
    if (-f $index) {
        my $map = JSON::PP->new->decode(_slurp($index))->{weight_map} or croak "$index: no weight_map";
        my %seen;
        @files = sort grep { !$seen{$_}++ } values %$map;
    } elsif (-f File::Spec->catfile($dir, 'model.safetensors')) {
        @files = ('model.safetensors');
    } else {
        croak "$dir: no model.safetensors(.index.json)";
    }
    my %st = (dir => $dir, tensors => {}, order => []);
    for my $file (@files) {
        my $path = File::Spec->catfile($dir, $file);
        open my $f, '<:raw', $path or croak "$path: $!";
        read($f, my $n8, 8) == 8 or croak "$path: not a safetensors file";
        my $n = unpack 'Q<', $n8;
        read($f, my $hj, $n) == $n or croak "$path: truncated header";
        close $f;
        my $h = JSON::PP->new->decode($hj);
        delete $h->{__metadata__};
        my @names = sort { $h->{$a}{data_offsets}[0] <=> $h->{$b}{data_offsets}[0] } keys %$h;
        for my $name (@names) {
            my $e = $h->{$name};
            my $size = $SIZE{$e->{dtype}} // croak "$path: $name: dtype $e->{dtype}";
            my ($a, $b) = @{$e->{data_offsets}};
            my $t = {name => $name, dtype => $e->{dtype}, shape => $e->{shape}, file => $path, start => 8 + $n + $a,
                     bytes => $b - $a, esize => $size};
            my $want = $size;
            $want *= $_ for @{$e->{shape}};
            croak "$path: $name: $t->{bytes} bytes for shape [@{$e->{shape}}] of $e->{dtype}" unless $want == $t->{bytes};
            croak "$path: $name appears twice" if $st{tensors}{$name};
            $st{tensors}{$name} = $t;
            push @{$st{order}}, $name;
        }
    }
    return \%st;
}

# Rows of a tensor: the product of all dimensions but the last (a 1-D tensor is one row)
sub st_rows {
    my ($t) = @_;
    my @s = @{$t->{shape}};
    my $n = 1;
    $n *= $_ for @s[0 .. $#s - 1];
    return $n;
}

sub st_row_bytes {
    my ($t) = @_;
    my @s = @{$t->{shape}};
    return (@s ? $s[-1] : 1) * $t->{esize};
}

# Rows [r0, r0 + n) of a tensor as a byte string; $fh is a cache {path => handle}
sub st_read_rows {
    my ($fh, $t, $r0, $n) = @_;
    my $f = $fh->{$t->{file}} //= do { open my $h, '<:raw', $t->{file} or croak "$t->{file}: $!"; $h };
    my $rb = st_row_bytes($t);
    sysseek($f, $t->{start} + $r0 * $rb, 0) or croak "$t->{file}: seek: $!";
    my $want = $n * $rb;
    my $buf = '';
    while (length($buf) < $want) {
        my $got = sysread($f, $buf, $want - length $buf, length $buf);
        croak "$t->{file}: $!" unless defined $got;
        croak "$t->{file}: $t->{name} is cut short" unless $got;
    }
    ROCm::Inference::drop_cache(fileno($f), $t->{start} + $r0 * $rb, $want) if $DROP_CACHE;
    return $buf;
}

1;

__END__

=head1 NAME

ROCm::Inference::SafeTensors - read a Hugging Face checkpoint's safetensors shards in plain Perl

=head1 SYNOPSIS

  use ROCm::Inference::SafeTensors qw(st_open st_read_rows);
  my $st = st_open('/data/hf/GLM-5.3-Flash');
  my $t = $st->{tensors}{'lm_head.weight'};
  my %fh;
  my $rows = st_read_rows(\%fh, $t, 0, 16);

=head1 DESCRIPTION

=over

=item st_open($dir)

The headers of every shard (F<model.safetensors.index.json>'s, or F<model.safetensors>): C<tensors> (name to
C<name>, C<dtype>, C<shape>, C<file>, C<start>, C<bytes>, C<esize>) and C<order> (the names in file order).

=item st_read_rows(\%handles, $tensor, $first_row, $n_rows)

Those rows (all dimensions but the last flattened) as a byte string.

=item st_rows($tensor), st_row_bytes($tensor)

The number of rows, and the bytes of one.

=back

=cut
