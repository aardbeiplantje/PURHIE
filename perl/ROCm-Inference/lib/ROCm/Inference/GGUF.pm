package ROCm::Inference::GGUF;

# GGUF files in plain Perl: the header (metadata and tensor table) read and written, tensor data read in row ranges.
# No model is loaded and no GPU is touched; bin/rocm-quantize builds on it.

use strict;
use warnings;

use Carp qw(croak);
use Exporter 'import';

our $VERSION = '0.01';
our @EXPORT_OK = qw(gguf_read gguf_write_header gguf_type_info gguf_type_id gguf_tensor_bytes gguf_read_rows);

# Metadata value types (gguf.h)
use constant {
    T_U8 => 0, T_I8 => 1, T_U16 => 2, T_I16 => 3, T_U32 => 4, T_I32 => 5, T_F32 => 6, T_BOOL => 7,
    T_STRING => 8, T_ARRAY => 9, T_U64 => 10, T_I64 => 11, T_F64 => 12,
};
my %PACK = (0 => 'C', 1 => 'c', 2 => 'v', 3 => 's<', 4 => 'V', 5 => 'l<', 6 => 'f<', 7 => 'C', 10 => 'Q<',
            11 => 'q<', 12 => 'd<');
my %WIDTH = (0 => 1, 1 => 1, 2 => 2, 3 => 2, 4 => 4, 5 => 4, 6 => 4, 7 => 1, 10 => 8, 11 => 8, 12 => 8);

# ggml tensor types: id => [name, values per block, bytes per block] (ggml.c's type_traits; 100+ are ROCmFPX's)
my %TYPE = (
    0 => ['f32', 1, 4], 1 => ['f16', 1, 2], 2 => ['q4_0', 32, 18], 3 => ['q4_1', 32, 20], 6 => ['q5_0', 32, 22],
    7 => ['q5_1', 32, 24], 8 => ['q8_0', 32, 34], 9 => ['q8_1', 32, 36], 10 => ['q2_K', 256, 84],
    11 => ['q3_K', 256, 110], 12 => ['q4_K', 256, 144], 13 => ['q5_K', 256, 176], 14 => ['q6_K', 256, 210],
    15 => ['q8_K', 256, 292], 16 => ['iq2_xxs', 256, 66], 17 => ['iq2_xs', 256, 74], 18 => ['iq3_xxs', 256, 98],
    19 => ['iq1_s', 256, 50], 20 => ['iq4_nl', 32, 18], 21 => ['iq3_s', 256, 110], 22 => ['iq2_s', 256, 82],
    23 => ['iq4_xs', 256, 136], 24 => ['i8', 1, 1], 25 => ['i16', 1, 2], 26 => ['i32', 1, 4], 27 => ['i64', 1, 8],
    28 => ['f64', 1, 8], 29 => ['iq1_m', 256, 56], 30 => ['bf16', 1, 2], 34 => ['tq1_0', 256, 54],
    35 => ['tq2_0', 256, 66], 39 => ['mxfp4', 32, 17], 40 => ['nvfp4', 64, 36],
    100 => ['q4_0_rocmfp4', 32, 18], 101 => ['q4_0_rocmfp4_fast', 32, 17],
);
my %TYPE_ID = map { lc($TYPE{$_}[0]) => $_ } keys %TYPE;  # names in any case

sub gguf_type_info { my $i = $TYPE{$_[0]} or return; return @$i }
sub gguf_type_id   { return $TYPE_ID{lc $_[0]} }

# Bytes of a tensor: rows of ne[0] values in blocks
sub gguf_tensor_bytes {
    my ($t) = @_;
    my (undef, $blk, $size) = gguf_type_info($t->{type}) or croak "tensor $t->{name}: unknown type $t->{type}";
    my $n = 1;
    $n *= $_ for @{$t->{ne}};
    croak "tensor $t->{name}: $t->{ne}[0] values are not whole blocks of $blk" if $t->{ne}[0] % $blk;
    return $n / $blk * $size;
}

# A reader over a file: the header is parsed from a buffer refilled in 16 MB reads
sub _reader {
    my ($fh, $path) = @_;
    my $buf = '';
    my $pos = 0;  # of $buf in the file
    my $off = 0;  # read position in $buf
    my $need = sub {
        my ($n) = @_;
        while (length($buf) - $off < $n) {
            $buf = substr($buf, $off);
            $pos += $off;
            $off = 0;
            my $got = sysread($fh, $buf, 16 << 20, length $buf);
            croak "$path: $!" unless defined $got;
            croak "$path: truncated GGUF header" unless $got;
        }
    };
    my $take = sub {
        my ($n) = @_;
        $need->($n);
        my $s = substr($buf, $off, $n);
        $off += $n;
        return $s;
    };
    return ($take, sub { $pos + $off });
}

sub _value {
    my ($take, $type) = @_;
    if ($type == T_STRING) {
        my $n = unpack 'Q<', $take->(8);
        return $take->($n);
    }
    if ($type == T_ARRAY) {
        my ($et, $n) = unpack 'VQ<', $take->(12);
        if ($et == T_STRING || $et == T_ARRAY) {
            return [$et, [map { _value($take, $et) } 1 .. $n]];
        }
        my $w = $WIDTH{$et} // croak "GGUF: array of unknown type $et";
        return [$et, [unpack "$PACK{$et}$n", $take->($w * $n)]] if $n;
        return [$et, []];
    }
    my $w = $WIDTH{$type} // croak "GGUF: metadata of unknown type $type";
    return unpack $PACK{$type}, $take->($w);
}

# gguf_read($path): {path, version, alignment, kv => [[key, type, value]...], kv_index => {key => i},
# tensors => [{name, ne => [...], type, offset (in the data section), bytes}...], data_offset}. An array value is
# [element type, [values]]. A split model's first file (-00001-of-0000N.gguf) reads every file: the tensors of all
# (each with its file in {path}), the metadata of the first.
sub gguf_read {
    my ($path) = @_;
    my @files = ($path);
    if ($path =~ /^(.*)-00001-of-(\d{5})\.gguf$/) {
        my ($stem, $n) = ($1, $2);
        @files = map { sprintf '%s-%05d-of-%s.gguf', $stem, $_, $n } 1 .. $n;
    }
    my $first;
    for my $file (@files) {
        my $g = _read_one($file);
        if ($first) { push @{$first->{tensors}}, @{$g->{tensors}} } else { $first = $g }
    }
    $first->{files} = \@files;
    return $first;
}

sub _read_one {
    my ($path) = @_;
    open my $fh, '<:raw', $path or croak "$path: $!";
    my ($take, $tell) = _reader($fh, $path);
    croak "$path: not a GGUF file" unless $take->(4) eq 'GGUF';
    my ($version, $nt, $nkv) = unpack 'VQ<Q<', $take->(20);
    croak "$path: GGUF version $version (2 or 3 is read)" unless $version == 2 || $version == 3;
    my %g = (path => $path, version => $version, alignment => 32, kv => [], kv_index => {}, tensors => []);
    for (1 .. $nkv) {
        my $key = _value($take, T_STRING);
        my $type = unpack 'V', $take->(4);
        my $v = _value($take, $type);
        $g{kv_index}{$key} = scalar @{$g{kv}};
        push @{$g{kv}}, [$key, $type, $v];
        $g{alignment} = $v if $key eq 'general.alignment';
    }
    for (1 .. $nt) {
        my $name = _value($take, T_STRING);
        my $nd = unpack 'V', $take->(4);
        my @ne = unpack "Q<$nd", $take->(8 * $nd);
        my ($type, $offset) = unpack 'VQ<', $take->(12);
        my $t = {name => $name, ne => \@ne, type => $type, offset => $offset, path => $path};
        $t->{bytes} = gguf_tensor_bytes($t);
        push @{$g{tensors}}, $t;
    }
    my $a = $g{alignment};
    $g{data_offset} = int(($tell->() + $a - 1) / $a) * $a;
    $_->{data_offset} = $g{data_offset} for @{$g{tensors}};
    close $fh;
    return \%g;
}

# Rows [r0, r0 + n) of tensor $t as a byte string, from an open handle of its file ($t->{path})
sub gguf_read_rows {
    my ($fh, $t, $r0, $n) = @_;
    my (undef, $blk, $size) = gguf_type_info($t->{type});
    my $row = $t->{ne}[0] / $blk * $size;
    my $at = $t->{data_offset} + $t->{offset} + $r0 * $row;
    sysseek($fh, $at, 0) or croak "$t->{path}: seek: $!";
    my $want = $n * $row;
    my $buf = '';
    while (length($buf) < $want) {
        my $got = sysread($fh, $buf, $want - length $buf, length $buf);
        croak "$t->{path}: $!" unless defined $got;
        croak "$t->{path}: tensor $t->{name} is cut short" unless $got;
    }
    return $buf;
}

sub _enc_value {
    my ($type, $v) = @_;
    return pack('Q<', length $v) . $v if $type == T_STRING;
    if ($type == T_ARRAY) {
        my ($et, $vals) = @$v;
        my $s = pack 'VQ<', $et, scalar @$vals;
        if ($et == T_STRING || $et == T_ARRAY) {
            $s .= _enc_value($et, $_) for @$vals;
            return $s;
        }
        return $s . pack "$PACK{$et}*", @$vals;
    }
    my $p = $PACK{$type} // croak "GGUF: metadata of unknown type $type";
    return pack $p, $v;
}

# The header of a GGUF v3 file: metadata [[key, type, value]...] and tensors [{name, ne, type, bytes}...], their
# offsets assigned here (each tensor starts on a multiple of $alignment). Returns (header bytes padded to the
# alignment, data section size); each tensor gets {offset}.
sub gguf_write_header {
    my ($kv, $tensors, $alignment) = @_;
    $alignment //= 32;
    my $h = 'GGUF' . pack 'VQ<Q<', 3, scalar @$tensors, scalar @$kv;
    $h .= _enc_value(T_STRING, $_->[0]) . pack('V', $_->[1]) . _enc_value($_->[1], $_->[2]) for @$kv;
    my $off = 0;
    for my $t (@$tensors) {
        $t->{offset} = $off;
        $h .= _enc_value(T_STRING, $t->{name}) . pack('V', scalar @{$t->{ne}}) . pack('Q<*', @{$t->{ne}}) .
            pack('VQ<', $t->{type}, $off);
        $off += int(($t->{bytes} + $alignment - 1) / $alignment) * $alignment;
    }
    $h .= "\0" x ((-length $h) % $alignment);
    return ($h, $off);
}

1;

__END__

=head1 NAME

ROCm::Inference::GGUF - read and write GGUF headers in plain Perl

=head1 SYNOPSIS

  use ROCm::Inference::GGUF qw(gguf_read gguf_read_rows gguf_write_header);
  my $g = gguf_read('model.gguf');
  for my $t (@{$g->{tensors}}) { printf "%s %s\n", $t->{name}, join 'x', @{$t->{ne}} }

=head1 DESCRIPTION

The GGUF container (versions 2 and 3) without loading a model: metadata and the tensor table are parsed from the
file's header, tensor data is read a range of rows at a time. A split model is read through its first file.

=over

=item gguf_read($path)

The header as a hash: C<version>, C<alignment>, C<kv> (C<[key, type, value]> in file order; an array value is
C<[element type, [values]]>), C<kv_index> (key to its position in C<kv>), C<tensors> (C<name>, C<ne>, C<type>,
C<offset>, C<bytes>, C<path>, C<data_offset>), C<files>.

=item gguf_read_rows($fh, $tensor, $first_row, $n_rows)

Those rows of a tensor as a byte string, from a handle open on C<< $tensor->{path} >>.

=item gguf_write_header(\@kv, \@tensors, $alignment)

The bytes of a GGUF v3 header for that metadata and those tensors (each with C<name>, C<ne>, C<type>, C<bytes>),
padded to the alignment, and the size of the data section that follows it. Sets each tensor's C<offset>; each
tensor's data is padded with zero bytes to a multiple of the alignment.

=item gguf_type_info($id)

C<(name, values per block, bytes per block)> of a ggml tensor type, or the empty list.

=item gguf_type_id($name)

The id of a ggml tensor type by name (C<q8_0>, C<bf16>, C<q4_0_rocmfp4_fast>, ...).

=item gguf_tensor_bytes($tensor)

Bytes of a tensor's data, from its C<type> and C<ne>.

=back

=cut
