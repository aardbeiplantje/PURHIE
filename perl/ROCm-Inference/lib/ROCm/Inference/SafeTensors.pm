package ROCm::Inference::SafeTensors;

# A Hugging Face checkpoint's safetensors files in plain Perl: the headers (names, dtypes, shapes, offsets) of every
# shard, tensor data read a range of rows at a time. No weights are loaded as a whole.

use strict;
use warnings;

use Carp qw(croak);
use Exporter 'import';
use File::Spec;
use Cwd ();
use JSON::PP ();

our $VERSION = '0.01';
our @EXPORT_OK = qw(st_open st_read_rows st_rows st_row_bytes lfs_path);
# 1: each range read is dropped from the page cache after (ROCm::Inference::drop_cache), for a converter streaming
# a checkpoint once
our $DROP_CACHE = 0;

# Bytes per element of the dtypes a checkpoint holds
my %SIZE = (F64 => 8, F32 => 4, F16 => 2, BF16 => 2, F8_E4M3 => 1, F8_E5M2 => 1, F8_E8M0 => 1, I64 => 8, I32 => 4,
            I16 => 2, I8 => 1, U8 => 1, BOOL => 1);

# A file of a git clone with LFS whose working tree holds pointer files (no checkout: the content is only in
# .git/lfs/objects): the object's path for a pointer, the path itself for any other file. A pointer is
# "version https://git-lfs.github.com/spec/v1", "oid sha256:<hex>", "size <n>"; the object is
# <git dir>/lfs/objects/<hex 0-1>/<hex 2-3>/<hex>, the git dir the nearest .git above the file (a directory, or a
# file with "gitdir: <path>" for a worktree or submodule).
sub lfs_path {
    my ($path) = @_;
    my $size = -s $path;
    return $path unless defined $size && $size < 1024;
    open my $f, '<:raw', $path or return $path;
    local $/;
    my $s = <$f> // '';
    close $f;
    return $path unless $s =~ m{\Aversion https://git-lfs\.github\.com/spec/v1\n}
                     && $s =~ /^oid sha256:([0-9a-f]{64})$/m;
    my $oid = $1;
    my ($want) = $s =~ /^size (\d+)$/m;
    my $dir = File::Spec->rel2abs((File::Spec->splitpath(File::Spec->rel2abs($path)))[1]);
    my $git;
    while (1) {
        my $g = File::Spec->catfile($dir, '.git');
        if (-d $g) { $git = $g; last }
        if (-f $g) {
            open my $h, '<', $g or croak "$g: $!";
            my ($gd) = <$h> =~ /^gitdir:\s*(.+?)\s*$/;
            croak "$g: no gitdir line" unless defined $gd;
            $git = File::Spec->rel2abs($gd, $dir);
            my $common = File::Spec->catfile($git, 'commondir');  # a worktree: the objects are the main repo's
            if (-f $common && open my $c, '<', $common) {
                my $cd = <$c> // '';
                chomp $cd;
                $git = File::Spec->rel2abs($cd, $git) if length $cd;
            }
            last;
        }
        my $up = File::Spec->catdir($dir, File::Spec->updir);
        last if File::Spec->rel2abs($up) eq $dir || $dir eq File::Spec->rootdir;
        $dir = Cwd::abs_path($up) // last;
    }
    croak "$path is a git-lfs pointer, but no .git above it holds the object" unless $git;
    my $obj = File::Spec->catfile($git, 'lfs', 'objects', substr($oid, 0, 2), substr($oid, 2, 2), $oid);
    croak "$path is a git-lfs pointer to $oid, not fetched (git lfs fetch)" unless -f $obj;
    croak "$obj: " . (-s $obj) . " bytes, the pointer says $want" if defined $want && -s $obj != $want;
    return $obj;
}

sub _slurp {
    my ($path) = @_;
    $path = lfs_path($path);
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
        my $path = lfs_path(File::Spec->catfile($dir, $file));
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

=item lfs_path($path)

The file's content path: for a git-lfs pointer file (a clone without a checkout), the object under the nearest
F<.git/lfs/objects> (dies when it was not fetched or its size is not the pointer's); C<$path> otherwise.
st_open and the converter read every file through it.

=item st_rows($tensor), st_row_bytes($tensor)

The number of rows, and the bytes of one.

=back

=cut
