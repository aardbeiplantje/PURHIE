use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use MIME::Base64 qw(encode_base64);
use Time::HiRes qw(sleep);

# bin/rocm-inference-server --trellis2 DIR: /v1/3d/generations starts a job on an image (TRELLIS.2, an image to a textured 3D
# model); GET /v1/3d/generations/ID polls it, GET .../content gives the binary glTF. Checked: the listing (kind 3d), the errors
# (no image, an image that cannot be decoded), a 512 job, the busy answers while it runs (429 for a second job, 503 for the image
# generator), the model's header, its file in --files and the download as nginx asks for it (X-Accel-Redirect, no body), the file's
# removal at exit; the job is its user's (X-Remote-User): its random id, another user's 404s and listing, the busy answers that
# name it only to its user. Needs the TRELLIS.2 files (ROCM_INFERENCE_TEST_TRELLIS2=DIR, as --trellis2 takes them) and an image
# (ROCM_INFERENCE_TEST_TRELLIS2_IMAGE: a PNG with the object cut out by its alpha channel, or, when DIR holds birefnet.safetensors, a
# photo that BiRefNet cuts out); about four minutes at 512.
my $dirt = $ENV{ROCM_INFERENCE_TEST_TRELLIS2};
my $img = $ENV{ROCM_INFERENCE_TEST_TRELLIS2_IMAGE};
plan skip_all => 'set ROCM_INFERENCE_TEST_TRELLIS2 (the --trellis2 directory) and ROCM_INFERENCE_TEST_TRELLIS2_IMAGE (a PNG) to run'
    unless $dirt && -d $dirt && $img && -f $img;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/t2.sock";
my $J = JSON::PP->new->utf8->canonical;

sub http {  # (status, body, head); extra request header lines after the data
    my ($method, $target, $data, @headers) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n" . join('', map { "$_\r\n" } @headers)
             . "Content-Length: " . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest, $head);
}

my $files = "$dir/files";
mkdir $files or die "$files: $!";
my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--trellis2', $dirt,
    '--listen', $sock, '--files', $files, '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server with --trellis2 listens') or BAIL_OUT('no server');

my ($st, $body) = http(GET => '/v1/models?kind=3d');
my ($m) = eval { @{ $J->decode($body)->{data} } };
is($m && $m->{id}, 'trellis-2', 'the 3D model is listed') or diag $body;
is($m && $m->{kind}, '3d', 'its kind is 3d');
is_deeply($m && $m->{endpoints}, ['/v1/3d/generations'], 'its endpoint');
ok($m && $m->{description} && $m->{usage} && $m->{parameters}{image}, 'its purpose and usage');

($st, $body) = http(POST => '/v1/3d/generations', { model => 'trellis-2' });
is($st, 400, 'no image: refused') or diag $body;
($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64('not an image', '') });
is($st, 400, 'an image that cannot be decoded: refused') or diag $body;
($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64('x', ''), resolution => 768 });
is($st, 400, 'a resolution other than 512 or 1024: refused') or diag $body;

open my $fh, '<:raw', $img or die "$img: $!";
my $png = do { local $/; <$fh> };
# the job is alice's (nginx's X-Remote-User); bob is another user
my @alice = ('X-Remote-User: alice');
my @bob = ('X-Remote-User: bob');
($st, $body) = http(POST => '/v1/3d/generations', { model => 'trellis-2', image => encode_base64($png, ''), resolution => 512, seed => 1 },
                    @alice);
is($st, 200, 'a job starts') or diag substr($body, 0, 300);
my $job = eval { $J->decode($body) } // {};
is($job->{status}, 'in_progress', 'the job is in progress');
my $id = $job->{id} // '';
like($id, qr/\A3d_[0-9a-f]{32}\z/, 'its id is random (128 bits)');

($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64($png, ''), resolution => 512 }, @alice);
is($st, 429, 'a second job while one runs: 429') or diag $body;
like($body, qr/\Q$id\E/, 'which names the running job to its user');
($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64($png, ''), resolution => 512 }, @bob);
is($st, 429, 'another user\'s job: 429 too');
unlike($body, qr/\Q$id\E/, 'which does not name the running job');
($st, $body) = http(POST => '/v1/images/generations', { prompt => 'a fox' }, @bob);
is($st, 503, 'the image generator while a job runs: 503') or diag $body;
unlike($body, qr/\Q$id\E/, 'without the running job\'s id for another user');
($st, $body) = http(GET => '/v1/models');
is($st, 200, 'the server answers while a job runs');
($st, $body) = http(GET => "/v1/3d/generations/$id", undef, @bob);
is($st, 404, 'another user does not see the job');
($st, $body) = http(GET => '/v1/3d/generations', undef, @bob);
is_deeply((eval { $J->decode($body) } // {})->{data}, [], 'nor list it');
($st, $body) = http(GET => '/v1/3d/generations');
is_deeply((eval { $J->decode($body) } // {})->{data}, [], 'nor does a request with no user');
($st, $body) = http(GET => '/v1/3d/generations', undef, @alice);
is(scalar @{ (eval { $J->decode($body) } // {})->{data} // [] }, 1, 'its user lists it');

my $state = {};
for (1 .. 2400) {  # at most 20 minutes
    ($st, $body) = http(GET => "/v1/3d/generations/$id", undef, @alice);
    $state = eval { $J->decode($body) } // {};
    last unless ($state->{status} // '') eq 'in_progress';
    sleep 0.5;
}
is($state->{status}, 'completed', 'the job completes') or diag substr($body, 0, 400);
($st, $body) = http(GET => "/v1/3d/generations/$id/content", undef, @bob);
is($st, 404, 'another user cannot download it');
($st, $body) = http(GET => "/v1/3d/generations/$id/content", undef, @alice);
is($st, 200, 'the model downloads');
is(substr($body // '', 0, 4), 'glTF', 'the model is a binary glTF');
is(length($body // ''), $state->{bytes}, 'its length is the one the job gave');
my $glb = $body // '';
my $file = "$files/rocm-$pid-$id.glb";
is(-s $file, $state->{bytes}, 'the model is a file in --files, of that length');
my $head;
($st, $body, $head) = http(GET => "/v1/3d/generations/$id/content", undef, @alice, 'X-Sendfile-Type: X-Accel-Redirect',
                           "X-Accel-Mapping: $files/=/rocm-files/");
is($st, 200, 'the download as nginx asks for it');
my ($xa) = ($head // '') =~ /^X-Accel-Redirect: (\S+)/mi;
is($xa, "/rocm-files/rocm-$pid-$id.glb", 'is the X-Accel-Redirect to the file');
is($body, '', 'with no body');
like($head // '', qr{^Content-Type: model/gltf-binary}mi, 'and the model\'s Content-Type');
($st, $body) = http(GET => '/v1/3d/generations/no-such-job');
is($st, 404, 'an unknown job: 404');

# the mesh alone as an STL (for 3D printing): no texture, its own content type, 84 bytes and 50 a face; ROCM_INFERENCE_TEST_TRELLIS2_OUT
# (a directory) keeps both models for a look
my $out = $ENV{ROCM_INFERENCE_TEST_TRELLIS2_OUT};
if ($out && -d $out) { open my $o, '>:raw', "$out/72-trellis2.glb" or die "$out: $!"; print $o $glb; close $o }
($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64($png, ''), resolution => 512, seed => 1, format => 'stl',
                                                    faces => 200000, size_mm => 80 }, @alice);
is($st, 200, 'an STL job starts') or diag substr($body, 0, 300);
my $sjob = eval { $J->decode($body) } // {};
is($sjob->{options}{format}, 'stl', 'its format is stl');
ok(!$sjob->{options}{texture}, 'and it has no texture');
for (1 .. 2400) {
    ($st, $body) = http(GET => "/v1/3d/generations/$sjob->{id}", undef, @alice);
    $state = eval { $J->decode($body) } // {};
    last unless ($state->{status} // '') eq 'in_progress';
    sleep 0.5;
}
is($state->{status}, 'completed', 'the STL job completes') or diag substr($body, 0, 400);
my $shead;
($st, $body, $shead) = http(GET => "/v1/3d/generations/$sjob->{id}/content", undef, @alice);
is($st, 200, 'the STL downloads');
like($shead // '', qr{^Content-Type: model/stl}mi, 'as model/stl');
like($shead // '', qr{filename="\Q$sjob->{id}\E\.stl"}m, 'named .stl');
my $nf = length($body // '') >= 84 ? unpack('V', substr($body, 80, 4)) : -1;
ok($nf > 1000 && $nf <= 200000, "its faces: $nf, at most the 200000 asked");
is(length($body // ''), 84 + 50 * $nf, 'a binary STL: 84 bytes, then 50 a face');
ok(-s "$files/rocm-$pid-$sjob->{id}.stl", 'the STL is a file in --files');
if ($out && -d $out) { open my $o, '>:raw', "$out/72-trellis2.stl" or die "$out: $!"; print $o $body; close $o }
my ($closed, $vol, $edges) = stl_check($body // '');
is($closed, $edges, "a closed solid: each of its $edges edges has two faces, wound opposite ways");
ok($vol > 0, "wound outward (volume $vol mm3)");

# a hollow STL: walls of 2 mm, a drain hole of 3; only for an STL
($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64($png, ''), format => 'glb', hollow_mm => 2 }, @alice);
is($st, 400, 'a hollow GLB: 400');
like($body // '', qr/hollow_mm is for format stl/, 'with the reason');
($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64($png, ''), resolution => 512, seed => 1, format => 'stl',
                                                    faces => 200000, size_mm => 80, hollow_mm => 2, drain_mm => 3 }, @alice);
is($st, 200, 'a hollow STL job starts') or diag substr($body, 0, 300);
my $hjob = eval { $J->decode($body) } // {};
is_deeply([@{$hjob->{options} // {}}{qw(hollow_mm drain_mm)}], [2, 3], 'its walls and drain');
for (1 .. 2400) {
    ($st, $body) = http(GET => "/v1/3d/generations/$hjob->{id}", undef, @alice);
    $state = eval { $J->decode($body) } // {};
    last unless ($state->{status} // '') eq 'in_progress';
    sleep 0.5;
}
is($state->{status}, 'completed', 'the hollow STL job completes') or diag substr($body, 0, 400);
($st, $body) = http(GET => "/v1/3d/generations/$hjob->{id}/content", undef, @alice);
my ($hclosed, $hvol, $hedges) = stl_check($body // '');
is($hclosed, $hedges, "the hollow is closed too ($hedges edges)");
ok($hvol > 0 && $hvol < 0.6 * $vol, "and holds less material: $hvol mm3 against $vol");
if ($out && -d $out) { open my $o, '>:raw', "$out/72-trellis2-hollow.stl" or die "$out: $!"; print $o $body; close $o }

kill TERM => $pid;
close $srv;
ok(!-e $file, 'the server removes its file at exit');
done_testing;

# a binary STL's edges, its vertices welded by their bytes: (the edges with two faces wound opposite ways, the signed volume, the
# edges)
sub stl_check {
    my ($stl) = @_;
    return (0, 0, -1) if length($stl) < 84;
    my $n = unpack('V', substr($stl, 80, 4));
    my (%id, %dir, $vol);
    for my $t (0 .. $n - 1) {
        my $rec = substr($stl, 84 + 50 * $t + 12, 36);
        my @v = map { substr($rec, 12 * $_, 12) } 0 .. 2;
        my @i = map { $id{$_} //= scalar keys %id } @v;
        my @p = map { [unpack('f<3', $_)] } @v;
        $vol += ($p[0][0] * ($p[1][1] * $p[2][2] - $p[1][2] * $p[2][1]) + $p[0][1] * ($p[1][2] * $p[2][0] - $p[1][0] * $p[2][2])
                 + $p[0][2] * ($p[1][0] * $p[2][1] - $p[1][1] * $p[2][0])) / 6;
        for my $k (0 .. 2) {
            my ($a, $b) = ($i[$k], $i[($k + 1) % 3]);
            my $key = $a < $b ? "$a,$b" : "$b,$a";
            push @{$dir{$key}}, $a < $b ? 1 : -1;
        }
    }
    my $good = grep { @$_ == 2 && $_->[0] != $_->[1] } values %dir;
    return ($good, sprintf('%.0f', $vol), scalar keys %dir);
}
