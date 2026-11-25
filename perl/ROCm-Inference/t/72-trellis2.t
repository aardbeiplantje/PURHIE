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
# generator), the model's header. Needs the TRELLIS.2 files (ROCM_INFERENCE_TEST_TRELLIS2=DIR, as --trellis2 takes them) and an image
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

sub http {
    my ($method, $target, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
             . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest);
}

my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--trellis2', $dirt,
    '--listen', $sock, '--nolog';
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
($st, $body) = http(POST => '/v1/3d/generations', { model => 'trellis-2', image => encode_base64($png, ''), resolution => 512, seed => 1 });
is($st, 200, 'a job starts') or diag substr($body, 0, 300);
my $job = eval { $J->decode($body) } // {};
is($job->{status}, 'in_progress', 'the job is in progress');
my $id = $job->{id} // '';

($st, $body) = http(POST => '/v1/3d/generations', { image => encode_base64($png, ''), resolution => 512 });
is($st, 429, 'a second job while one runs: 429') or diag $body;
($st, $body) = http(POST => '/v1/images/generations', { prompt => 'a fox' });
is($st, 503, 'the image generator while a job runs: 503') or diag $body;
($st, $body) = http(GET => '/v1/models');
is($st, 200, 'the server answers while a job runs');

my $state = {};
for (1 .. 2400) {  # at most 20 minutes
    ($st, $body) = http(GET => "/v1/3d/generations/$id");
    $state = eval { $J->decode($body) } // {};
    last unless ($state->{status} // '') eq 'in_progress';
    sleep 0.5;
}
is($state->{status}, 'completed', 'the job completes') or diag substr($body, 0, 400);
($st, $body) = http(GET => "/v1/3d/generations/$id/content");
is($st, 200, 'the model downloads');
is(substr($body // '', 0, 4), 'glTF', 'the model is a binary glTF');
is(length($body // ''), $state->{bytes}, 'its length is the one the job gave');
($st, $body) = http(GET => '/v1/3d/generations/no-such-job');
is($st, 404, 'an unknown job: 404');

kill TERM => $pid;
close $srv;
done_testing;
