use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use MIME::Base64 qw(encode_base64);
use Time::HiRes qw(sleep);

# bin/rocm-inference-server --trellis2 DIR: DELETE /v1/3d/generations/ID cancels a running job: it ends as failed with the error
# "cancelled" at the sampler's next step (or stage); a job that is not running answers 409; another user's job 404s; a job started
# after the cancel runs (the device is free). Needs the TRELLIS.2 files and an image, as t/72 (ROCM_INFERENCE_TEST_TRELLIS2=DIR,
# ROCM_INFERENCE_TEST_TRELLIS2_IMAGE=PNG); the jobs are at 512, a few seconds each before the cancel.
my $dirt = $ENV{ROCM_INFERENCE_TEST_TRELLIS2};
my $img = $ENV{ROCM_INFERENCE_TEST_TRELLIS2_IMAGE};
plan skip_all => 'set ROCM_INFERENCE_TEST_TRELLIS2 (the --trellis2 directory) and ROCM_INFERENCE_TEST_TRELLIS2_IMAGE (a PNG) to run'
    unless $dirt && -d $dirt && $img && -f $img;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/t2c.sock";
my $J = JSON::PP->new->utf8->canonical;

sub http {  # (status, body); the method, the target, the data, then the request header lines
    my ($method, $target, $data, @headers) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n" . join('', map { "$_\r\n" } @headers)
             . "Content-Length: " . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = ($head // '') =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest // '');
}

open my $fh, '<:raw', $img or die "$img: $!";
my $png = do { local $/; <$fh> };
my @alice = ('X-Remote-User: alice');
my @bob = ('X-Remote-User: bob');

my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--trellis2', $dirt, '--listen', $sock, '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server with --trellis2 listens') or BAIL_OUT('no server');

sub start_job {
    my ($user) = @_;
    my ($st, $body) = http(POST => '/v1/3d/generations', { model => 'trellis-2', image => encode_base64($png, ''), resolution => 512,
                                                           seed => 1 }, @$user);
    return ($st, eval { $J->decode($body) } // {});
}
sub wait_stage {  # the job's stage: the first sampler step, or 60 s
    my ($id) = @_;
    my $state = {};
    for (1 .. 120) {
        my ($st, $body) = http(GET => "/v1/3d/generations/$id", undef, @alice);
        $state = eval { $J->decode($body) } // {};
        last if ($state->{progress} // {})->{step} // 0;
        last unless ($state->{status} // '') eq 'in_progress';
        sleep 0.5;
    }
    return $state;
}

my ($st, $job) = start_job(\@alice);
is($st, 200, 'a job starts') or diag(join ' ', %$job);
my $id = $job->{id} // '';
my $before = wait_stage($id);
ok(($before->{progress} // {})->{step}, sprintf('the job reaches a sampler step (%s, step %s)', $before->{progress}{stage} // '-',
    $before->{progress}{step} // '-')) or diag($J->encode($before));

($st, my $body) = http(DELETE => "/v1/3d/generations/$id", undef, @bob);
is($st, 404, 'another user cannot cancel it');
($st, $body) = http(DELETE => "/v1/3d/generations/$id", undef, @alice);
is($st, 200, 'a DELETE cancels the running job') or diag($body);

my $state = {};
for (1 .. 480) {  # the step in progress finishes first: at most a few minutes
    ($st, $body) = http(GET => "/v1/3d/generations/$id", undef, @alice);
    $state = eval { $J->decode($body) } // {};
    last unless ($state->{status} // '') eq 'in_progress';
    sleep 0.5;
}
is($state->{status}, 'failed', 'the job ends as failed') or diag($body);
like($state->{error} // '', qr/cancelled/, 'with the error "cancelled"');
($st, $body) = http(DELETE => "/v1/3d/generations/$id", undef, @alice);
is($st, 409, 'a second cancel: 409 (the job is not running)') or diag($body);

($st, $job) = start_job(\@alice);
is($st, 200, 'a job starts after the cancel: the device is free') or diag(join ' ', %$job);
my $id2 = $job->{id} // '';
wait_stage($id2);
($st, $body) = http(DELETE => "/v1/3d/generations/$id2", undef, @alice);
is($st, 200, 'and it can be cancelled too');
for (1 .. 480) {
    ($st, $body) = http(GET => "/v1/3d/generations/$id2", undef, @alice);
    $state = eval { $J->decode($body) } // {};
    last unless ($state->{status} // '') eq 'in_progress';
    sleep 0.5;
}
like($state->{error} // '', qr/cancelled/, 'the second job ends cancelled too');

kill TERM => $pid;
close $srv;
done_testing;
