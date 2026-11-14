use strict;
use warnings;
use Test::More;

use Web::Route;
use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use POSIX ();
use Time::HiRes qw(sleep);

# a pull whose last piece is a reference to a string: the body ends with it in that round (a third call would send 'X')
my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/s.sock";
my $calls = 0;
route::get '/last' => sub {
    pull(200, [ 'Content-Type' => 'text/plain' ], sub {
        $calls++;
        return 'a' if $calls == 1;
        return \'z' if $calls == 2;
        return 'X';  # never reached: the body ended with the reference
    });
};

my $pid = fork // die "fork: $!";
if (!$pid) {
    open STDERR, '>', "$dir/err" or die;
    run(listen => $sock, timeout => 1);
    POSIX::_exit(0);
}
for (1 .. 200) { last if -S $sock; sleep 0.01 }

my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
print $c "GET /last HTTP/1.1\r\nHost: x\r\n\r\n";
my $all = do { local $/; readline $c };
my ($body) = $all =~ /\r\n\r\n(.*)\z/s;
is $body, 'az', 'the body is the pieces up to the reference, and nothing after it';
kill 'TERM', $pid;  # run's loop stops on TERM
waitpid $pid, 0;

done_testing;
