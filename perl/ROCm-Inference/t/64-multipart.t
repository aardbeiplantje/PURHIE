use strict;
use warnings;
use Test::More;

use ROCm::Inference::Multipart;

# the multipart parser: the parts of a body, the file's bytes as sent (no model needed)
{
    my $b = 'XyZ';
    my $body = join '', "--$b\r\n", qq(Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n),
               "--$b\r\n", qq(Content-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n),
               "RIFF\r\n\0binary", "\r\n--$b--\r\n";
    my $p = ROCm::Inference::Multipart::parse($body, qq(multipart/form-data; boundary="$b"));
    is($p->{model}{data}, 'whisper-1', 'a field');
    ok(!exists $p->{model}{filename}, 'a field has no filename');
    is($p->{file}{filename}, 'a.wav', 'a file part keeps its filename');
    is($p->{file}{type}, 'audio/wav', 'and its type');
    is($p->{file}{data}, "RIFF\r\n\0binary", 'the bytes as sent, CRLF and NUL included');
    ok(!eval { ROCm::Inference::Multipart::parse($body, 'application/json'); 1 }, 'not multipart: refused');
}

done_testing;
