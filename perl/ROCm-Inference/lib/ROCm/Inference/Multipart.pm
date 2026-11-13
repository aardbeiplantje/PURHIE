package ROCm::Inference::Multipart;

# multipart/form-data (RFC 7578, what a browser or OpenAI's client sends for /v1/audio/transcriptions): the parts of a
# request body by their field names. Each part is { name, filename, type, data } with the bytes as they were sent.
use strict;
use warnings;
use Carp qw(croak);

# parse($body, $content_type) -> { name => { name, filename, type, data }, ... }; croaks on a body that is not
# multipart/form-data with its boundary, or whose parts do not end.
sub parse {
    my ($body, $content_type) = @_;
    croak 'not multipart/form-data' unless defined $content_type && $content_type =~ m{multipart/form-data}i;
    my ($boundary) = $content_type =~ /boundary=(?:"([^"]+)"|([^;\s]+))/i ? ($1 // $2) : ();
    croak 'multipart/form-data: no boundary' unless defined $boundary && length $boundary;
    my $delim = "--$boundary";
    my %parts;
    my $pos = index($body, $delim);
    croak 'multipart/form-data: no first boundary' if $pos < 0;
    $pos += length $delim;
    while (1) {
        # after a boundary: "--" closes the body, else CRLF starts a part
        last if substr($body, $pos, 2) eq '--';
        croak 'multipart/form-data: a boundary not followed by CRLF' unless substr($body, $pos, 2) eq "\r\n";
        $pos += 2;
        my $head_end = index($body, "\r\n\r\n", $pos);
        croak 'multipart/form-data: a part without headers' if $head_end < 0;
        my %head = _headers(substr($body, $pos, $head_end - $pos));
        my $disp = $head{'content-disposition'} // '';
        my ($name) = $disp =~ /\bname="([^"]*)"/;
        croak 'multipart/form-data: a part without a name' unless defined $name;
        my ($filename) = $disp =~ /\bfilename="([^"]*)"/;
        my $data_at = $head_end + 4;
        my $next = index($body, "\r\n$delim", $data_at);
        croak 'multipart/form-data: a part that does not end' if $next < 0;
        $parts{$name} = {
            name => $name,
            (defined $filename ? (filename => $filename) : ()),
            type => $head{'content-type'},
            data => substr($body, $data_at, $next - $data_at),
        };
        $pos = $next + 2 + length $delim;
    }
    return \%parts;
}

# The part's headers as name => value (names lower-cased)
sub _headers {
    my ($text) = @_;
    my %h;
    for my $line (split /\r\n/, $text) {
        my ($k, $v) = split /:\s*/, $line, 2;
        $h{lc $k} = $v if defined $v;
    }
    return %h;
}

1;

__END__

=head1 NAME

ROCm::Inference::Multipart - the parts of a multipart/form-data request body

=head1 SYNOPSIS

  my $parts = ROCm::Inference::Multipart::parse($req->{body}, $req->{headers}{'content-type'});
  my $wav = $parts->{file}{data};

=head1 FUNCTIONS

=over

=item parse($body, $content_type)

Returns a hash reference from each part's field name to C<{ name, filename, type, data }>: C<filename> only for file
parts, C<type> the part's Content-Type (undef when it has none), C<data> the bytes between its headers and the next
boundary. Croaks when the content type is not multipart/form-data, the boundary is missing, or a part does not end.

=back

=cut
