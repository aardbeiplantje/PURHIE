package ROCm::Inference::Quant;

# Which type each tensor of a model is written as: the policy of bin/rocm-convert (a checkpoint or a GGUF in).

use strict;
use warnings;

use Carp qw(croak);
use ROCm::Inference ();
use ROCm::Inference::GGUF qw(gguf_type_id gguf_type_info);

our $VERSION = '0.01';

my %ALIAS = (fast => 'q4_0_rocmfp4_fast', rocmfp4_fast => 'q4_0_rocmfp4_fast', rocmfp4 => 'q4_0_rocmfp4');
# general.file_type (llama_ftype) of a file whose tensors are mostly this type
our %FTYPE = (f32 => 0, f16 => 1, q8_0 => 7, bf16 => 32, q4_0_rocmfp4 => 100, q4_0_rocmfp4_fast => 103);
# Sources re-encoded by default: the ones a 4-bit or 8-bit encoding loses nothing new from. Anything else (K-quants,
# MXFP4 / NVFP4 experts, ...) is already quantized and is kept unless a rule names it.
my %HIGH = map { $_ => 1 } qw(f32 f16 bf16 q8_0);
my %FLOAT = map { $_ => 1 } qw(f32 f16 bf16);

# Recipes: rules after the user's, for a 4-bit target. strix spends bits where a token reads little of the tensor or
# it is small: Q8_0 for the lookup tables (a token reads one row of token_embd / the PLE table), the LM head (read
# whole every token, but one matrix: on Qwen3-30B-A3B it closes 60% of FAST's perplexity gap to Q8_0 for 1% of the
# file and no measurable tok/s), the DSA indexer and the mHC mixers (small; they select keys and mix the streams).
# Attention K / V stay at the target: neither ROCmFP4 nor Q8_0 there improved perplexity (2026-11-09).
our %RECIPE = (
    plain => [],
    strix => [
        ['^(?:per_layer_)?token_embd\.weight$' => 'q8_0'],
        ['^output\.weight$' => 'q8_0'],
        ['\.indexer[._]' => 'q8_0'],
        ['(?:^|\.)hc_|^output_hc_' => 'q8_0'],
    ],
);

# Tensors a quantization leaves alone whatever the target (the same set as llama.cpp's llama-quantize): 1-D
# tensors, names not ending in "weight", norms, routers (expert gating), convolutions, position / token-type
# embeddings and vision-encoder parts
my @NEVER = (qr/_norm\.weight$/, qr/ffn_gate_inp\.weight/, qr/ffn_gate_tid2eid\.weight/, qr/ssm_conv1d/,
             qr/shortconv\.conv\.weight/, qr/altup/, qr/laurel/, qr/per_layer_model_proj/, qr/^position_embd\./,
             qr/^token_types\./, qr/time_mix_(?:first|w[012]|[av][012]|a[012]|g[12]|decay_w[12]|lerp_fused)\.weight/,
             qr/attn_rel_b\.weight/, qr/\.position_embd/, qr/sam\.(?:pos_embd|neck\.|net_)/, qr/\.rel_pos/,
             qr/\.patch_embd/, qr/\.patch_merger/);

# A tensor type name, normalized (fast, rocmfp4, ... accepted); keep passes through
sub type_name {
    my ($n) = @_;
    my $t = lc $n;
    $t = $ALIAS{$t} // $t;
    return $t if $t eq 'keep';
    croak "unknown tensor type '$n'" unless defined gguf_type_id($t);
    croak "cannot encode $t" unless ROCm::Inference::quant_can_encode(gguf_type_id($t));
    return $t;
}

# new(target => TYPE, recipe => NAME, rules => ['REGEX=TYPE', ...])
sub new {
    my ($class, %a) = @_;
    my $target = type_name($a{target} // 'q4_0_rocmfp4_fast');
    croak "$target is not a file type (one of: " . join(', ', sort keys %FTYPE) . ")" unless exists $FTYPE{$target};
    my $recipe = $a{recipe} // 'strix';
    croak "unknown recipe '$recipe' (" . join(', ', sort keys %RECIPE) . ")" unless $RECIPE{$recipe};
    my @rules = map {
        my ($re, $t) = /^(.+)=([^=]+)$/ or croak "a type rule is REGEX=TYPE, not '$_'";
        [qr/$re/, type_name($t), $_];
    } @{$a{rules} // []};
    my @recipe = $target =~ /^q4_0_rocmfp4/ ? map { [qr/$_->[0]/, $_->[1], "recipe $recipe"] } @{$RECIPE{$recipe}} : ();
    return bless {target => $target, ftype => $FTYPE{$target}, rules => \@rules, recipe => \@recipe,
                  recipe_name => $recipe}, $class;
}

sub target { $_[0]{target} }
sub ftype  { $_[0]{ftype} }

# plan($name, \@ne, $src_type_name, %opt): (the type to write, why when it is not the target) for a GGUF tensor of
# that name and shape whose data is of that type. %opt: f32 => 1 (the converter's rule: keep the tensor F32).
sub plan {
    my ($self, $name, $ne, $src, %opt) = @_;
    my $eligible = @$ne >= 2 && $name =~ /weight$/ && !grep { $name =~ $_ } @NEVER;
    my ($want, $why);
    if ($opt{f32}) {
        ($want, $why) = ('f32', 'kept F32');
    } elsif (!$eligible) {
        ($want, $why) = ($src, 'kept');
    } elsif (my ($r) = grep { $name =~ $_->[0] } @{$self->{rules}}) {
        ($want, $why) = $r->[1] eq 'keep' ? ($src, "kept by $r->[2]") : ($r->[1], "by $r->[2]");
    } elsif (!$HIGH{$src} && !($opt{decode} && $FLOAT{$self->{target}})) {
        ($want, $why) = ($src, "kept ($src is already quantized)");
    } elsif (my ($rr) = grep { $name =~ $_->[0] } @{$self->{recipe}}) {
        ($want, $why) = ($rr->[1], "by $rr->[2]");
    } else {
        $want = $self->{target};
    }
    return ($src, undef) if $want eq $src;
    my (undef, $blk) = gguf_type_info(gguf_type_id($want));
    if ($ne->[0] % $blk) {  # rows that are not whole blocks of the target: F16 (as llama-quantize falls back)
        $why = "$ne->[0] values a row are not whole blocks of $blk for $want";
        $want = 'f16';
    }
    return ($want, $why);
}

1;

__END__

=head1 NAME

ROCm::Inference::Quant - the per-tensor type policy of rocm-convert

=head1 SYNOPSIS

  my $q = ROCm::Inference::Quant->new(target => 'fast', recipe => 'strix', rules => ['ffn_down_exps=rocmfp4']);
  my ($type, $why) = $q->plan('blk.0.attn_q.weight', [4096, 4096], 'bf16');

=head1 DESCRIPTION

=over

=item new(target => TYPE, recipe => NAME, rules => [REGEX=TYPE ...])

TYPE is the file's type (default C<q4_0_rocmfp4_fast>); the recipe (default C<strix>, or C<plain>) applies to a
4-bit TYPE; the rules come first, the first matching one wins, C<keep> leaves a tensor as it is.

=item plan($name, \@ne, $source_type, %opt)

The type a tensor is written as and, when it is not TYPE, why. Matrices (2 or more dimensions, names ending in
C<weight>) except norms, routers, convolutions, position embeddings and vision-encoder parts are re-encoded, from
F32, F16, BF16 or Q8_0 only; rows that are not whole blocks of the type become F16. C<< f32 => 1 >> keeps the
tensor F32.

=item type_name($name)

A tensor type's canonical name (C<fast> is C<q4_0_rocmfp4_fast>); dies on a type that cannot be encoded.

=item target, ftype

The file's type and its C<general.file_type> value.

=back

=cut
