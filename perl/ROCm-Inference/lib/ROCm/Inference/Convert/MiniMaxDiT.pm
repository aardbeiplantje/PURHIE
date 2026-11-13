package ROCm::Inference::Convert::MiniMaxDiT;
# MiniMax-H3-Image's diffusion transformer (MiniMaxH3DiTModel, PLAN.md M3): the checkpoint's own tensors in BF16 (raw copies,
# no conversion, so no loss), under their checkpoint names (all shorter than GGUF's 63 bytes). The configuration is keys
# minimax_h3_dit.*. The token refiner, the 50 blocks (fused QKV, RMS-normed q and k, GELU or SiLU MLP as the checkpoint
# has it: fc1, fc2), the adaLN projections and the time embedder are all written; rope.inv_freq is written as it is.
use strict;
use warnings;
use Carp qw(croak);
use File::Spec;
use JSON::PP ();

use ROCm::Inference::Convert ();
use ROCm::Inference::SafeTensors qw(lfs_path);
use ROCm::Inference::Convert::Diffusers ();

sub _json {
    my ($path) = @_;
    open my $fh, '<:raw', $path or croak "$path: $!";
    local $/;
    return JSON::PP->new->decode(scalar <$fh>);
}

sub is_dit_dir {
    my ($dir) = @_;
    my $cfg = eval { _json(File::Spec->catfile($dir, 'config.json')) } or return 0;
    return (($cfg->{_class_name} // '') eq 'MiniMaxH3DiTModel') ? 1 : 0;
}

# the checkpoint's shards (model.safetensors.index.json's weight_map) in file order
sub _shards {
    my ($dir) = @_;
    my $idx = _json(File::Spec->catfile($dir, 'model.safetensors.index.json'));
    my %files = map { $_ => 1 } values %{$idx->{weight_map}};
    return sort keys %files;
}

sub convert {
    my ($dir, $out, %a) = @_;
    croak 'minimax dit: BF16 copies only (the checkpoint as it is)' if $a{quant};
    my $cfg = _json(File::Spec->catfile($dir, 'config.json'));
    croak 'not a MiniMax-H3 DiT' unless ($cfg->{_class_name} // '') eq 'MiniMaxH3DiTModel';
    my $U32 = ROCm::Inference::Convert::U32;
    my $F32 = ROCm::Inference::Convert::F32;
    my $STR = ROCm::Inference::Convert::STR;
    my @kv = (
        ['general.architecture', $STR, 'minimax_h3_dit'],
        ['general.type', $STR, 'model'],
        ['minimax_h3_dit.hidden_size', $U32, $cfg->{hidden_size}],
        ['minimax_h3_dit.layers', $U32, $cfg->{num_layers}],
        ['minimax_h3_dit.token_refiner_layers', $U32, $cfg->{token_refiner_num_layers}],
        ['minimax_h3_dit.heads', $U32, $cfg->{num_attention_heads}],
        ['minimax_h3_dit.head_dim', $U32, $cfg->{attention_head_dim}],
        ['minimax_h3_dit.ffn_size', $U32, $cfg->{ffn_hidden_size}],
        ['minimax_h3_dit.latents_dim', $U32, $cfg->{latents_dim}],
        ['minimax_h3_dit.text_dim', $U32, $cfg->{text_dim}],
        ['minimax_h3_dit.timestep_input_dim', $U32, $cfg->{timestep_input_dim}],
        ['minimax_h3_dit.time_embed_hidden_size', $U32, $cfg->{time_embed_hidden_size}],
        ['minimax_h3_dit.time_embed_dim', $U32, $cfg->{time_embed_dim}],
        ['minimax_h3_dit.patch_t', $U32, $cfg->{patch_size}[0]],
        ['minimax_h3_dit.patch_h', $U32, $cfg->{patch_size}[1]],
        ['minimax_h3_dit.patch_w', $U32, $cfg->{patch_size}[2]],
        ['minimax_h3_dit.norm_eps', $F32, 0 + $cfg->{norm_eps}],
        ['minimax_h3_dit.qk_norm_eps', $F32, 0 + $cfg->{qk_norm_eps}],
        ['minimax_h3_dit.final_norm_eps', $F32, 0 + $cfg->{final_norm_eps}],
        ['minimax_h3_dit.rope_inv_freq_len', $U32, $cfg->{rope_inv_freq_len}],
    );
    my @specs;
    my %seen;
    for my $shard (_shards($dir)) {
        my $file = lfs_path(File::Spec->catfile($dir, $shard));
        my $h = ROCm::Inference::Convert::Diffusers::safetensors_header($file);
        for my $name (sort keys %$h) {
            croak "$name in two shards" if $seen{$name}++;
            my $t = $h->{$name};
            my $dt = uc $t->{dtype};
            my @shape = @{$t->{shape}};
            my $bytes = $t->{end} - $t->{start};
            my ($type, $per);
            if ($dt eq 'BF16') { $type = ROCm::Inference::Convert::gguf_type_id('bf16'); $per = 2 }
            elsif ($dt eq 'F32') { $type = ROCm::Inference::Convert::gguf_type_id('f32'); $per = 4 }
            else { croak "$name: $dt (the DiT is BF16 and F32)" }
            my $numel = 1;
            $numel *= $_ for @shape;
            croak "$name: $bytes bytes, expected " . ($per * $numel) unless $bytes == $per * $numel;
            my $start = $t->{start};
            push @specs, {name => $name, ne => [reverse(@shape ? @shape : (1))], type => $type, from => lc $dt,
                          in_bytes => $bytes, why => "the checkpoint's $name",
                          raw => sub {
                              open my $fh, '<:raw', $file or croak "$file: $!";
                              seek($fh, $start, 0) or croak "$file: seek";
                              read($fh, my $buf, $bytes) == $bytes or croak "$file: short read";
                              $buf;
                          }};
        }
    }
    croak 'no tensors in ' . $dir unless @specs;
    return ROCm::Inference::Convert::write_file($out, \@kv, \@specs, 32, %a);
}

1;
