package ROCm::Inference::Convert::MiniMaxVAE;
# MiniMax-H3-Image's VAE, the decoder part (PLAN.md, MiniMax M2): a ViT over the latent positions (diffusers'
# AutoencoderKLMiniMaxH3Image: post_quant_conv, MiniMaxH3ImageViTDecoder2d). The encoder (3-D causal convolutions, for
# edit only) is not written. The tensors are the checkpoint's own values in F32 (no conversion, so no loss), under their
# checkpoint names (all shorter than GGUF's 63 bytes); the latents' per-channel mean and std are the tensors
# latents.mean and latents.std. The configuration is keys: minimax_h3_vae.*. The spatial compression (16) is the
# reference's constructor default: the VAE's config does not set the downsample factors (2 x 2 x 2 x 2 x 1 x 1).
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

sub is_vae_dir {
    my ($dir) = @_;
    my $cfg = eval { _json(File::Spec->catfile($dir, 'config.json')) } or return 0;
    return (($cfg->{_class_name} // '') eq 'MiniMaxH3ImageVAE') ? 1 : 0;
}

sub convert {
    my ($dir, $out, %a) = @_;
    croak 'minimax vae: F32 copies only (the checkpoint as it is)' if $a{quant};
    my $cfg = _json(File::Spec->catfile($dir, 'config.json'));
    croak 'not a MiniMax-H3-Image VAE' unless ($cfg->{_class_name} // '') eq 'MiniMaxH3ImageVAE';
    my $dec = $cfg->{vit_decoder_kwargs} // croak 'vae config: no vit_decoder_kwargs';
    my $U32 = ROCm::Inference::Convert::U32;
    my $F32 = ROCm::Inference::Convert::F32;
    my $STR = ROCm::Inference::Convert::STR;
    my $ARR = ROCm::Inference::Convert::ARR;
    my $heads = $dec->{heads} // croak 'no heads';
    my $hd = $dec->{dim_head} // croak 'no dim_head';
    my @kv = (
        ['general.architecture', $STR, 'minimax_h3_vae'],
        ['general.type', $STR, 'model'],
        ['minimax_h3_vae.latent_channels', $U32, $cfg->{latent_channels}],
        ['minimax_h3_vae.spatial_compression', $U32, 16],
        ['minimax_h3_vae.out_channels', $U32, 3],
        ['minimax_h3_vae.decoder.layers', $U32, $dec->{num_layers}],
        ['minimax_h3_vae.decoder.heads', $U32, $heads],
        ['minimax_h3_vae.decoder.head_dim', $U32, $hd],
        ['minimax_h3_vae.decoder.register_tokens', $U32, 4],
        ['minimax_h3_vae.decoder.rope_theta', $F32, 0 + $dec->{rope_theta}],
        ['minimax_h3_vae.decoder.rope_dim_ratio', $F32, 0 + $dec->{rope_dim_ratio}],
        ['minimax_h3_vae.decoder.norm_eps', $F32, 1e-5],
        ['minimax_h3_vae.tile_size', $U32, $cfg->{vae_tile_size}],
        ['minimax_h3_vae.tile_overlap', $U32, $cfg->{vae_tile_overlap_min}],
    );
    my @specs;
    my $file = File::Spec->catfile($dir, 'diffusion_pytorch_model.safetensors');
    my $h = ROCm::Inference::Convert::Diffusers::safetensors_header(lfs_path($file));
    my $wanted = qr/^(?:decoder\.|post_quant_conv\.)/;
    for my $name (sort keys %$h) {
        next unless $name =~ $wanted;
        my $t = $h->{$name};
        my $dt = uc $t->{dtype};
        croak "$name: $dt (the VAE is F32)" unless $dt eq 'F32';
        my @shape = @{$t->{shape}};
        my $bytes = $t->{end} - $t->{start};
        my $numel = 1;
        $numel *= $_ for @shape;
        croak "$name: $bytes bytes, expected " . 4 * $numel unless $bytes == 4 * $numel;
        my $fname = lfs_path($t->{file});
        my $start = $t->{start};
        push @specs, _spec($name, [reverse(@shape ? @shape : (1))], $fname, $start, $bytes);
    }
    # the latents' normalization (diffusers: z * std + mean before the decoder), 24 values each
    for my $k (qw(latents_mean latents_std)) {
        my @v = map { 0 + $_ } @{$cfg->{$k} // croak "vae config: no $k"};
        my $bytes = 4 * @v;
        my $buf = pack('f<*', @v);
        my $name = $k eq 'latents_mean' ? 'latents.mean' : 'latents.std';
        push @specs, {name => $name, ne => [scalar @v], type => ROCm::Inference::Convert::gguf_type_id('f32'), from => 'f32',
                      in_bytes => $bytes, why => "the config's $k",
                      raw => sub { $buf }};
    }
    croak 'no VAE decoder tensors in ' . $file unless @specs > 2;
    return ROCm::Inference::Convert::write_file($out, \@kv, \@specs, 32, %a);
}

sub _spec {
    my ($name, $ne, $file, $start, $bytes) = @_;
    return {name => $name, ne => $ne, type => ROCm::Inference::Convert::gguf_type_id('f32'), from => 'f32',
            in_bytes => $bytes, why => "the checkpoint's $name",
            raw => sub {
                open my $fh, '<:raw', $file or croak "$file: $!";
                seek($fh, $start, 0) or croak "$file: seek";
                read($fh, my $buf, $bytes) == $bytes or croak "$file: short read";
                $buf;
            }};
}

1;
