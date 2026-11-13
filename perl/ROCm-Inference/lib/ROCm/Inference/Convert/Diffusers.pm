package ROCm::Inference::Convert::Diffusers;
# A diffusers pipeline (Stable Diffusion 2.x, e.g. stabilityai/sd-turbo: model_index.json, the UNet, the text encoder, the
# VAE, the scheduler, the CLIP tokenizer) as one GGUF (PLAN.md I2 of the image phase). The tensors are the checkpoint's own
# values as F16 (the fp16 variant's files, read as they are). GGUF names hold 63 bytes and diffusers' names are longer, so a
# tensor is named by its component and index (te0001, un0042, va0007) and the component's list of checkpoint names is a KV
# string array (sd.names.te, sd.names.un, sd.names.va): the engine maps one to the other. The configs the engine needs are
# keys: sd.unet.*, sd.te.*, sd.vae.*, sd.sched.*. The CLIP tokenizer: sd.tok.vocab and sd.tok.merges.
use strict;
use warnings;
use Carp qw(croak);
use File::Basename qw(basename);
use File::Spec;
use JSON::PP ();

use ROCm::Inference::Convert ();

my %COMPONENT = (text_encoder => 'te', unet => 'un', vae => 'va');

sub _json {
    my ($path) = @_;
    open my $fh, '<:raw', $path or croak "$path: $!";
    local $/;
    return JSON::PP->new->decode(scalar <$fh>);
}

# The tensors of one safetensors file: name => {dtype, shape, start, end} (start and end in the file)
sub safetensors_header {
    my ($path) = @_;
    open my $fh, '<:raw', $path or croak "$path: $!";
    my $n = 0;
    read($fh, my $len, 8) == 8 or croak "$path: no header";
    $n = unpack 'Q<', $len;
    read($fh, my $hdr, $n) == $n or croak "$path: short header";
    my $h = JSON::PP->new->decode($hdr);
    my $base = 8 + $n;
    my %t;
    for my $k (keys %$h) {
        next if $k eq '__metadata__';
        my $e = $h->{$k};
        $t{$k} = {dtype => $e->{dtype}, shape => $e->{shape}, start => $base + $e->{data_offsets}[0],
                  end => $base + $e->{data_offsets}[1], file => $path};
    }
    return \%t;
}

# The component's checkpoint file: the fp16 variant when the folder has one, else the plain file
sub component_file {
    my ($dir, $sub) = @_;
    my $d = File::Spec->catdir($dir, $sub);
    opendir my $dh, $d or croak "$d: $!";
    my @st = sort grep { /\.safetensors\z/ } readdir $dh;
    closedir $dh;
    croak "$d: no safetensors" unless @st;
    my ($fp16) = grep { /fp16/ } @st;
    return File::Spec->catfile($d, $fp16 // $st[0]);
}

sub convert {
    my ($dir, $out, %a) = @_;
    croak 'diffusers: F16 copies only (the fp16 checkpoint as it is: no quantization yet)' if $a{quant};
    my $mi = _json(File::Spec->catfile($dir, 'model_index.json'));
    croak 'not a Stable Diffusion pipeline' unless ($mi->{_class_name} // '') =~ /StableDiffusion/;
    my $U32 = ROCm::Inference::Convert::U32;
    my $F32 = ROCm::Inference::Convert::F32;
    my $STR = ROCm::Inference::Convert::STR;
    my $ARR = ROCm::Inference::Convert::ARR;
    my $unet = _json(File::Spec->catfile($dir, 'unet', 'config.json'));
    my $te = _json(File::Spec->catfile($dir, 'text_encoder', 'config.json'));
    my $vae = _json(File::Spec->catfile($dir, 'vae', 'config.json'));
    my $sch = _json(File::Spec->catfile($dir, 'scheduler', 'scheduler_config.json'));
    my @kv = (
        ['general.architecture', $STR, 'sd'],
        ['general.type', $STR, 'model'],
        ['general.name', $STR, basename($dir)],
        ['sd.unet.in_channels', $U32, $unet->{in_channels}],
        ['sd.unet.out_channels', $U32, $unet->{out_channels}],
        ['sd.unet.block_out_channels', $ARR, [$U32, [map { int $_ } @{$unet->{block_out_channels}}]]],
        ['sd.unet.layers_per_block', $U32, $unet->{layers_per_block}],
        ['sd.unet.attention_head_dim', $ARR, [$U32, [map { int $_ } @{$unet->{attention_head_dim}}]]],
        ['sd.unet.cross_attention_dim', $U32, $unet->{cross_attention_dim}],
        ['sd.unet.norm_groups', $U32, $unet->{norm_num_groups}],
        ['sd.unet.linear_projection', $U32, $unet->{use_linear_projection} ? 1 : 0],
        ['sd.te.n_embd', $U32, $te->{hidden_size}],
        ['sd.te.n_layer', $U32, $te->{num_hidden_layers}],
        ['sd.te.n_head', $U32, $te->{num_attention_heads}],
        ['sd.te.n_ff', $U32, $te->{intermediate_size}],
        ['sd.te.n_ctx', $U32, $te->{max_position_embeddings}],
        ['sd.te.vocab', $U32, $te->{vocab_size}],
        ['sd.te.act', $STR, $te->{hidden_act} // 'gelu'],
        ['sd.vae.block_out_channels', $ARR, [$U32, [map { int $_ } @{$vae->{block_out_channels}}]]],
        ['sd.vae.layers_per_block', $U32, $vae->{layers_per_block}],
        ['sd.vae.latent_channels', $U32, $vae->{latent_channels}],
        ['sd.vae.norm_groups', $U32, $vae->{norm_num_groups}],
        ['sd.vae.scaling_factor', $F32, 0 + $vae->{scaling_factor}],
        ['sd.sched.num_train_timesteps', $U32, $sch->{num_train_timesteps}],
        ['sd.sched.beta_start', $F32, 0 + $sch->{beta_start}],
        ['sd.sched.beta_end', $F32, 0 + $sch->{beta_end}],
        ['sd.sched.beta_schedule', $STR, $sch->{beta_schedule}],
        ['sd.sched.prediction_type', $STR, $sch->{prediction_type}],
        ['sd.sched.timestep_spacing', $STR, $sch->{timestep_spacing}],
        ['sd.sched.steps_offset', $U32, $sch->{steps_offset} // 0],
    );
    # the CLIP tokenizer: the vocabulary by id and the merges in rank order
    my $tokdir = File::Spec->catdir($dir, 'tokenizer');
    my $vocab = _json(File::Spec->catfile($tokdir, 'vocab.json'));
    my @tok = ('') x (1 + (sort { $b <=> $a } values %$vocab)[0]);
    $tok[$vocab->{$_}] = $_ for keys %$vocab;
    open my $mh, '<:raw', File::Spec->catfile($tokdir, 'merges.txt') or croak "merges.txt: $!";
    my @merges;
    while (my $l = <$mh>) {
        chomp $l;
        next if $l =~ /^#version/ || $l eq '';
        push @merges, $l;
    }
    close $mh;
    # the sequence's bounds: the tokenizer's start and end of text (the text encoder's config has its own, unrelated ones)
    my $bos = $vocab->{'<|startoftext|>'} // croak 'no <|startoftext|> in the vocabulary';
    my $eos = $vocab->{'<|endoftext|>'} // croak 'no <|endoftext|> in the vocabulary';
    push @kv, ['sd.te.bos', $U32, $bos], ['sd.te.eos', $U32, $eos];
    # the padding: the tokenizer's pad token (sd-turbo pads with "!", not with the end of text)
    my $tc = _json(File::Spec->catfile($tokdir, 'tokenizer_config.json'));
    my $pad_text = ref $tc->{pad_token} eq 'HASH' ? $tc->{pad_token}{content} : $tc->{pad_token};
    my $pad_id = defined $pad_text ? ($vocab->{$pad_text} // croak "pad token $pad_text is not in the vocabulary") : $tok[0] // 0;
    push @kv, ['sd.te.pad', $U32, $pad_id];
    push @kv, ['sd.tok.vocab', $ARR, [$STR, \@tok]], ['sd.tok.merges', $ARR, [$STR, \@merges]];
    my @specs;
    my $total = 0;
    for my $sub (sort keys %COMPONENT) {
        my $file = component_file($dir, $sub);
        my $h = safetensors_header($file);
        my $prefix = $COMPONENT{$sub};
        my @names = sort keys %$h;
        push @kv, ["sd.names.$prefix", $ARR, [$STR, \@names]];
        my $i = 0;
        for my $name (@names) {
            my $t = $h->{$name};
            my $dt = uc $t->{dtype};
            croak "$name: $dt (the fp16 variant: F16 only)" unless $dt eq 'F16';
            my @shape = @{$t->{shape}};
            my $rows = 1;
            $rows *= $_ for @shape[0 .. $#shape - 1];
            my $row_bytes = 2 * ($shape[-1] // 1);
            my $bytes = $t->{end} - $t->{start};
            croak "$name: $bytes bytes, expected " . ($rows * $row_bytes) unless $bytes == $rows * $row_bytes;
            my $fh_file = $file;
            my $start = $t->{start};
            push @specs, {name => sprintf('%s%04d', $prefix, $i++), ne => [reverse(@shape ? @shape : (1))],
                          type => ROCm::Inference::Convert::gguf_type_id('f16'), from => 'f16', rows => $rows,
                          in_bytes => $bytes, why => "the checkpoint's $name",
                          raw => sub {
                              open my $fh, '<:raw', $fh_file or croak "$fh_file: $!";
                              seek($fh, $start, 0) or croak "$fh_file: seek";
                              read($fh, my $buf, $bytes) == $bytes or croak "$fh_file: short read";
                              $buf;
                          }};
            $total += $bytes;
        }
    }
    croak 'no tensors in ' . $dir unless @specs;
    return ROCm::Inference::Convert::write_file($out, \@kv, \@specs, 32, %a);
}

1;
