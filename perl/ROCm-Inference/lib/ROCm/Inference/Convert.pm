package ROCm::Inference::Convert;

# A Hugging Face checkpoint (safetensors, config.json, tokenizer.json) to one GGUF, streamed tensor by tensor and
# written in the engine's types directly: an FP8 or BF16 release needs no full-precision GGUF in between. The
# metadata, vocabulary and tensor names are llama.cpp's convert_hf_to_gguf.py's (so llama.cpp and the engine read
# the file alike); bin/rocm-convert is the command, ROCm::Inference::Convert::<Arch> the architectures.

use strict;
use warnings;
no warnings qw(experimental::builtin);

use Carp qw(croak);
use File::Basename qw(basename);
use File::Spec;

use ROCm::Inference ();
use ROCm::Inference::GGUF qw(gguf_write_header gguf_type_id gguf_type_info);
use ROCm::Inference::GGUF ();
use ROCm::Inference::Quant ();
use ROCm::Inference::SafeTensors qw(st_open st_read_rows st_rows st_row_bytes);

our $VERSION = '0.01';

# Metadata value types (gguf.h)
use constant {U8 => 0, I8 => 1, U16 => 2, I16 => 3, U32 => 4, I32 => 5, F32 => 6, BOOL => 7, STR => 8, ARR => 9,
              U64 => 10, I64 => 11, F64 => 12};

# Token types (llama.cpp's llama_token_type)
use constant {TOK_NORMAL => 1, TOK_UNKNOWN => 2, TOK_CONTROL => 3, TOK_USER_DEFINED => 4, TOK_UNUSED => 5,
              TOK_BYTE => 6};

my %ARCH;  # Hugging Face architecture name => package

sub register { my ($pkg, @names) = @_; $ARCH{$_} = $pkg for @names }

sub architectures { return sort keys %ARCH }

sub _slurp {
    my ($path) = @_;
    $path = ROCm::Inference::SafeTensors::lfs_path($path);
    open my $f, '<:raw', $path or croak "$path: $!";
    local $/;
    my $s = <$f>;
    close $f;
    return $s;
}

sub json_file {
    my ($path) = @_;
    return -f $path ? ROCm::Inference::json_decode(_slurp($path)) : undef;
}

sub is_bool { my ($v) = @_; return defined $v && !ref $v && builtin::is_bool($v) }

# ---------------------------------------------------------------- a model

# new($dir): the checkpoint's config (text_config moved up, the rope parameters normalized as transformers 5 does),
# its tensors, and the architecture's package
sub open_model {
    my ($class, $dir, %opt) = @_;
    my $cfg = json_file(File::Spec->catfile($dir, 'config.json')) or croak "$dir: no config.json";
    my $hf_arch = $opt{arch} // ($cfg->{architectures} // [])->[0] // croak "$dir/config.json: no architectures";
    my $pkg = $ARCH{$hf_arch}
        or croak "$hf_arch is not supported (supported: " . join(', ', sort keys %ARCH) . ")";
    my %hp = %$cfg;
    %hp = (%hp, %{$cfg->{text_config}}) if ref $cfg->{text_config} eq 'HASH';
    my $self = bless {dir => $dir, config => $cfg, hp => \%hp, hf_arch => $hf_arch, kv => [], kv_at => {},
                      opt => \%opt}, $pkg;
    $self->{st} = st_open($dir);
    # rope_parameters (transformers 5) or rope_scaling, with rope_theta / rope_type mirrored into it
    my %rp = %{$hp{rope_parameters} // $hp{rope_scaling} // {}};
    if (!exists $rp{full_attention} && !exists $rp{sliding_attention}) {
        my $theta = $self->find(qw(global_rope_theta rope_global_theta rope_theta_global rope_theta rotary_emb_base));
        my $local = $self->find(qw(local_rope_theta rope_local_theta rope_theta_local swa_rope_theta rope_local_base_freq));
        my $prf = $self->find(qw(partial_rotary_factor rope_pct rope_percent));
        my $omp = $self->find(qw(original_max_position_embeddings));
        $rp{sliding_attention} = {rope_theta => $local} if defined $local;
        $rp{rope_theta} //= $theta if defined $theta;
        $rp{rope_type} //= $rp{type} if defined $rp{type};
        $rp{partial_rotary_factor} //= $prf if defined $prf;
        $rp{original_max_position_embeddings} //= $omp if defined $omp;
    }
    $self->{rope} = \%rp;
    $self->{block_count} = $self->find(qw(n_layers num_hidden_layers n_layer num_layers))
        // croak "$dir: no layer count";
    $self->init if $self->can('init');
    return $self;
}

sub hp    { $_[0]{hp} }
sub arch  { croak ref($_[0]) . ': no arch' }
sub st    { $_[0]{st} }

# The first of these hyperparameters present
sub find {
    my ($self, @keys) = @_;
    for (@keys) { return $self->{hp}{$_} if exists $self->{hp}{$_} && defined $self->{hp}{$_} }
    return undef;
}

sub need {
    my ($self, @keys) = @_;
    my $v = $self->find(@keys);
    croak "$self->{dir}: none of @keys in config.json" unless defined $v;
    return $v;
}

# Metadata, in the order added; a key added again replaces the value in place
sub kv {
    my ($self, $key, $type, $value) = @_;
    if (defined(my $i = $self->{kv_at}{$key})) { $self->{kv}[$i] = [$key, $type, $value]; return }
    $self->{kv_at}{$key} = scalar @{$self->{kv}};
    push @{$self->{kv}}, [$key, $type, $value];
}
sub kv_arch { my ($self, $k, @r) = @_; $self->kv($self->arch . ".$k", @r) }
sub u32  { $_[0]->kv_arch($_[1], U32, int $_[2]) }
sub f32  { $_[0]->kv_arch($_[1], F32, 0 + $_[2]) }
sub bool { $_[0]->kv_arch($_[1], BOOL, $_[2] ? 1 : 0) }
sub str  { $_[0]->kv_arch($_[1], STR, $_[2]) }

# An array value as gguf-py types it: strings, bools, floats (FLOAT32) or ints (INT32), from the elements
sub arr {
    my ($self, $k, $vals, $et) = @_;
    return unless @$vals;
    $et //= _elem_type($vals);
    $self->kv_arch($k, ARR, [$et, [map { $et == F32 ? 0 + $_ : $et == BOOL ? ($_ ? 1 : 0) : $et == STR ? $_ : int $_ } @$vals]]);
}

sub _elem_type {
    my ($vals) = @_;
    return STR if grep { !is_bool($_) && !_num($_) } @$vals;
    return BOOL if is_bool($vals->[0]);
    return F32 if grep { /[.eE]/ || $_ != int $_ } @$vals;
    return I32;
}
sub _num { my ($v) = @_; return defined $v && $v =~ /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/ }

# An int or a list of ints: u32, or an INT32 array (gguf-py's add_head_count_kv and the like)
sub u32_or_arr {
    my ($self, $k, $v) = @_;
    ref $v eq 'ARRAY' ? $self->arr($k, $v, I32) : $self->u32($k, $v);
}

# ---------------------------------------------------------------- metadata (convert_hf_to_gguf.py's TextModel)

sub general {
    my ($self, $q) = @_;
    $self->kv('general.architecture', STR, $self->arch);
    $self->kv('general.type', STR, 'model');
    if (my $gc = json_file(File::Spec->catfile($self->{dir}, 'generation_config.json'))) {
        # gguf-py's Metadata: [key, its type, generation_config.json's names]
        $gc->{penalty_repeat} //= $gc->{repetition_penalty};
        for (['sequence', STR], ['top_k', I32], ['top_p', F32], ['min_p', F32], ['xtc_probability', F32],
             ['xtc_threshold', F32], ['temp', F32, 'temperature'], ['penalty_last_n', I32], ['penalty_repeat', F32],
             ['mirostat', I32], ['mirostat_tau', F32], ['mirostat_eta', F32]) {
            my ($k, $ty, $src) = @$_;
            my $v = $gc->{$src // $k};
            next unless defined $v && !ref $v;
            $self->kv("general.sampling.$k", $ty, $ty == STR ? $v : $ty == I32 ? int $v : 0 + $v);
        }
    }
    $self->kv('general.name', STR, $self->{opt}{name} // _model_name(basename($self->{dir})));
}

# "qwen3-0.6b" -> "Qwen3 0.6b" (gguf-py's metadata heuristics give the same for plain names)
sub _model_name {
    my ($n) = @_;
    $n =~ s/[-_]+/ /g;
    return join ' ', map { /^[a-z]/ && !/^\d/ ? ucfirst : $_ } split / /, $n;
}

sub text_params {
    my ($self) = @_;
    $self->u32('block_count', $self->{block_count});
    my $v;
    $self->u32('context_length', $v)
        if defined($v = $self->find(qw(max_position_embeddings n_ctx n_positions max_length max_sequence_length model_max_length)));
    $self->u32('embedding_length', $v) if defined($v = $self->find(qw(hidden_size n_embd dim)));
    $self->u32('feed_forward_length', $v)
        if defined($v = $self->find(qw(prefix_dense_intermediate_size dense_intermediate_size intermediate_size n_inner hidden_dim)));
    $self->u32_or_arr('attention.head_count', $v) if defined($v = $self->find(qw(num_attention_heads n_head n_heads)));
    $self->u32_or_arr('attention.head_count_kv', $v) if defined($v = $self->find(qw(num_key_value_heads n_kv_heads)));
    my $hp = $self->{hp};
    $self->bool('attention.causal', 0)
        if (exists $hp->{is_causal} && is_bool($hp->{is_causal}) && !$hp->{is_causal})
           || (is_bool($hp->{use_bidirectional_attention}) && $hp->{use_bidirectional_attention});
    my $rp = $self->{rope}{full_attention} // $self->{rope};
    if (defined(my $rt = $rp->{rope_type})) {
        my $fac = $rp->{factor};
        if ($rt eq 'linear' && defined $fac) {
            $self->str('rope.scaling.type', 'linear');
            $self->f32('rope.scaling.factor', $fac);
        } elsif ($rt eq 'yarn' && defined $fac) {
            $self->str('rope.scaling.type', 'yarn');
            $self->f32('rope.scaling.factor', $fac);
            $self->u32('rope.scaling.original_context_length', $rp->{original_max_position_embeddings});
            $self->f32('rope.scaling.yarn_ext_factor', $rp->{extrapolation_factor}) if defined $rp->{extrapolation_factor};
            my $af = $rp->{attention_factor} // $rp->{attn_factor};
            $self->f32('rope.scaling.yarn_attn_factor', $af) if defined $af;
            $self->f32('rope.scaling.yarn_beta_fast', $rp->{beta_fast}) if defined $rp->{beta_fast};
            $self->f32('rope.scaling.yarn_beta_slow', $rp->{beta_slow}) if defined $rp->{beta_slow};
        } elsif ($rt eq 'su' || $rt eq 'longrope') {
            $self->str('rope.scaling.type', 'longrope');
        }
    }
    if (my $ms = $self->{rope}{mrope_section}) {
        my @m = @$ms;
        push @m, 0 while @m < 4;
        $self->arr('rope.dimension_sections', [@m[0 .. 3]], I32);
    }
    $self->f32('rope.freq_base', $rp->{rope_theta}) if defined $rp->{rope_theta};
    $self->f32('rope.freq_base_swa', $self->{rope}{sliding_attention}{rope_theta})
        if defined $self->{rope}{sliding_attention} && defined $self->{rope}{sliding_attention}{rope_theta};
    $self->f32('attention.layer_norm_rms_epsilon', $v) if defined($v = $self->find(qw(rms_norm_eps norm_eps)));
    $self->f32('attention.layer_norm_epsilon', $v)
        if defined($v = $self->find(qw(layer_norm_eps layer_norm_epsilon norm_epsilon)));
    $self->u32('expert_count', $v) if defined($v = $self->find(qw(num_local_experts num_experts n_routed_experts)));
    $self->u32('expert_used_count', $v)
        if defined($v = $self->find(qw(num_experts_per_tok num_experts_per_token top_k_experts)));
    $self->u32('expert_group_count', $hp->{n_group}) if defined $hp->{n_group};
    $self->u32('expert_group_used_count', $hp->{topk_group}) if defined $hp->{topk_group};
    if (defined(my $sf = $self->find(qw(score_function scoring_func score_func moe_router_activation
                                        moe_router_activation_func expert_selection_fn router_score_func router_score_function)))) {
        my %g = (softmax => 1, sigmoid => 2, sqrtsoftplus => 4);
        croak "unsupported expert gating function $sf" unless $g{$sf};
        $self->u32('expert_gating_func', $g{$sf});
    }
    if (defined(my $hd = $hp->{head_dim})) {
        $self->u32('attention.key_length', $hd);
        $self->u32('attention.value_length', $hd);
    }
}

# ---------------------------------------------------------------- vocabulary (gpt2 / byte-level BPE)

# tokenizer.json's vocabulary as convert_hf_to_gguf.py's get_vocab_base writes it: tokens by id (gaps [PADi]),
# CONTROL for special added tokens, USER_DEFINED for other added ones, NORMAL otherwise
sub vocab_gpt2 {
    my ($self, %opt) = @_;
    my $tj = $self->{tokenizer} //= json_file(File::Spec->catfile($self->{dir}, 'tokenizer.json'))
        or croak "$self->{dir}: no tokenizer.json";
    my %rev;
    my $vocab = $tj->{model}{vocab} or croak "tokenizer.json: no model.vocab";
    $rev{$vocab->{$_}} = $_ for keys %$vocab;
    my %added;
    for my $a ($self->added_tokens) {
        $rev{$a->{id}} = $a->{content};
        $added{$a->{id}} = $a;
    }
    my $max = 0;
    $max < $_ and $max = $_ for keys %rev;
    my $n = $self->find('vocab_size') // $max + 1;
    croak "tokenizer.json has id $max past vocab_size $n" unless $max < $n;
    my (@tok, @type);
    for my $i (0 .. $n - 1) {
        if (!exists $rev{$i}) {
            push @tok, "[PAD$i]";
            push @type, TOK_UNUSED;
            next;
        }
        my $t = $rev{$i};
        if (my $a = $added{$i}) {
            if ($a->{special} || looks_special($t)) {
                push @type, TOK_CONTROL;
            } else {
                $t =~ s/\x{2581}/ /g;
                push @type, TOK_USER_DEFINED;
            }
        } else {
            push @type, TOK_NORMAL;
        }
        push @tok, $t;
    }
    $self->kv('tokenizer.ggml.model', STR, 'gpt2');
    $self->kv('tokenizer.ggml.pre', STR, $self->{opt}{pre} // $opt{pre} // $self->tokenizer_pre);
    $self->kv('tokenizer.ggml.tokens', ARR, [STR, \@tok]);
    $self->kv('tokenizer.ggml.token_type', ARR, [I32, \@type]);
    $self->special_vocab($n, merges => 1, extra => $opt{special});
}

# The added tokens as transformers loads them: tokenizer.json's, and tokenizer_config.json's added_tokens_decoder
# for ids tokenizer.json lacks
sub added_tokens {
    my ($self) = @_;
    return @{$self->{added} //= do {
        my $tj = $self->{tokenizer} //= json_file(File::Spec->catfile($self->{dir}, 'tokenizer.json'));
        my @a = @{$tj->{added_tokens} // []};
        my %have = map { $_->{id} => 1 } @a;
        my $tc = json_file(File::Spec->catfile($self->{dir}, 'tokenizer_config.json')) // {};
        my $d = $tc->{added_tokens_decoder} // {};
        push @a, map { {%{$d->{$_}}, id => 0 + $_} } sort { $a <=> $b } grep { !$have{$_} } keys %$d;
        \@a;
    }};
}

# A byte-fallback BPE vocabulary as gguf-py's LlamaHfVocab reads it (through transformers): the model's tokens by
# id, then added tokens past them; score -1000; BYTE for <0xNN>, CONTROL for special tokens, NORMAL otherwise;
# %opt: model (tokenizer.ggml.model), visible => [tokens written USER_DEFINED though special]
sub vocab_llama_hf {
    my ($self, %opt) = @_;
    my $tj = $self->{tokenizer} //= json_file(File::Spec->catfile($self->{dir}, 'tokenizer.json'))
        or croak "$self->{dir}: no tokenizer.json";
    my $m = $tj->{model};
    croak 'tokenizer.json: not a byte-fallback BPE' unless ($m->{type} // '') eq 'BPE' && $m->{byte_fallback};
    my %rev;
    $rev{$m->{vocab}{$_}} = $_ for keys %{$m->{vocab}};
    my $base = keys %rev;
    my @added = $self->added_tokens;
    $rev{$_->{id}} = $_->{content} for @added;
    my $tc = json_file(File::Spec->catfile($self->{dir}, 'tokenizer_config.json')) // {};
    # transformers' all_special_tokens: the special added tokens, the config's named tokens
    my %special = map { $_->{content} => 1 } grep { $_->{special} } @added;
    for my $k (grep { /_token$/ } keys %$tc) {
        my $v = ref $tc->{$k} eq 'HASH' ? $tc->{$k}{content} : $tc->{$k};
        $special{$v} = 1 if defined $v && !ref $v && !is_bool($v);
    }
    $special{$_} = 1 for map { ref $_ eq 'HASH' ? $_->{content} : $_ }
        @{ref $tc->{extra_special_tokens} eq 'ARRAY' ? $tc->{extra_special_tokens} : []},
        @{$tc->{additional_special_tokens} // []};
    my %visible = map { $_ => 1 } @{$opt{visible} // []};
    my (@tok, @type);
    my @extra = sort { $a <=> $b } grep { $_ >= $base } map { $_->{id} } @added;
    for my $i (0 .. $base - 1, @extra) {
        my $t = $rev{$i} // croak "tokenizer.json: no token $i";
        push @tok, $t;
        push @type, $visible{$t} ? TOK_USER_DEFINED : $i < $base && $t =~ /^<0x[0-9A-Fa-f]{2}>\z/ ? TOK_BYTE
            : $special{$t} ? TOK_CONTROL : $i < $base ? TOK_NORMAL : TOK_USER_DEFINED;
    }
    $self->kv('tokenizer.ggml.model', STR, $opt{model} // 'llama');
    $self->kv('tokenizer.ggml.tokens', ARR, [STR, \@tok]);
    $self->kv('tokenizer.ggml.scores', ARR, [F32, [(-1000) x @tok]]);
    $self->kv('tokenizer.ggml.token_type', ARR, [I32, \@type]);
    $self->special_vocab(scalar @tok, merges => 1);
    return scalar @tok;
}

# A SentencePiece tokenizer.model's pieces: [piece, score, type] by id (the protobuf ModelProto: field 1 the
# pieces, each piece 1 / score 2 / type 3, type NORMAL when absent)
sub spm_pieces {
    my ($path) = @_;
    my $b = _slurp($path);
    my $pos = 0;
    my $varint = sub {
        my ($s, $p) = @_;
        my ($v, $sh) = (0, 0);
        while (1) {
            my $c = ord substr $$s, $$p++, 1;
            $v |= ($c & 0x7f) << $sh;
            last unless $c & 0x80;
            $sh += 7;
        }
        return $v;
    };
    my $skip = sub {
        my ($s, $p, $wt) = @_;
        if ($wt == 0) { $varint->($s, $p) }
        elsif ($wt == 1) { $$p += 8 }
        elsif ($wt == 2) { $$p += $varint->($s, $p) }
        elsif ($wt == 5) { $$p += 4 }
        else { croak "$path: protobuf wire type $wt" }
    };
    my @pieces;
    while ($pos < length $b) {
        my $key = $varint->(\$b, \$pos);
        my ($f, $wt) = ($key >> 3, $key & 7);
        if ($f == 1 && $wt == 2) {
            my $len = $varint->(\$b, \$pos);
            my $m = substr $b, $pos, $len;
            $pos += $len;
            my ($q, $piece, $score, $type) = (0, '', 0, 1);
            while ($q < length $m) {
                my $k = $varint->(\$m, \$q);
                my ($g, $w) = ($k >> 3, $k & 7);
                if ($g == 1 && $w == 2) { my $l = $varint->(\$m, \$q); $piece = substr $m, $q, $l; $q += $l }
                elsif ($g == 2 && $w == 5) { $score = unpack 'f<', substr $m, $q, 4; $q += 4 }
                elsif ($g == 3 && $w == 0) { $type = $varint->(\$m, \$q) }
                else { $skip->(\$m, \$q, $w) }
            }
            utf8::decode($piece);
            push @pieces, [$piece, $score, $type];
        } else {
            $skip->(\$b, \$pos, $wt);
        }
    }
    return @pieces;
}

# convert_hf_to_gguf.py's _set_vocab_sentencepiece: tokenizer.model's pieces (types as SentencePiece reports them:
# UNKNOWN, CONTROL, UNUSED, BYTE, NORMAL otherwise), added_tokens.json's and tokenizer_config.json's added tokens
# over them; tokenizer.ggml.model "llama", pre "default"
sub vocab_sentencepiece {
    my ($self) = @_;
    my @p = spm_pieces(File::Spec->catfile($self->{dir}, 'tokenizer.model'));
    my $n = $self->find(qw(vocab_size_per_layer_input vocab_size)) || scalar @p;
    my @tok = map { "[PAD$_]" } 0 .. $n - 1;
    my @score = (-10000) x $n;
    my @type = (TOK_UNUSED) x $n;
    my %t = (2 => TOK_UNKNOWN, 3 => TOK_CONTROL, 5 => TOK_UNUSED, 6 => TOK_BYTE);
    for my $i (0 .. $#p) {
        last if $i >= $n;
        ($tok[$i], $score[$i], $type[$i]) = ($p[$i][0], $p[$i][1], $t{$p[$i][2]} // TOK_NORMAL);
    }
    if (my $at = json_file(File::Spec->catfile($self->{dir}, 'added_tokens.json'))) {
        for my $k (keys %$at) {
            next if $at->{$k} >= $n;
            ($tok[$at->{$k}], $score[$at->{$k}], $type[$at->{$k}]) = ($k, -1000, TOK_USER_DEFINED);
        }
    }
    my $tc = json_file(File::Spec->catfile($self->{dir}, 'tokenizer_config.json')) // {};
    for my $id (keys %{$tc->{added_tokens_decoder} // {}}) {
        next if $id >= $n;
        my $d = $tc->{added_tokens_decoder}{$id};
        my $c = $d->{content};
        if ($d->{special} || looks_special($c)) { $type[$id] = TOK_CONTROL }
        else { $c =~ s/\x{2581}/ /g; $type[$id] = TOK_USER_DEFINED }
        ($tok[$id], $score[$id]) = ($c, -1000);
    }
    $self->kv('tokenizer.ggml.model', STR, 'llama');
    $self->kv('tokenizer.ggml.pre', STR, 'default');
    $self->kv('tokenizer.ggml.tokens', ARR, [STR, \@tok]);
    $self->kv('tokenizer.ggml.scores', ARR, [F32, \@score]);
    $self->kv('tokenizer.ggml.token_type', ARR, [I32, \@type]);
    $self->special_vocab(scalar @tok);
    return scalar @tok;
}

# general pooling of a sentence-transformers model (modules.json's Pooling module): pooling_type
sub pooling_type {
    my ($self) = @_;
    my $mods = json_file(File::Spec->catfile($self->{dir}, 'modules.json')) or return;
    my ($m) = grep { $_->{type} =~ /Pooling$/ } @$mods or return;
    my $c = json_file(File::Spec->catfile($self->{dir}, $m->{path}, 'config.json')) or return;
    my %mode = (mean => 1, cls => 2, lasttoken => 3);
    my $t = $c->{pooling_mode_mean_tokens} ? 1 : $c->{pooling_mode_cls_token} ? 2 : $c->{pooling_mode_lasttoken} ? 3
          : $mode{$c->{pooling_mode} // ''} // croak "$m->{path}: pooling type not supported";
    $self->u32('pooling_type', $t);
}

# Added tokens that are control tokens though not marked special (convert_hf_to_gguf.py's does_token_look_special)
sub looks_special {
    my ($t) = @_;
    return 1 if $t eq '<pad>' || $t eq '<mask>' || $t eq '<2mass>' || $t eq '[@BOS@]';
    return 1 if $t =~ /^<\|.*\|>\z/s || $t =~ /^<\x{ff5c}.*\x{ff5c}>\z/s || $t =~ /^<unused.*>\z/s;
    return 0;
}

# tokenizer.ggml.pre: the pre-tokenizer llama.cpp names; convert_hf_to_gguf.py finds it by hashing a test string's
# tokens, here each architecture says (a converter option overrides)
sub tokenizer_pre { croak ref($_[0]) . ': no tokenizer_pre (give --pre)' }

# gguf-py's SpecialVocab: merges, special token ids, add_*_token, chat template, normalizer flags
sub special_vocab {
    my ($self, $n_vocab, %opt) = @_;
    my $dir = $self->{dir};
    my $tj = $self->{tokenizer} //= json_file(File::Spec->catfile($dir, 'tokenizer.json'));
    my $tc = json_file(File::Spec->catfile($dir, 'tokenizer_config.json'));
    my @types = qw(bos eos unk sep pad cls mask);
    my (%ids, %add, $merges, %norm, $template, $clean);
    my $set = sub {
        my ($typ, $id) = @_;
        return unless defined $id && !ref $id && !is_bool($id) && $id =~ /^\d+$/;
        return if exists $ids{$typ};
        $ids{$typ} = $id if $id < $n_vocab;
    };
    if ($tj) {
        if ($opt{merges} and my $m = $tj->{model}{merges}) {
            if (@$m && ref $m->[0] eq 'ARRAY') {
                $merges = [map { join ' ', map { s/ /\x{120}/gr } @$_ } @$m];
            } elsif (@$m) {
                $merges = $m;
            }
        }
        _normalizer($tj->{normalizer}, \%norm) if ref $tj->{normalizer} eq 'HASH';
        _pre_tokenizer($tj->{pre_tokenizer}, \%norm) if ref $tj->{pre_tokenizer} eq 'HASH';
        my $added = [$self->added_tokens];
        my ($bos, $cls, $eos, $sep) = map { $tc ? $tc->{"${_}_token"} : undef } qw(bos cls eos sep);
        if ($tc) {
            $tc->{bos_token} = $bos = $cls if !$bos && $cls;
            $tc->{eos_token} = $eos = $sep if !$eos && $sep;
        }
        if (my $pp = $tj->{post_processor}) {
            for my $p (@{$pp->{processors} // [$pp]}) {
                my $pt = $p->{type} // '';
                if ($pt eq 'ByteLevel') {
                    $add{bos} //= 0;
                    $add{eos} //= 0;
                } elsif ($pt eq 'RobertaProcessing') {
                    @add{qw(bos eos sep)} = (1, 1, 1);
                } elsif ($pt eq 'TemplateProcessing') {
                    my @single = @{$p->{single} // []};
                    my ($sfirst, $slast) = @single > 1
                        ? ($single[0]{SpecialToken}{id}, $single[-1]{SpecialToken}{id}) : (undef, undef);
                    if (@single > 1) {
                        my $first = $single[0]{SpecialToken}{id};
                        if (defined $first) {
                            if (!$tc) { $bos = $first }
                            elsif ($first ne ($bos // '') && $first ne ($cls // '')) {
                                $tc->{bos_token} = $bos = $first unless $bos;
                                $tc->{cls_token} = $cls = $first unless $cls;
                            }
                            $add{bos} = ($first eq ($bos // '') || $first eq ($cls // '')) ? 1 : 0;
                        }
                        my $last = $single[-1]{SpecialToken}{id};
                        if (defined $last) {
                            if (!$tc) { $eos = $last }
                            elsif ($last ne ($eos // '')) {
                                if (!grep { $_ eq 'eot' } @types) { push @types, 'eot'; $tc->{eot_token} = $eos }
                                elsif (!grep { $_ eq 'eom' } @types) { push @types, 'eom'; $tc->{eom_token} = $eos }
                                $tc->{eos_token} = $eos = $last;
                            }
                            $add{eos} = $last eq ($eos // '') ? 1 : 0;
                        }
                    }
                    if (my @pair = @{$p->{pair} // []}) {  # A [sep] [eos] B: add_sep_token
                        shift @pair if defined $sfirst && ($pair[0]{SpecialToken}{id} // '') eq $sfirst;
                        pop @pair if defined $slast && @pair && ($pair[-1]{SpecialToken}{id} // '') eq $slast;
                        if (@pair && ($pair[0]{Sequence}{id} // '') eq 'A' && ($pair[-1]{Sequence}{id} // '') eq 'B'
                            && (@pair = @pair[1 .. $#pair - 1])) {
                            my $sepc = $tc ? (ref $tc->{sep_token} eq 'HASH' ? $tc->{sep_token}{content} : $tc->{sep_token}) : undef;
                            my $eosc = ref $eos eq 'HASH' ? $eos->{content} : $eos;
                            my $is = sub { my $e = shift; defined $e && grep { defined $_ && $_ eq $e } $sepc, $eosc };
                            my $add_sep = 0;
                            my $e0 = $pair[0]{SpecialToken}{id};
                            $add_sep = 1 if $is->($e0) && !defined $slast;
                            $add_sep = 1 if @pair == 2 && $is->($pair[1]{SpecialToken}{id});
                            $add{sep} = $add_sep;
                            $tc->{sep_token} = $eosc if $add_sep && !defined $sepc && $tc;
                        }
                    }
                }
            }
        }
        if ($tc) {
            my $tpl = $tc->{chat_template};
            my $jinja = File::Spec->catfile($dir, 'chat_template.jinja');
            my $jsonf = File::Spec->catfile($dir, 'chat_template.json');
            if (-f $jinja) {
                $tpl = _slurp($jinja);
                utf8::decode($tpl);
            } elsif (-f $jsonf) {
                $tpl = json_file($jsonf)->{chat_template};
            }
            $template = $tpl if defined $tpl && !ref $tpl;
            $clean = $tc->{clean_up_tokenization_spaces} if is_bool($tc->{clean_up_tokenization_spaces});
            for my $typ (@types) {
                my $ae = $tc->{"add_${typ}_token"};
                $add{$typ} //= $ae ? 1 : 0 if is_bool($ae);
                my $e = $tc->{"${typ}_token"};
                my $content = ref $e eq 'HASH' ? $e->{content} : $e;
                next unless defined $content && !ref $content && !is_bool($content);
                my ($id) = map { $_->{id} } grep { $_->{content} eq $content } @$added;
                $set->($typ, $id);
            }
        }
    }
    for my $typ (@types) {
        my $id = $self->{config}{"${typ}_token_id"};
        $id = $self->{config}{text_config}{"${typ}_token_id"}
            if !defined $id && ref $self->{config}{text_config} eq 'HASH';
        $set->($typ, $id);
    }
    # The architecture's own: [type, token text] for slots still empty (convert_hf_to_gguf.py's _set_special_token
    # calls after SpecialVocab loads)
    for my $e (@{$opt{extra} // []}) {
        my ($typ, $text) = @$e;
        push @types, $typ unless grep { $_ eq $typ } @types;
        my ($id) = map { $_->{id} } grep { $_->{content} eq $text } $self->added_tokens;
        croak "special token $text is not an added token" unless defined $id;
        $set->($typ, $id);
    }
    $self->kv('tokenizer.ggml.merges', ARR, [STR, $merges]) if $merges && @$merges;
    my %key = (bos => 'bos_token_id', eos => 'eos_token_id', unk => 'unknown_token_id', sep => 'seperator_token_id',
               pad => 'padding_token_id', cls => 'cls_token_id', mask => 'mask_token_id', eot => 'eot_token_id',
               eom => 'eom_token_id');
    for my $typ (@types) { $self->kv("tokenizer.ggml.$key{$typ}", U32, $ids{$typ}) if exists $ids{$typ} }
    for my $typ (qw(bos eos unk sep pad cls mask eot eom)) {
        $self->kv("tokenizer.ggml.add_${typ}_token", BOOL, $add{$typ}) if exists $add{$typ};
    }
    $self->kv('tokenizer.chat_template', STR, $template) if defined $template;
    $self->kv('tokenizer.ggml.normalizer.nfc', BOOL, $norm{nfc}) if exists $norm{nfc};
    $self->kv('tokenizer.ggml.normalizer.lowercase', BOOL, $norm{lowercase}) if exists $norm{lowercase};
    $self->kv('tokenizer.ggml.normalizer.strip_accents', BOOL, $norm{strip_accents}) if exists $norm{strip_accents};
    $self->kv('tokenizer.ggml.clean_spaces', BOOL, $clean ? 1 : 0) if defined $clean;
    $self->kv('tokenizer.ggml.metaspace_prepend_scheme', STR, $norm{prepend}) if exists $norm{prepend};
}

sub _normalizer {
    my ($n, $o) = @_;
    my $t = $n->{type} // '';
    if ($t eq 'Lowercase') { $o->{lowercase} = 1 }
    elsif ($t eq 'NFC') { $o->{nfc} = 1 }
    elsif ($t eq 'StripAccents') { $o->{strip_accents} = 1 }
    elsif ($t eq 'BertNormalizer') {
        $o->{lowercase} = $n->{lowercase} ? 1 : 0 if exists $n->{lowercase};
        $o->{strip_accents} = $n->{strip_accents} ? 1 : 0 if exists $n->{strip_accents};
    } elsif ($t eq 'Sequence') { _normalizer($_, $o) for @{$n->{normalizers} // []} }
}

sub _pre_tokenizer {
    my ($p, $o) = @_;
    my $t = $p->{type} // '';
    if ($t eq 'Metaspace') { $o->{prepend} = $p->{prepend_scheme} if exists $p->{prepend_scheme} }
    elsif ($t eq 'Sequence') { _pre_tokenizer($_, $o) for @{$p->{pretokenizers} // []} }
}

# ---------------------------------------------------------------- tensors

# A checkpoint tensor by name (dies if absent)
sub t {
    my ($self, $name) = @_;
    return $self->{st}{tensors}{$name} // croak "$self->{dir}: no tensor $name";
}

# The GGUF tensor of one checkpoint tensor: ne is the reversed shape; rows are copied as they are
sub same {
    my ($self, $gname, $t, %o) = @_;
    return {name => $gname, ne => [reverse @{$t->{shape}}], parts => [[$t, 0, st_rows($t)]], %o};
}

# Several checkpoint tensors of one shape stacked on a new outer dimension (the experts of a layer)
sub stack {
    my ($self, $gname, $ts, %o) = @_;
    my @s = @{$ts->[0]{shape}};
    for (@$ts) { croak "$gname: stacking [@{$_->{shape}}] onto [@s]" unless "@{$_->{shape}}" eq "@s" }
    return {name => $gname, ne => [reverse(@s), scalar @$ts], parts => [map { [$_, 0, st_rows($_)] } @$ts], %o};
}

# Every checkpoint tensor under "$prefix." through a table of [regex on the rest, GGUF name (with $1.. from the
# match), options]: the GGUF tensors, in checkpoint order; a tensor no row matches dies (an unknown tensor is a new
# variant of the architecture, not something to drop). Rows with an undefined name are skipped (used elsewhere).
sub map_prefix {
    my ($self, $prefix, $gp, $table) = @_;
    my @s;
    my $T = $self->st->{tensors};
    for my $name (grep { index($_, "$prefix.") == 0 } @{$self->st->{order}}) {
        my $rest = substr $name, length($prefix) + 1;
        my $hit;
        for my $row (@$table) {
            my ($re, $g, %o) = @$row;
            my @m = $rest =~ /^$re\z/ or next;
            $hit = 1;
            last unless defined $g;
            my $n = ref $g eq 'CODE' ? $g->(@m) : $g =~ s/\$(\d)/$m[$1 - 1]/gr;
            push @s, $self->same("$gp$n", $T->{$name}, %o);
            last;
        }
        croak "$name: no GGUF name" unless $hit;
    }
    return @s;
}

# The common decoder-layer names (convert_hf_to_gguf.py's TensorNameMap for the Llama family)
our @LAYER = (
    [qr/input_layernorm\.weight/, 'attn_norm.weight'], [qr/post_attention_layernorm\.weight/, 'ffn_norm.weight'],
    [qr/self_attn\.([qkv])_proj\.(weight|bias)/, 'attn_$1.$2'], [qr/self_attn\.o_proj\.(weight|bias)/, 'attn_output.$1'],
    [qr/self_attn\.([qk])_norm\.weight/, 'attn_$1_norm.weight'],
    [qr/mlp\.(gate|up|down)_proj\.weight/, 'ffn_$1.weight'], [qr/mlp\.gate\.weight/, 'ffn_gate_inp.weight'],
    [qr/mlp\.shared_experts?\.(gate|up|down)_proj\.weight/, 'ffn_$1_shexp.weight'],
    [qr/mlp\.shared_expert_gate\.weight/, 'ffn_gate_inp_shexp.weight'],
    [qr/mlp\.experts\.\d+\.\w+\.weight/, undef], [qr/mlp\.experts\.(?:gate_up_proj|down_proj)/, undef],
);

# The routed experts of a layer as ffn_{gate,up,down}_exps: from one tensor an expert ("$p.E.gate_proj.weight") or
# fused ones ("$p.gate_up_proj" [E, 2 * ff, n] split in halves, "$p.down_proj" [E, n, ff])
sub experts {
    my ($self, $p, $b, $n, %o) = @_;
    my $T = $self->st->{tensors};
    my $w = $o{suffix} // '.weight';
    if (my $gu = $T->{"$p.gate_up_proj"} // $T->{"$p.gate_up_proj.weight"}) {
        my ($e, $ff2, $h) = @{$gu->{shape}};
        my $ff = $ff2 / 2;
        my $dn = $T->{"$p.down_proj"} // $self->t("$p.down_proj.weight");
        return ({name => "$b.ffn_gate_exps.weight", ne => [$h, $ff, $e], parts => [map { [$gu, $_ * $ff2, $ff] } 0 .. $e - 1]},
                {name => "$b.ffn_up_exps.weight", ne => [$h, $ff, $e], parts => [map { [$gu, $_ * $ff2 + $ff, $ff] } 0 .. $e - 1]},
                $self->same("$b.ffn_down_exps.weight", $dn));
    }
    my %gg = (gate => $o{gate} // 'gate_proj', up => $o{up} // 'up_proj', down => $o{down} // 'down_proj');
    my @s;
    for my $k (qw(gate up down)) {
        push @s, $self->stack("$b.ffn_${k}_exps.weight", [map { $self->t("$p.$_.$gg{$k}$w") } 0 .. $n - 1]);
    }
    return @s;
}

# The weight_scale_inv of an FP8 tensor, if it has one
sub fp8_scale {
    my ($self, $t) = @_;
    return undef unless $t->{dtype} eq 'F8_E4M3';
    (my $base = $t->{name}) =~ s/\.weight$//;
    return $self->{st}{tensors}{"$base.weight_scale_inv"} // $self->{st}{tensors}{"$base.scale"}
        // croak "$t->{name}: FP8 without $base.weight_scale_inv or $base.scale";
}

# Block scales as float32 bytes: F32 as they are, E8M0 (a power-of-two exponent, bias 127) expanded
sub _scale_f32 {
    my ($sc, $bytes) = @_;
    return $bytes if $sc->{dtype} eq 'F32';
    return pack 'f<*', map { 2 ** ($_ - 127) } unpack 'C*', $bytes if $sc->{dtype} eq 'F8_E8M0';
    croak "$sc->{name}: block scales of $sc->{dtype}";
}

# convert_hf_to_gguf.py's output type for a tensor in a BF16 file: F32 for 1-D tensors, norms, routers,
# convolutions and the architecture's exact ones (f32 => 1), BF16 otherwise
my @F32_NAMES = (qr/\.ffn_gate_inp\.weight$/, qr/\.ffn_gate_inp_shexp\.weight$/, qr/^position_embd\.weight$/,
                 qr/^token_types\.weight$/, qr/\.ssm_conv1d(?:_[qkv])?\.weight$/, qr/\.shortconv\.conv\.weight$/,
                 qr/\.time_mix_(?:first|w1|w2|decay_w1|decay_w2|lerp_fused)\.weight$/);
sub base_type {
    my ($self, $s) = @_;
    return 'f32' if $s->{f32} || @{$s->{ne}} <= 1 || $s->{name} =~ /_norm\.weight$/ || grep { $s->{name} =~ $_ } @F32_NAMES;
    return 'bf16';
}

# ---------------------------------------------------------------- writing

my %DT = (BF16 => 'bf16', F16 => 'f16', F32 => 'f32');

# convert($out, quant => ROCm::Inference::Quant | undef (BF16 as convert_hf_to_gguf.py --outtype bf16 writes),
# threads => N, progress => sub {...}, dry_run => 1)
sub convert {
    my ($self, $out, %a) = @_;
    my $q = $a{quant};
    $self->general;
    $self->text_params;
    $self->params;
    $self->kv('general.file_type', U32, $q ? $q->ftype : $self->ref_ftype);
    $self->kv('general.quantization_version', U32, 2);
    $self->vocab;
    my @specs = $self->tensors;
    # Every checkpoint tensor is used, or skipped on purpose (the architecture's skipped(): vision, a separate MTP)
    my %used;
    for my $s (@specs) {
        for my $t ((map { $_->[0] } @{$s->{parts} // []}), (map { @$_ } @{$s->{mxfp4} // []}), @{$s->{uses} // []},
                   @{$s->{hcat} // []}) {
            $used{$t->{name}} = 1;
            my $sc = $self->fp8_scale($t);
            $used{$sc->{name}} = 1 if $sc;
        }
    }
    my $skip = $self->skipped;
    my @unused = grep { !$used{$_} && !($skip && /$skip/) } @{$self->st->{order}};
    croak scalar(@unused) . " checkpoint tensors not converted: @unused[0 .. ($#unused < 9 ? $#unused : 9)]" if @unused;
    for my $s (@specs) {
        if ($s->{type_fixed}) {  # not the policy's (an I32 table)
            $s->{type} = gguf_type_id($s->{type_fixed});
            $s->{in_bytes} = 0;
            next;
        }
        my $base = $s->{base} // $self->base_type($s);
        my ($want, $why) = $q ? $q->plan($s->{name}, $s->{ne}, $base) : ($base);
        $s->{why} = $why;
        $s->{type} = gguf_type_id($want);
        my $t0 = $s->{parts} && $s->{parts}[0][0] || $s->{hcat} && $s->{hcat}[0];
        $s->{from} //= $t0 ? lc $t0->{dtype} : 'f32';
        $s->{in_bytes} = 0;
        $s->{in_bytes} += $_->[2] * ROCm::Inference::SafeTensors::st_row_bytes($_->[0]) for @{$s->{parts} // []};
        $s->{in_bytes} += $_->[0]{bytes} + $_->[1]{bytes} for @{$s->{mxfp4} // []};
        $s->{in_bytes} += $_->{bytes} for @{$s->{hcat} // []};
    }
    return write_file($out, $self->{kv}, \@specs, 32, %a, reader => $self);
}

# write_file($out, \@kv, \@specs, $alignment, %a): the GGUF, each spec (name, ne, type; rows from safetensors parts
# [tensor, first row, rows], a GGUF tensor gguf => $t, or data => sub { F32 bytes }) encoded and written in order,
# to $out.part, renamed when complete. %a: threads, progress => sub ($spec, $i, $n), dry_run, reader (the model,
# for safetensors parts), keep_cache (do not drop what is read and written from the page cache). Returns {kv,
# tensors, bytes}.
sub write_file {
    my ($out, $kv, $specs, $align, %a) = @_;
    my %seen;
    for my $s (@$specs) {
        croak "tensor $s->{name} twice" if $seen{$s->{name}}++;
        my $rows = 1;
        $rows *= $_ for @{$s->{ne}}[1 .. $#{$s->{ne}}];
        $s->{rows} = $rows;
        my ($tn, $blk, $tsz) = gguf_type_info($s->{type});
        croak "$s->{name}: $tn cannot hold rows of $s->{ne}[0]" unless $blk && $s->{ne}[0] % $blk == 0;
        $s->{bytes} = $s->{ne}[0] / $blk * $tsz * $rows;
    }
    my ($header, $data) = gguf_write_header($kv, $specs, $align);
    my $res = {kv => $kv, tensors => $specs, bytes => length($header) + $data};
    return $res if $a{dry_run};
    local $ROCm::Inference::SafeTensors::DROP_CACHE = !$a{keep_cache};
    local $ROCm::Inference::GGUF::DROP_CACHE = !$a{keep_cache};
    my $part = "$out.part";
    open my $fo, '>:raw', $part or croak "$part: $!";
    my ($off, $dropped) = (0, 0);
    my $wr = sub {
        _wr($fo, $_[0], $part);
        $off += length $_[0];
        if (!$a{keep_cache} && $off - $dropped >= (1 << 30)) {  # written back and dropped a GB at a time
            ROCm::Inference::drop_cache(fileno($fo), $dropped, $off - $dropped, 1);
            $dropped = $off;
        }
    };
    $wr->($header);
    my %fh;
    my $th = $a{threads} // 0;
    my $i = 0;
    for my $s (@$specs) {
        my $written = 0;
        if ($s->{data}) {  # computed (small) tensors: F32 values
            my $f = $s->{data}->();
            croak "$s->{name}: " . length($f) . " bytes of F32 for $s->{rows} rows of $s->{ne}[0]"
                unless length($f) == 4 * $s->{rows} * $s->{ne}[0];
            my $b = ROCm::Inference::quantize($s->{type}, 0, $f, $s->{rows}, $s->{ne}[0], $th);
            $wr->($b);
            $written += length $b;
        }
        if ($s->{raw}) {  # computed bytes already in the tensor's type (an I32 table)
            my $b = $s->{raw}->();
            $wr->($b);
            $written += length $b;
        }
        for my $e (@{$s->{mxfp4} // []}) {  # packed FP4 + E8M0 scales: MXFP4 blocks (re-encoded if asked)
            my ($w, $sc) = @$e;
            my $rows = ROCm::Inference::SafeTensors::st_rows($w);
            my $cols = 2 * $w->{shape}[-1];
            croak "$s->{name}: rows of $cols from $w->{name}, $s->{ne}[0] wanted" unless $cols == $s->{ne}[0];
            my $chunk = int((64 << 20) / $cols) || 1;
            for (my $r = 0; $r < $rows; $r += $chunk) {
                my $n = $r + $chunk > $rows ? $rows - $r : $chunk;
                my $b = ROCm::Inference::repack_mxfp4(ROCm::Inference::SafeTensors::st_read_rows(\%fh, $w, $r, $n),
                                                      ROCm::Inference::SafeTensors::st_read_rows(\%fh, $sc, $r, $n),
                                                      $n, $cols);
                $b = ROCm::Inference::quantize($s->{type}, gguf_type_id('mxfp4'), $b, $n, $cols, $th)
                    unless $s->{type} == gguf_type_id('mxfp4');
                $wr->($b);
                $written += length $b;
            }
        }
        if (my $t = $s->{gguf}) {  # a GGUF's tensor, re-encoded (or copied) ~64M values a pass
            my $fi = $fh{$t->{path}} //= do { open my $f, '<:raw', $t->{path} or croak "$t->{path}: $!"; $f };
            my $copy = $s->{type} == $t->{type};
            my $chunk = int((64 << 20) / $t->{ne}[0]) || 1;
            for (my $r = 0; $r < $s->{rows}; $r += $chunk) {
                my $n = $r + $chunk > $s->{rows} ? $s->{rows} - $r : $chunk;
                my $b = ROCm::Inference::GGUF::gguf_read_rows($fi, $t, $r, $n);
                $b = ROCm::Inference::quantize($s->{type}, $t->{type}, $b, $n, $t->{ne}[0], $th) unless $copy;
                $wr->($b);
                $written += length $b;
            }
        }
        for my $p (@{$s->{parts} // []}) {
            $written += $a{reader}->_write_part($wr, \%fh, $s, @$p, $th);
        }
        if (my $hc = $s->{hcat}) {  # rows of several tensors side by side: row r = each one's row r in turn
            my $src = $DT{$hc->[0]{dtype}} // croak "$s->{name}: dtype $hc->[0]{dtype}";
            croak "$s->{name}: tensors of different dtypes" if grep { $_->{dtype} ne $hc->[0]{dtype} } @$hc;
            my @rb = map { ROCm::Inference::SafeTensors::st_row_bytes($_) } @$hc;
            my $chunk = int((64 << 20) / $s->{ne}[0]) || 1;
            for (my $r = 0; $r < $s->{rows}; $r += $chunk) {
                my $n = $r + $chunk > $s->{rows} ? $s->{rows} - $r : $chunk;
                my @b = map { ROCm::Inference::SafeTensors::st_read_rows(\%fh, $_, $r, $n) } @$hc;
                my $rows = join '', map { my $i = $_; map { substr $b[$_], $i * $rb[$_], $rb[$_] } 0 .. $#b } 0 .. $n - 1;
                my $q = ROCm::Inference::quantize($s->{type}, gguf_type_id($src), $rows, $n, $s->{ne}[0], $th);
                $wr->($q);
                $written += length $q;
            }
        }
        croak "$s->{name}: wrote $written bytes, $s->{bytes} planned" unless $written == $s->{bytes};
        $wr->("\0" x ((-$written) % $align));
        $a{progress}->($s, ++$i, scalar @$specs) if $a{progress};
    }
    ROCm::Inference::drop_cache(fileno($fo), $dropped, $off - $dropped, 1) unless $a{keep_cache};
    close $fo or croak "$part: $!";
    rename $part, $out or croak "rename $part -> $out: $!";
    return $res;
}

# A GGUF re-encoded: its metadata (general.file_type and general.quantization_version set, moved to the end as
# llama.cpp's llama-quantize does; the split keys dropped: the output is one file), its tensors as the policy says
# (copied when the type stays). A split model is given by its first file.
sub requantize {
    my ($class, $in, $out, %a) = @_;
    my $q = $a{quant} or croak 'requantize: a policy (quant) is needed';
    my $g = ROCm::Inference::GGUF::gguf_read($in);
    my @kv = grep { $_->[0] !~ /^(?:general\.quantization_version|general\.file_type|split\.(?:no|count|tensors\.count))$/ }
        @{$g->{kv}};
    push @kv, ['general.quantization_version', U32, 2], ['general.file_type', U32, $q->ftype];
    my @specs;
    for my $t (@{$g->{tensors}}) {
        my ($src) = gguf_type_info($t->{type});
        # a float target decodes the quantized tensors too (a reference copy of a quantized file)
        my ($want, $why) = $q->plan($t->{name}, $t->{ne}, $src, decode => 1);
        $why = undef if $want eq $src;
        croak "$t->{name}: the engine cannot decode $src to re-encode it"
            unless $want eq $src || ROCm::Inference::quant_row_size($t->{type}, $t->{ne}[0]);
        push @specs, {name => $t->{name}, ne => [@{$t->{ne}}], type => gguf_type_id($want), gguf => $t, why => $why,
                      from => $src, in_bytes => $t->{bytes}};
    }
    my $res = write_file($out, \@kv, \@specs, $g->{alignment}, %a);
    $res->{gguf} = $g;
    return $res;
}

# Rows [r0, r0 + n) of checkpoint tensor $t encoded as the spec's type, ~64M values a pass
sub _write_part {
    my ($self, $wr, $fh, $s, $t, $r0, $n, $threads) = @_;
    my $cols = $t->{shape}[-1] // 1;
    croak "$s->{name}: rows of $cols from $t->{name}, $s->{ne}[0] wanted" unless $cols == $s->{ne}[0];
    my $written = 0;
    if (my $sc = $self->fp8_scale($t)) {
        my @ss = @{$sc->{shape}};
        my @ws = @{$t->{shape}};
        croak "$t->{name}: FP8 scales [@ss] for [@ws]" unless @ws == 2 && @ss == 2;
        my $br = int(($ws[0] + $ss[0] - 1) / $ss[0]);
        my $bc = int(($ws[1] + $ss[1] - 1) / $ss[1]);
        # the block size from the config when it divides; the shapes otherwise
        my $bs = ($self->{hp}{quantization_config} // {})->{weight_block_size};
        ($br, $bc) = @$bs if ref $bs eq 'ARRAY' && $ss[0] == int(($ws[0] + $bs->[0] - 1) / $bs->[0])
                                                && $ss[1] == int(($ws[1] + $bs->[1] - 1) / $bs->[1]);
        croak "$t->{name}: a part from row $r0 does not start a block of $br rows" if $r0 % $br;
        my $chunk = $br * (int((64 << 20) / ($cols * $br)) || 1);
        for (my $r = $r0; $r < $r0 + $n; $r += $chunk) {
            my $m = $r + $chunk > $r0 + $n ? $r0 + $n - $r : $chunk;
            my $w = st_read_rows($fh, $t, $r, $m);
            my $sr = _scale_f32($sc, st_read_rows($fh, $sc, $r / $br, int(($m + $br - 1) / $br)));
            my $b = ROCm::Inference::quantize_fp8_blocks($s->{type}, $w, $sr, $m, $cols, $br, $bc, $ss[1], $threads);
            $wr->($b);
            $written += length $b;
        }
        return $written;
    }
    my $src = $DT{$t->{dtype}} // croak "$t->{name}: dtype $t->{dtype} is not read";
    my $sid = gguf_type_id($src);
    my $chunk = int((64 << 20) / $cols) || 1;
    # col_blocks => [w, [source block of each output block]]: each row's columns moved w at a time
    my ($cw, $order) = @{$s->{col_blocks} // []};
    croak "$s->{name}: column blocks of $cw do not cover $cols" if $cw && $cw * @$order != $cols;
    for (my $r = $r0; $r < $r0 + $n; $r += $chunk) {
        my $m = $r + $chunk > $r0 + $n ? $r0 + $n - $r : $chunk;
        my $rows = st_read_rows($fh, $t, $r, $m);
        if ($cw) {
            my ($eb, $rb) = ($t->{esize} * $cw, $t->{esize} * $cols);
            $rows = join '', map { my $o = $_ * $rb; map { substr $rows, $o + $_ * $eb, $eb } @$order } 0 .. $m - 1;
        }
        my $b = ROCm::Inference::quantize($s->{type}, $sid, $rows, $m, $cols, $threads);
        $wr->($b);
        $written += length $b;
    }
    return $written;
}

sub _wr {
    my ($f, $s, $path) = @_;
    my $off = 0;
    while ($off < length $s) {
        my $w = syswrite($f, $s, length($s) - $off, $off);
        croak "$path: $!" unless defined $w;
        $off += $w;
    }
}

# F32 values of a whole (small) checkpoint tensor, as a list
sub floats {
    my ($self, $t) = @_;
    my %fh;
    my $src = $DT{$t->{dtype}} // croak "$t->{name}: dtype $t->{dtype}";
    my $f = ROCm::Inference::quantize(0, gguf_type_id($src), st_read_rows(\%fh, $t, 0, st_rows($t)), st_rows($t),
                                      $t->{shape}[-1] // 1);
    return unpack 'f<*', $f;
}

# Float values rounded to BF16 (to nearest, ties to even) and back, as a torch BF16 tensor holds the result of an
# operation on it
sub bf16_round {
    return map {
        my $u = unpack 'L<', pack 'f<', $_;
        $u = ($u & 0x7fffffff) > 0x7f800000 ? ($u | 0x00400000) & 0xffff0000 : ($u + 0x7fff + (($u >> 16) & 1)) & 0xffff0000;
        unpack 'f<', pack 'L<', $u;
    } @_;
}

# A small tensor computed from a checkpoint tensor's values: the GGUF tensor $gname of shape @ne, F32 values
# $fn->(@values) (convert_hf_to_gguf.py computes in F32: a BF16 tensor is widened first)
sub computed {
    my ($self, $gname, $t, $ne, $fn, %o) = @_;
    return {name => $gname, ne => $ne, uses => [$t], from => lc $t->{dtype}, %o, data => sub {
        return pack 'f<*', $fn->($self->floats($t));
    }};
}

sub vocab { $_[0]->vocab_gpt2 }
sub skipped { undef }  # a regex of checkpoint tensors the architecture leaves out on purpose
sub ref_ftype { 32 }  # convert_hf_to_gguf.py --outtype bf16's general.file_type
sub params { }

# ---------------------------------------------------------------- architectures

# A Llama-family decoder (model.embed_tokens, model.layers.N.{self_attn,mlp}, model.norm, lm_head) with the common
# names; routed experts stacked. The base of the Qwen and Llama packages.
package ROCm::Inference::Convert::Decoder;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert');

sub prefix { 'model' }  # where embed_tokens / layers / norm are
sub layer_table { \@ROCm::Inference::Convert::LAYER }
sub layer_extra { () }  # an architecture's own tensors of layer $l (after the table's)
sub n_experts { $_[0]->find(qw(num_local_experts num_experts n_routed_experts)) }

sub tensors {
    my ($self) = @_;
    my $T = $self->st->{tensors};
    my $P = $self->prefix;
    my @s = ($self->same('token_embd.weight', $self->t("$P.embed_tokens.weight")));
    for my $l (0 .. $self->{block_count} - 1) {
        my $p = "$P.layers.$l";
        push @s, $self->map_prefix($p, "blk.$l.", $self->layer_table);
        push @s, $self->experts("$p.mlp.experts", "blk.$l", $self->n_experts)
            if grep { /^\Q$p.mlp.experts.\E/ } keys %$T;
        push @s, $self->layer_extra($l, $p);
    }
    push @s, $self->same('output_norm.weight', $T->{"$P.norm.weight"}) if $T->{"$P.norm.weight"};
    push @s, $self->same('output.weight', $T->{'lm_head.weight'}) if $T->{'lm_head.weight'};
    return @s;
}

package ROCm::Inference::Convert::Qwen2;
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(Qwen2ForCausalLM));
sub arch { 'qwen2' }
sub tokenizer_pre { 'qwen2' }

package ROCm::Inference::Convert::Qwen3;
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(Qwen3ForCausalLM));
sub arch { 'qwen3' }
sub tokenizer_pre { 'qwen2' }

# Llama 3.x (convert_hf_to_gguf.py's LlamaModel): Q and K rows reordered from HF's rotate-half RoPE layout to the
# interleaved one llama.cpp's "llama" uses; the llama3 RoPE frequency factors as rope_freqs.
package ROCm::Inference::Convert::Llama;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(LlamaForCausalLM));
sub arch { 'llama' }
sub tokenizer_pre { 'llama-bpe' }

sub vocab {
    my ($self) = @_;
    croak 'a SentencePiece Llama (tokenizer.model) is not supported yet' if -f File::Spec->catfile($self->{dir}, 'tokenizer.model');
    my $tc = ROCm::Inference::Convert::json_file(File::Spec->catfile($self->{dir}, 'tokenizer_config.json')) // {};
    $self->kv('tokenizer.ggml.add_space_prefix', ROCm::Inference::Convert::BOOL, $tc->{add_prefix_space} ? 1 : 0)
        if defined $tc->{add_prefix_space};
    $self->vocab_gpt2;
}

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->u32('vocab_size', $hp->{vocab_size});
    $self->u32('rope.dimension_count', $hp->{head_dim} // $hp->{hidden_size} / $hp->{num_attention_heads});
}

# Rows of a head-major projection in the order llama.cpp's interleaved RoPE wants: within each head, rows
# [0, hd/2) and [hd/2, hd) interleaved
sub _unpermute {
    my ($self, $t, $n_head) = @_;
    my $rows = $t->{shape}[0];
    my $hd = $rows / $n_head;
    croak "$t->{name}: $rows rows are not $n_head heads of an even size" if $hd != int $hd || $hd % 2;
    return map { my $h = $_ * $hd; map { my $i = $_; map { [$t, $h + $_ * $hd / 2 + $i, 1] } 0, 1 } 0 .. $hd / 2 - 1 } 0 .. $n_head - 1;
}

sub layer_table {
    return [[qr/self_attn\.[qk]_proj\.(?:weight|bias)/, undef], @ROCm::Inference::Convert::LAYER];
}

sub layer_extra {
    my ($self, $l, $p) = @_;
    my $hp = $self->hp;
    my $T = $self->st->{tensors};
    my @s;
    for (['q', $hp->{num_attention_heads}], ['k', $hp->{num_key_value_heads} // $hp->{num_attention_heads}]) {
        for my $w (qw(weight bias)) {
            my $t = $T->{"$p.self_attn.$_->[0]_proj.$w"} or next;
            push @s, {name => "blk.$l.attn_$_->[0].$w", ne => [reverse @{$t->{shape}}], parts => [$self->_unpermute($t, $_->[1])]};
        }
    }
    return @s;
}

sub tensors {
    my ($self) = @_;
    my @s = $self->SUPER::tensors;
    my $rp = $self->{rope}{full_attention} // $self->{rope};
    return @s unless lc($rp->{rope_type} // '') eq 'llama3';
    # convert_hf_to_gguf.py's, in float32 as torch computes it
    my $hp = $self->hp;
    my $base = $rp->{rope_theta} // 10000;
    my $dim = $hp->{head_dim} // $hp->{hidden_size} / $hp->{num_attention_heads};
    my ($factor, $lo, $hi, $old) = map { $rp->{$_->[0]} // $_->[1] }
        [factor => 8], [low_freq_factor => 1], [high_freq_factor => 4], [original_max_position_embeddings => 8192];
    my $f = sub { unpack 'f<', pack 'f<', $_[0] };
    my @r;
    for (my $i = 0; $i < $dim; $i += 2) {
        my $freq = $f->(1 / $f->($f->($base) ** $f->($i / $dim)));
        my $wl = $f->(2 * 3.141592653589793 / $freq);
        if ($wl < $old / $hi) { push @r, 1 }
        elsif ($wl > $old / $lo) { push @r, $factor }
        else {
            my $smooth = $f->(($f->($old / $wl) - $lo) / ($hi - $lo));
            push @r, $f->(1 / $f->($f->((1 - $smooth) / $factor) + $smooth));
        }
    }
    return (@s, {name => 'rope_freqs.weight', ne => [scalar @r], data => sub { pack 'f<*', @r }});
}

# Nemotron-H / Nemotron 3 (convert_hf_to_gguf.py's NemotronHModel, nemotron_h_moe with experts): Mamba-2, attention
# and MoE layers by hybrid_override_pattern (M, *, E; - a dense MLP); the MTP head (mtp.layers.0: input fusion and
# attention, .1: MoE and the head norm) as one more block.
package ROCm::Inference::Convert::NemotronH;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert');
__PACKAGE__->register(qw(NemotronHForCausalLM));
sub arch { $_[0]{moe} ? 'nemotron_h_moe' : 'nemotron_h' }
sub tokenizer_pre { 'pixtral' }

sub init {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->{moe} = defined $hp->{num_experts_per_tok};
    # the layer kinds: transformers' layers_block_type, or the pattern string it is made from
    my %k = (M => 'mamba', '*' => 'attention', E => 'moe', '-' => 'mlp');
    my $pat = $hp->{layers_block_type} // [map { $k{$_} // croak "hybrid_override_pattern: $_" } split //, $hp->{hybrid_override_pattern} // ''];
    $pat = [map { {linear_attention => 'mamba', full_attention => 'attention'}->{$_} // $_ } @$pat];
    $self->{kinds} = $pat;
    $self->{block_count} = @$pat;
    $self->{n_main} = @$pat;
    $self->{mtp} = $self->{moe} && ($hp->{num_nextn_predict_layers} // 0) && grep { /^mtp\./ } @{$self->st->{order}};
    croak 'num_nextn_predict_layers > 1 is not supported' if $self->{mtp} && $hp->{num_nextn_predict_layers} > 1;
    if ($self->{mtp}) { push @$pat, 'mtp'; $self->{block_count}++ }
}

sub vocab {
    my ($self) = @_;
    my $hp = $self->hp;
    $hp->{vocab_size} = 8 * int(($hp->{vocab_size} + 7) / 8);  # Mamba-2's padding
    $self->vocab_gpt2;
}

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    my @k = @{$self->{kinds}};
    my $attn = sub { $_[0] eq 'attention' || $_[0] eq 'mtp' };
    my $mlp = sub { $_[0] eq ($self->{moe} ? 'moe' : 'mlp') || $_[0] eq 'mtp' };
    my $ff = $self->{moe} ? $hp->{moe_intermediate_size} : $hp->{intermediate_size};
    $self->arr('feed_forward_length', [map { $mlp->($_) ? $ff : 0 } @k], ROCm::Inference::Convert::I32);
    $self->arr('attention.head_count_kv', [map { $attn->($_) ? $hp->{num_key_value_heads} : 0 } @k], ROCm::Inference::Convert::I32);
    $self->u32('vocab_size', $hp->{vocab_size});
    my $hd = $hp->{head_dim} // $hp->{attention_head_dim};
    $self->u32('rope.dimension_count', $hp->{head_dim} // $hp->{hidden_size} / $hp->{num_attention_heads});
    $self->u32('ssm.conv_kernel', $hp->{conv_kernel});
    $self->u32('ssm.state_size', $hp->{ssm_state_size});
    $self->u32('ssm.group_count', $hp->{n_groups});
    $self->u32('ssm.inner_size', $hp->{mamba_num_heads} * $hp->{mamba_head_dim});
    $self->u32('ssm.time_step_rank', $hp->{mamba_num_heads});
    $self->bool('rope.scaling.finetuned', 0);
    $self->u32('context_length', 1 << 20);  # no RoPE: any length
    $self->u32('attention.key_length', $hd);
    $self->u32('attention.value_length', $hd);
    $self->f32('attention.layer_norm_epsilon', $hp->{layer_norm_epsilon}) if defined $hp->{layer_norm_epsilon};
    return unless $self->{moe};
    $self->u32('expert_used_count', $hp->{num_experts_per_tok});
    $self->u32('expert_feed_forward_length', $hp->{moe_intermediate_size});
    $self->u32('expert_shared_feed_forward_length', $hp->{moe_shared_expert_intermediate_size});
    $self->u32('expert_count', $hp->{n_routed_experts});
    $self->u32('expert_shared_count', $hp->{n_shared_experts});
    $self->bool('expert_weights_norm', $hp->{norm_topk_prob});
    $self->f32('expert_weights_scale', $hp->{routed_scaling_factor});
    $self->u32('expert_group_count', $hp->{n_group});
    $self->u32('moe_latent_size', $hp->{moe_latent_size}) if defined $hp->{moe_latent_size};
    $self->u32('nextn_predict_layers', $hp->{num_nextn_predict_layers}) if $self->{mtp};
}

sub _mixer {
    my ($self, $p, $b) = @_;
    my $hp = $self->hp;
    my $T = $self->st->{tensors};
    my @s = $self->map_prefix($p, "$b.", [
        [qr/in_proj\.weight/, 'ssm_in.weight'], [qr/out_proj\.weight/, 'ssm_out.weight'],
        [qr/conv1d\.bias/, 'ssm_conv1d.bias'], [qr/dt_bias/, 'ssm_dt.bias', f32 => 1],
        [qr/v_proj\.weight/, 'attn_v.weight'], [qr/o_proj\.weight/, 'attn_output.weight'], [qr/[qk]_proj\.weight/, undef],
        [qr/gate\.weight/, 'ffn_gate_inp.weight'], [qr/gate\.e_score_correction_bias/, 'exp_probs_b.bias'],
        [qr/shared_experts\.(up|down)_proj\.weight/, 'ffn_$1_shexp.weight'],
        [qr/(up|down)_proj\.weight/, 'ffn_$1.weight'],
        [qr/fc1_latent_proj\.weight/, 'ffn_latent_down.weight'], [qr/fc2_latent_proj\.weight/, 'ffn_latent_up.weight'],
        [qr/experts\.\d+\.\w+\.weight/, undef], [qr/(?:conv1d\.weight|A_log|D|norm\.weight)/, undef]]);
    for (['q', $hp->{num_attention_heads}], ['k', $hp->{num_key_value_heads}]) {  # as LlamaModel's (undo_permute)
        my $t = $T->{"$p.$_->[0]_proj.weight"} or next;
        push @s, {name => "$b.attn_$_->[0].weight", ne => [reverse @{$t->{shape}}],
                  parts => [ROCm::Inference::Convert::Llama::_unpermute($self, $t, $_->[1])]};
    }
    if (my $c = $T->{"$p.conv1d.weight"}) {  # [channels, 1, kernel] as [kernel, channels]
        push @s, {name => "$b.ssm_conv1d.weight", ne => [$c->{shape}[-1], $c->{shape}[0]], parts => [[$c, 0, $c->{shape}[0]]]};
    }
    if (my $a = $T->{"$p.A_log"}) {
        push @s, $self->computed("$b.ssm_a", $a, [1, $a->{shape}[0]], sub { map { -ROCm::Inference::Convert::Qwen35::exp_f32($_) } @_ },
                                 f32 => 1);
    }
    if (my $d = $T->{"$p.D"}) {
        push @s, $self->computed("$b.ssm_d", $d, [1, $d->{shape}[0]], sub { @_ }, f32 => 1);
    }
    if (my $n = $T->{"$p.norm.weight"}) {  # one group's widths a row
        push @s, $self->computed("$b.ssm_norm.weight", $n, [$n->{shape}[0] / $hp->{n_groups}, $hp->{n_groups}], sub { @_ });
    }
    if (grep { /^\Q$p.experts.\E/ } keys %$T) {
        my $ne = $hp->{n_routed_experts};
        for my $w (qw(up down)) {
            push @s, $self->stack("$b.ffn_${w}_exps.weight", [map { $self->t("$p.experts.$_.${w}_proj.weight") } 0 .. $ne - 1]);
        }
    }
    return @s;
}

sub tensors {
    my ($self) = @_;
    my $T = $self->st->{tensors};
    my @s = ($self->same('token_embd.weight', $self->t('backbone.embeddings.weight')),
             $self->same('output_norm.weight', $self->t('backbone.norm_f.weight')));
    push @s, $self->same('output.weight', $T->{'lm_head.weight'}) if $T->{'lm_head.weight'};
    for my $l (0 .. $self->{n_main} - 1) {
        my $p = "backbone.layers.$l";
        push @s, $self->same("blk.$l.attn_norm.weight", $self->t("$p.norm.weight")), $self->_mixer("$p.mixer", "blk.$l");
    }
    if ($self->{mtp}) {
        my $b = "blk.$self->{n_main}";
        my %special = ('mtp.layers.0.enorm.weight' => 'nextn.enorm.weight', 'mtp.layers.0.hnorm.weight' => 'nextn.hnorm.weight',
                       'mtp.layers.0.eh_proj.weight' => 'nextn.eh_proj.weight', 'mtp.layers.0.norm.weight' => 'attn_norm.weight',
                       'mtp.layers.1.norm.weight' => 'post_attention_norm.weight',
                       'mtp.layers.1.final_layernorm.weight' => 'nextn.shared_head_norm.weight');
        for my $n (grep { /^mtp\.layers\.[01]\./ && !/\.mixer\./ } @{$self->st->{order}}) {
            push @s, $self->same("$b." . ($special{$n} // croak "$n: no GGUF name"), $T->{$n});
        }
        # convert_hf_to_gguf.py keeps the MTP block's router in BF16 (its F32 rule matches the trunk's block ids)
        push @s, map { $_->{name} eq "$b.ffn_gate_inp.weight" ? {%$_, base => 'bf16'} : $_ } $self->_mixer("mtp.layers.$_.mixer", $b)
            for 0, 1;
    }
    return @s;
}

# Gemma 4 (convert_hf_to_gguf.py's Gemma4Model; gemma-4-26B-A4B's layout): sliding and full attention layers
# (full ones K = V, wider heads, proportional RoPE through rope_freqs), a dense MLP and MoE in every layer, the
# router's input scale and the per-expert output scale; norms as they are (no 1 + w in Gemma 4).
package ROCm::Inference::Convert::Gemma4;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(Gemma4ForConditionalGeneration Gemma4ForCausalLM));
sub arch { 'gemma4' }
sub prefix { 'model.language_model' }
sub skipped { qr/^model\.(?:vision_tower|embed_vision|audio_tower|embed_audio)\./ }  # the mmproj's

sub vocab {
    my ($self) = @_;
    $self->{n_vocab} = $self->vocab_llama_hf(model => 'gemma4',
        visible => ['<|channel>', '<channel|>', '<|tool_call>', '<tool_call|>', '<|tool_response>', '<tool_response|>', '<|"|>']);
    $self->kv('tokenizer.ggml.add_space_prefix', ROCm::Inference::Convert::BOOL, 0);
    $self->kv('tokenizer.ggml.add_bos_token', ROCm::Inference::Convert::BOOL, 1);
}

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    my @lt = @{$hp->{layer_types}};
    my @swa = map { $_ eq 'sliding_attention' ? 1 : 0 } @lt;
    $self->u32('context_length', $hp->{max_position_embeddings} // 131072);
    $self->f32('attention.layer_norm_rms_epsilon', $hp->{rms_norm_eps} // 1e-6);
    $self->f32('final_logit_softcapping', $hp->{final_logit_softcapping}) if $hp->{final_logit_softcapping};
    $self->u32('attention.sliding_window', $hp->{sliding_window}) if ($hp->{sliding_window_pattern} // 0) != 1;
    $self->u32('attention.shared_kv_layers', $hp->{num_kv_shared_layers});
    $self->u32('embedding_length_per_layer_input', $hp->{hidden_size_per_layer_input} || 0);
    $self->arr('attention.sliding_window_pattern', \@swa, ROCm::Inference::Convert::BOOL);
    my $hdf = $hp->{global_head_dim} // croak 'global_head_dim is needed';
    my $hds = $hp->{head_dim};
    $self->u32('attention.key_length', $hdf);
    $self->u32('attention.value_length', $hdf);
    $self->u32('attention.key_length_swa', $hds);
    $self->u32('attention.value_length_swa', $hds);
    my $v;
    $self->u32('expert_feed_forward_length', $v) if defined($v = $self->find(qw(expert_intermediate_size moe_intermediate_size)));
    croak 'use_double_wide_mlp is not supported' if $hp->{use_double_wide_mlp};
    if (defined(my $kvf = $hp->{num_global_key_value_heads})) {
        $self->arr('attention.head_count_kv', [map { $_ ? $hp->{num_key_value_heads} : $kvf } @swa], ROCm::Inference::Convert::I32);
    }
    $self->u32('rope.dimension_count', $hdf);
    $self->u32('rope.dimension_count_swa', int($hds * ($self->{rope}{sliding_attention}{partial_rotary_factor} // 1)));
}

sub layer_table {
    return [[qr/input_layernorm\.weight/, 'attn_norm.weight'], [qr/post_attention_layernorm\.weight/, 'post_attention_norm.weight'],
            [qr/pre_feedforward_layernorm\.weight/, 'ffn_norm.weight'],
            [qr/pre_feedforward_layernorm_2\.weight/, 'pre_ffw_norm_2.weight'],
            [qr/post_feedforward_layernorm(_[12])?\.weight/, sub { 'post_ffw_norm' . ($_[0] // '') . '.weight' }],
            [qr/self_attn\.([qkv])_proj\.weight/, 'attn_$1.weight'], [qr/self_attn\.o_proj\.weight/, 'attn_output.weight'],
            [qr/self_attn\.([qk])_norm\.weight/, 'attn_$1_norm.weight'],
            [qr/mlp\.(gate|up|down)_proj\.weight/, 'ffn_$1.weight'],
            [qr/experts\.gate_up_proj/, 'ffn_gate_up_exps.weight'], [qr/experts\.down_proj/, 'ffn_down_exps.weight'],
            [qr/router\.proj\.weight/, 'ffn_gate_inp.weight'], [qr/router\.scale/, 'ffn_gate_inp.scale', f32 => 1],
            [qr/router\.per_expert_scale/, 'ffn_down_exps.scale', f32 => 1],
            [qr/layer_scalar/, 'layer_output_scale.weight', f32 => 1]];
}

sub tensors {
    my ($self) = @_;
    my $P = $self->prefix;
    my @s = map {  # the token embeddings without rows past the tokenizer's tokens
        my $s = $_;
        if ($s->{name} eq 'token_embd.weight' && $s->{ne}[1] > $self->{n_vocab}) {
            $s = {%$s, ne => [$s->{ne}[0], $self->{n_vocab}], parts => [[$s->{parts}[0][0], 0, $self->{n_vocab}]]};
        }
        $s;
    } $self->SUPER::tensors;
    my $rf = $self->{rope}{full_attention} // croak 'rope_parameters.full_attention is needed';
    croak "full attention RoPE $rf->{rope_type}: proportional expected" unless ($rf->{rope_type} // '') eq 'proportional';
    my $hdf = $self->hp->{global_head_dim};
    my $nrot = int($hdf * $rf->{partial_rotary_factor} / 2);
    # the dimensions past the rotated ones: a factor that makes their frequency 0 (for the full attention layers)
    my @f = ((1) x $nrot, (1e30) x (int($hdf / 2) - $nrot));
    return (@s, {name => 'rope_freqs.weight', ne => [scalar @f], data => sub { pack 'f<*', @f }});
}

# EmbeddingGemma (convert_hf_to_gguf.py's EmbeddingGemma, a Gemma 3 text encoder as gemma-embedding): norms as 1 + w,
# bidirectional attention, sentence-transformers pooling. The Dense modules (2_Dense, 3_Dense) are left out, as
# convert_hf_to_gguf.py leaves them without --sentence-transformers-dense-modules.
package ROCm::Inference::Convert::GemmaEmbedding;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(Gemma3TextModel));
sub arch { 'gemma-embedding' }
sub prefix { '' }

sub init {
    my ($self) = @_;
    # the checkpoint's names have no model. prefix
    $self->{P} = exists $self->st->{tensors}{'model.embed_tokens.weight'} ? 'model' : '';
}

sub vocab {
    my ($self) = @_;
    $self->{n_vocab} = -f File::Spec->catfile($self->{dir}, 'tokenizer.model') ? $self->vocab_sentencepiece : $self->vocab_gpt2;
    $self->kv('tokenizer.ggml.add_space_prefix', ROCm::Inference::Convert::BOOL, 0);
}

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->u32('context_length', $hp->{max_position_embeddings} // 131072);
    $self->u32('attention.head_count', $hp->{num_attention_heads} // 8);
    $self->f32('attention.layer_norm_rms_epsilon', $hp->{rms_norm_eps} // 1e-6);
    $self->u32('attention.key_length', $hp->{head_dim} // 256);
    $self->u32('attention.value_length', $hp->{head_dim} // 256);
    croak 'attn_logit_softcapping is not supported' if defined $hp->{attn_logit_softcapping};
    $self->f32('final_logit_softcapping', $hp->{final_logit_softcapping}) if $hp->{final_logit_softcapping};
    $self->u32('attention.sliding_window', $hp->{sliding_window} // croak 'sliding_window is needed');
    $self->u32('attention.head_count_kv', $hp->{num_key_value_heads} // 4);
    $self->pooling_type;
}

sub tensors {
    my ($self) = @_;
    my $T = $self->st->{tensors};
    my $p0 = $self->{P} eq '' ? '' : "$self->{P}.";
    my $e = $self->t("${p0}embed_tokens.weight");
    my $nv = $self->{n_vocab} < $e->{shape}[0] ? $self->{n_vocab} : $e->{shape}[0];
    my @s = ({name => 'token_embd.weight', ne => [$e->{shape}[1], $nv], parts => [[$e, 0, $nv]]});
    my %norm = (input_layernorm => 'attn_norm', post_attention_layernorm => 'post_attention_norm',
                pre_feedforward_layernorm => 'ffn_norm', post_feedforward_layernorm => 'post_ffw_norm',
                'self_attn.q_norm' => 'attn_q_norm', 'self_attn.k_norm' => 'attn_k_norm');
    my $re = join '|', map { quotemeta } keys %norm;
    for my $l (0 .. $self->{block_count} - 1) {
        my $p = "${p0}layers.$l";
        push @s, $self->map_prefix($p, "blk.$l.", [[qr/(?:$re)\.weight/, undef],
            [qr/self_attn\.([qkv])_proj\.weight/, 'attn_$1.weight'], [qr/self_attn\.o_proj\.weight/, 'attn_output.weight'],
            [qr/mlp\.(gate|up|down)_proj\.weight/, 'ffn_$1.weight']]);
        push @s, map { $self->ROCm::Inference::Convert::Qwen35::_norm1("blk.$l.$norm{$_}.weight", $self->t("$p.$_.weight")) }
            sort keys %norm;
    }
    push @s, $self->ROCm::Inference::Convert::Qwen35::_norm1('output_norm.weight', $self->t("${p0}norm.weight"));
    return @s;
}

package ROCm::Inference::Convert::Qwen3Moe;
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(Qwen3MoeForCausalLM));
sub arch { 'qwen3moe' }
sub tokenizer_pre { 'qwen2' }

sub params {
    my ($self) = @_;
    my $v;
    $self->u32('expert_feed_forward_length', $v) if defined($v = $self->hp->{moe_intermediate_size});
    $self->u32('expert_shared_feed_forward_length', $v) if defined($v = $self->hp->{shared_expert_intermediate_size});
}

# Qwen3.5 / 3.6 / 3.8 (convert_hf_to_gguf.py's Qwen3_5TextModel / Qwen3_5MoeTextModel): Gated DeltaNet layers and
# gated full attention, the MTP block as one more layer. The norms are stored as w where the model computes 1 + w
# (written as 1 + w); the DeltaNet's V heads, grouped by K head in the checkpoint, are written tiled (the order
# ggml's broadcast repeats K heads in).
package ROCm::Inference::Convert::Qwen35;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert::Decoder');
__PACKAGE__->register(qw(Qwen3_5ForConditionalGeneration Qwen3_5ForCausalLM));
sub arch { 'qwen35' }
sub tokenizer_pre { 'qwen35' }
sub skipped { qr/^model\.visual\./ }  # the vision encoder: a separate (mmproj) file
sub prefix { 'model.language_model' }

sub init {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->{n_main} = $hp->{num_hidden_layers};
    $self->{n_mtp} = $hp->{mtp_num_hidden_layers} // 0;
    $self->{n_mtp} = 0 unless grep { /^mtp\./ } @{$self->st->{order}};
    $self->{block_count} += $self->{n_mtp};
    $self->{rope}{mrope_section} //= [11, 11, 10];
}

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->ROCm::Inference::Convert::Qwen3Moe::params;
    $self->u32('nextn_predict_layers', $self->{n_mtp}) if $self->{n_mtp};
    $self->u32('ssm.conv_kernel', $hp->{linear_conv_kernel_dim});
    $self->u32('ssm.state_size', $hp->{linear_key_head_dim});
    $self->u32('ssm.group_count', $hp->{linear_num_key_heads});
    $self->u32('ssm.time_step_rank', $hp->{linear_num_value_heads});
    $self->u32('ssm.inner_size', $hp->{linear_value_head_dim} * $hp->{linear_num_value_heads});
    if (my $lt = $hp->{layer_types}) {
        croak "layer_types has " . @$lt . " entries, num_hidden_layers $self->{n_main}" unless @$lt == $self->{n_main};
        $self->arr('attention.recurrent_layers', [(map { $_ eq 'linear_attention' ? 1 : 0 } @$lt), (0) x $self->{n_mtp}],
                   ROCm::Inference::Convert::BOOL);
    }
    $self->u32('full_attention_interval', $hp->{full_attention_interval} // 4);
    my $hd = $hp->{head_dim} // $hp->{hidden_size} / $hp->{num_attention_heads};
    $self->u32('rope.dimension_count', int($hd * ($self->{rope}{partial_rotary_factor} // 0.25)));
}

# The output order of V heads: tiled (for each head of a K group, the K groups in turn) from grouped
sub _v_order {
    my ($self) = @_;
    my $hp = $self->hp;
    my ($nk, $nv) = ($hp->{linear_num_key_heads}, $hp->{linear_num_value_heads});
    my $r = $nv / $nk;
    return map { my $j = $_; map { $_ * $r + $j } 0 .. $nk - 1 } 0 .. $r - 1;
}

sub layer_table {
    my ($self) = @_;
    my @t = grep { "$_->[0]" !~ /post_attention_layernorm|_norm\\\.weight|input_layernorm/ } @ROCm::Inference::Convert::LAYER;
    return [@t, [qr/(?:input_layernorm|post_attention_layernorm|self_attn\.[qk]_norm)\.weight/, undef],
            [qr/linear_attn\.(?:A_log|dt_bias|conv1d\.weight|in_proj_\w+\.weight|out_proj\.weight)/, undef],
            [qr/linear_attn\.norm\.weight/, 'ssm_norm.weight']];
}

sub layer_extra {
    my ($self, $l, $p) = @_;
    my $hp = $self->hp;
    my $T = $self->st->{tensors};
    my $b = "blk.$l";
    my @s;
    my %norm = (input_layernorm => 'attn_norm', post_attention_layernorm => 'post_attention_norm',
                'self_attn.q_norm' => 'attn_q_norm', 'self_attn.k_norm' => 'attn_k_norm');
    for my $k (sort keys %norm) {
        my $t = $T->{"$p.$k.weight"} or next;
        push @s, $self->computed("$b.$norm{$k}.weight", $t, [st_rows_cols($t)], sub { map { $_ + 1 } @_ });
    }
    my $a = "$p.linear_attn";
    if (my $qkv = $T->{"$a.in_proj_qkv.weight"}) {
        my ($nk, $nv, $dk, $dv) = @$hp{qw(linear_num_key_heads linear_num_value_heads linear_key_head_dim linear_value_head_dim)};
        my @vo = $self->_v_order;
        my $qk = 2 * $nk * $dk;
        # rows of head blocks in the tiled order: [tensor, first row of the V part, rows a head]
        my $heads = sub { my ($t, $at, $w) = @_; map { [$t, $at + $_ * $w, $w] } @vo };
        push @s, {name => "$b.attn_qkv.weight", ne => [$qkv->{shape}[1], $qkv->{shape}[0]],
                  parts => [[$qkv, 0, $qk], $heads->($qkv, $qk, $dv)]};
        my $z = $self->t("$a.in_proj_z.weight");
        push @s, {name => "$b.attn_gate.weight", ne => [reverse @{$z->{shape}}], parts => [$heads->($z, 0, $dv)]};
        for (['a', 'ssm_alpha'], ['b', 'ssm_beta']) {
            my $w = $self->t("$a.in_proj_$_->[0].weight");
            push @s, {name => "$b.$_->[1].weight", ne => [reverse @{$w->{shape}}], parts => [$heads->($w, 0, 1)]};
        }
        my $c = $self->t("$a.conv1d.weight");  # [channels, 1, kernel] as [kernel, channels]
        push @s, {name => "$b.ssm_conv1d.weight", ne => [$c->{shape}[-1], $c->{shape}[0]],
                  parts => [[$c, 0, $qk], $heads->($c, $qk, $dv)]};
        my $o = $self->t("$a.out_proj.weight");
        push @s, {%{$self->same("$b.ssm_out.weight", $o)}, col_blocks => [$dv, \@vo]};
        my $al = $self->t("$a.A_log");
        push @s, $self->computed("$b.ssm_a", $al, [$nv], sub { my @v = @_; map { -exp_f32($v[$_]) } @vo });
        my $dt = $self->t("$a.dt_bias");
        push @s, $self->computed("$b.ssm_dt.bias", $dt, [$nv], sub { my @v = @_; @v[@vo] });
    }
    return @s;
}

# expf as torch computes it (the XS exp_neg_f32 negated back)
sub exp_f32 { -unpack 'f<', ROCm::Inference::exp_neg_f32(pack 'f<', $_[0]) }
sub st_rows_cols { my ($t) = @_; return reverse @{$t->{shape}} }

sub tensors {
    my ($self) = @_;
    my $T = $self->st->{tensors};
    my $P = $self->prefix;
    my @s = grep { $_->{name} ne 'output_norm.weight' } $self->SUPER::tensors;
    if (my $n = $T->{"$P.norm.weight"}) {
        push @s, $self->computed('output_norm.weight', $n, [$n->{shape}[0]], sub { map { $_ + 1 } @_ });
    }
    for my $m (0 .. $self->{n_mtp} - 1) {  # mtp.layers.M: block n_main + M, its head tensors on the first
        my $l = $self->{n_main} + $m;
        my $p = "mtp.layers.$m";
        push @s, $self->map_prefix($p, "blk.$l.", $self->layer_table), $self->layer_extra($l, $p);
        push @s, $self->experts("$p.mlp.experts", "blk.$l", $self->n_experts) if grep { /^\Q$p.mlp.experts.\E/ } keys %$T;
    }
    if ($self->{n_mtp}) {
        my $b = "blk.$self->{n_main}";
        push @s, $self->same("$b.nextn.eh_proj.weight", $T->{'mtp.fc.weight'}) if $T->{'mtp.fc.weight'};
        for (['pre_fc_norm_embedding', 'enorm'], ['pre_fc_norm_hidden', 'hnorm'], ['norm', 'shared_head_norm']) {
            my $t = $T->{"mtp.$_->[0].weight"} or next;
            push @s, $self->computed("$b.nextn.$_->[1].weight", $t, [$t->{shape}[0]], sub { map { $_ + 1 } @_ });
        }
        push @s, $self->mtp_extra($b);
    }
    return @s;
}

sub mtp_extra { () }

sub _norm1 {  # a norm stored as w, applied as 1 + w
    my ($self, $gname, $t) = @_;
    return $self->computed($gname, $t, [$t->{shape}[0]], sub { map { $_ + 1 } @_ });
}

# Qwen3.8-Flash-Next (convert_hf_to_gguf.py's Qwen4ExpTextModel): Qwen3.5's gated DeltaNet with hyper-connections in
# place of the layer norms, QSA sparse attention (an indexer) on the full-attention layers, PLE n-gram hash
# embeddings on one layer. Matmuls reading the same input are written as one weight: [hc down | hc inject],
# DeltaNet [qkv | z] and [beta | alpha].
# The vision tower as an mmproj GGUF (rocm-convert --mmproj): every model.visual tensor, F32 (a BF16 widened, exact),
# under v.* names (v.blk.N.ln1 / ln2 / attn_qkv / attn_out / ffn_up / ffn_down, v.patch_embd, v.position_embd,
# v.post_ln, v.merger.fc1 / fc2) with the vision hyperparameters under qwen3_5_vision.*. The patch embedding (a Conv3d
# [out, in, T, H, W]) is flattened to [out, in*T*H*W] in the order the pixel rows come in (channel, temporal, row,
# column), so the engine applies it as a matmul. No quantization: the vision tower is small, and the engine's check
# against transformers is exact.
sub vision_gname {
    my ($n) = @_;
    return 'v.patch_embd.weight' if $n eq 'model.visual.patch_embed.proj.weight';
    return "v.patch_embd.$1" if $n =~ /^model\.visual\.patch_embed\.proj\.(bias)$/;
    return 'v.position_embd.weight' if $n eq 'model.visual.pos_embed.weight';
    return "v.blk.$1.ln$2.$3" if $n =~ /^model\.visual\.blocks\.(\d+)\.norm([12])\.(weight|bias)$/;
    return "v.blk.$1.attn_qkv.$2" if $n =~ /^model\.visual\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)$/;
    return "v.blk.$1.attn_out.$2" if $n =~ /^model\.visual\.blocks\.(\d+)\.attn\.proj\.(weight|bias)$/;
    return "v.blk.$1.ffn_up.$2" if $n =~ /^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.(weight|bias)$/;
    return "v.blk.$1.ffn_down.$2" if $n =~ /^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.(weight|bias)$/;
    return "v.post_ln.$1" if $n =~ /^model\.visual\.merger\.norm\.(weight|bias)$/;
    return "v.merger.fc1.$1" if $n =~ /^model\.visual\.merger\.linear_fc1\.(weight|bias)$/;
    return "v.merger.fc2.$1" if $n =~ /^model\.visual\.merger\.linear_fc2\.(weight|bias)$/;
    return;
}

sub convert_mmproj {
    my ($self, $out, %a) = @_;
    my $vc = $self->{config}{vision_config} // croak "$self->{dir}: no vision_config";
    my $pp = ROCm::Inference::Convert::json_file(File::Spec->catfile($self->{dir}, 'preprocessor_config.json')) // {};
    my @kv = (
        ['general.architecture', ROCm::Inference::Convert::STR, 'qwen3_5_vision'],
        ['general.type', ROCm::Inference::Convert::STR, 'mmproj'],
        ['general.file_type', ROCm::Inference::Convert::U32, 0],
        ['qwen3_5_vision.block_count', ROCm::Inference::Convert::U32, $vc->{depth} // croak 'vision_config: no depth'],
        ['qwen3_5_vision.embedding_length', ROCm::Inference::Convert::U32, $vc->{hidden_size}],
        ['qwen3_5_vision.feed_forward_length', ROCm::Inference::Convert::U32, $vc->{intermediate_size}],
        ['qwen3_5_vision.attention.head_count', ROCm::Inference::Convert::U32, $vc->{num_heads}],
        ['qwen3_5_vision.patch_size', ROCm::Inference::Convert::U32, $vc->{patch_size}],
        ['qwen3_5_vision.temporal_patch_size', ROCm::Inference::Convert::U32, $vc->{temporal_patch_size}],
        ['qwen3_5_vision.spatial_merge_size', ROCm::Inference::Convert::U32, $vc->{spatial_merge_size}],
        ['qwen3_5_vision.projection_dim', ROCm::Inference::Convert::U32, $vc->{out_hidden_size}],
        ['qwen3_5_vision.num_position_embeddings', ROCm::Inference::Convert::U32, $vc->{num_position_embeddings}],
        ['qwen3_5_vision.hidden_act', ROCm::Inference::Convert::STR, $vc->{hidden_act} // 'gelu_pytorch_tanh'],
        ['qwen3_5_vision.layer_norm_epsilon', ROCm::Inference::Convert::F32, 1e-6],
        ['qwen3_5_vision.image_mean', ROCm::Inference::Convert::ARR, [ROCm::Inference::Convert::F32, [map { 0 + $_ } @{ $pp->{image_mean} // [0.5, 0.5, 0.5] }]]],
        ['qwen3_5_vision.image_std', ROCm::Inference::Convert::ARR, [ROCm::Inference::Convert::F32, [map { 0 + $_ } @{ $pp->{image_std} // [0.5, 0.5, 0.5] }]]],
    );
    my @specs;
    for my $name (@{$self->st->{order}}) {
        next unless $name =~ /^model\.visual\./;
        my $g = vision_gname($name) // croak "$name: no mmproj tensor name (a vision tensor the engine does not know)";
        my $t = $self->t($name);
        my $ne = $name eq 'model.visual.patch_embed.proj.weight'
            ? [reduce_mul(@{$t->{shape}}[1 .. $#{$t->{shape}}]), $t->{shape}[0]]
            : [reverse @{$t->{shape}}];
        push @specs, {name => $g, ne => $ne, f32 => 1, type => ROCm::Inference::Convert::gguf_type_id('f32'), from => lc $t->{dtype},
                      in_bytes => $t->{bytes}, uses => [$t], data => sub { pack 'f<*', $self->floats($t) }};
    }
    croak 'no model.visual tensors in ' . $self->{dir} unless @specs;
    return ROCm::Inference::Convert::write_file($out, \@kv, \@specs, 32, %a, reader => $self);
}

sub reduce_mul { my $n = 1; $n *= $_ for @_; return $n }

package ROCm::Inference::Convert::Qwen4Exp;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert::Qwen35');
__PACKAGE__->register(qw(Qwen4ExpForConditionalGeneration Qwen4ExpForCausalLM));
sub arch { 'qwen4exp' }

sub layer_table {
    my ($self) = @_;
    return [@{$self->SUPER::layer_table}, [qr/(?:attn|mlp)_hyper_connection\.\w+\.weight/, undef],
            [qr/ple\..+/, undef], [qr/self_attn\.indexer\.\w+\.weight/, undef]];
}

# 1-based in the config
sub _ple_layers { map { $_ - 1 } @{$_[0]->hp->{ple_layer_ids} // []} }

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->SUPER::params;
    $self->u32('hyper_connection.count', $hp->{hc_count});
    $self->u32('hyper_connection.low_rank', $hp->{hc_lowrank});
    $self->u32('attention.indexer.head_count', $hp->{indexer_n_heads});
    $self->u32('attention.indexer.key_length', $hp->{indexer_head_dim});
    $self->u32('attention.indexer.top_k', $hp->{indexer_budget});
    # The trunk's entries test layer_types for "full_attention", which transformers' Qwen4Exp config renames
    # "indexed_attention" on load: convert_hf_to_gguf.py (and the published GGUFs) write 0 for every trunk layer and
    # the ratio for the MTP block only. The engine reads them so; kept for the same file.
    my $ratio = $hp->{indexer_compress_ratio};
    $self->arr('attention.compress_ratios', [(0) x $self->{n_main}, ($ratio) x $self->{n_mtp}], ROCm::Inference::Convert::I32);
    my @ple = $self->_ple_layers;
    return unless @ple;
    $self->arr('ple.layers', \@ple, ROCm::Inference::Convert::I32);
    $self->u32('ple.ngram_size', $hp->{ngram_size});
    $self->u32('ple.heads_per_ngram', $hp->{heads_per_ngram});
    $self->u32('ple.conv_kernel', $hp->{ple_conv_kernel_size});
    my $eos = $hp->{eos_token_id} // croak 'eos_token_id is needed: the PLE hash resets its n-grams on it';
    $self->u32('ple.eos_token_id', ref $eos ? $eos->[-1] : $eos);
    my $img = $self->{config}{image_token_id} // $hp->{image_token_id};
    $self->u32('ple.image_token_id', $img) if defined $img;
    my ($sh) = $self->_ple_shards;
    $self->u32('embedding_length_per_layer_input', $sh->{shape}[-1]) if $sh;
    for my $c (['layer_multipliers', 'ple.layer_multipliers'], ['ngram_heads_offsets', 'ple.head_offsets'],
               ['ngram_heads_vocab_sizes', 'ple.head_vocab_sizes']) {
        my ($t) = grep { $_->{name} =~ /ple_embedding\.\Q$c->[0]\E$/ } map { $self->st->{tensors}{$_} } @{$self->st->{order}};
        croak "PLE constant $c->[0] missing from the checkpoint" unless $t;
        $self->kv_arch($c->[1], ROCm::Inference::Convert::ARR, [ROCm::Inference::Convert::U64, [$self->ints($t)]]);
        push @{$self->{const_used}}, $t;
    }
}

# Integer values of a small I64 / I32 tensor
sub ints {
    my ($self, $t) = @_;
    my %fh;
    my $b = ROCm::Inference::SafeTensors::st_read_rows(\%fh, $t, 0, ROCm::Inference::SafeTensors::st_rows($t));
    return unpack $t->{dtype} eq 'I64' ? 'q<*' : $t->{dtype} eq 'I32' ? 'l<*' : croak("$t->{name}: $t->{dtype} is not an integer"), $b;
}

# The n-gram table's shards, by index
sub _ple_shards {
    my ($self) = @_;
    my $T = $self->st->{tensors};
    my %sh = map { /\.ngram_embedding\.shard_(\d+)\.weight$/ ? ($1 => $T->{$_}) : () } keys %$T;
    my $n = $self->hp->{split_ngram_parts} // scalar keys %sh;
    croak scalar(keys %sh) . " PLE embedding shards, $n expected" if %sh && keys %sh != $n;
    return map { $sh{$_} } sort { $a <=> $b } keys %sh;
}

sub _hc {  # hyper-connection mixers: [down | inject] as one, up, the norm as 1 + w
    my ($self, $p, $b, $to) = @_;
    my $T = $self->st->{tensors};
    my $d = $T->{"$p.input_mix_weight_down.weight"} or return;
    my @s = ($self->_norm1("$b.${to}_norm.weight", $self->t("$p.hc_norm.weight")),
             $self->same("$b.${to}_up.weight", $self->t("$p.input_mix_weight_up.weight")));
    if (my $inj = $T->{"$p.block_inject_weight.weight"}) {
        push @s, {name => "$b.${to}_down_inject.weight", ne => [$d->{shape}[1], $d->{shape}[0] + $inj->{shape}[0]],
                  parts => [[$d, 0, $d->{shape}[0]], [$inj, 0, $inj->{shape}[0]]]};
    } else {
        push @s, $self->same("$b.${to}_down.weight", $d);
    }
    return @s;
}

sub layer_extra {
    my ($self, $l, $p) = @_;
    my $hp = $self->hp;
    my $T = $self->st->{tensors};
    my $b = "blk.$l";
    my %s = map { $_->{name} => $_ } $self->SUPER::layer_extra($l, $p);
    for (['attn_qkv', 'attn_gate', 'attn_qkv_gate'], ['ssm_beta', 'ssm_alpha', 'ssm_beta_alpha']) {
        my ($x, $y) = map { delete $s{"$b.$_.weight"} } @$_[0, 1];
        next unless $x;
        $s{"$b.$_->[2].weight"} = {name => "$b.$_->[2].weight", ne => [$x->{ne}[0], $x->{ne}[1] + $y->{ne}[1]],
                                    parts => [@{$x->{parts}}, @{$y->{parts}}]};
    }
    my @s = map { $s{$_} } sort keys %s;
    push @s, $self->_hc("$p.attn_hyper_connection", $b, 'hc_attn'), $self->_hc("$p.mlp_hyper_connection", $b, 'hc_ffn');
    if (my $qk = $T->{"$p.self_attn.indexer.index_qk_proj.weight"}) {
        my $nq = $hp->{indexer_n_heads} * $hp->{indexer_head_dim};
        push @s, {name => "$b.indexer.q_proj.weight", ne => [$qk->{shape}[1], $nq], parts => [[$qk, 0, $nq]]},
                 {name => "$b.indexer.k_proj.weight", ne => [$qk->{shape}[1], $qk->{shape}[0] - $nq],
                  parts => [[$qk, $nq, $qk->{shape}[0] - $nq]]};
        push @s, $self->_norm1("$b.indexer.q_norm.weight", $self->t("$p.self_attn.indexer.q_layernorm.weight")),
                 $self->_norm1("$b.indexer.k_norm.weight", $self->t("$p.self_attn.indexer.k_layernorm.weight"));
    }
    if (my $c = $T->{"$p.ple.conv1d.weight"}) {
        push @s, {name => "$b.ple_conv1d.weight", ne => [$c->{shape}[-1], $c->{shape}[0]], parts => [[$c, 0, $c->{shape}[0]]]};
        push @s, $self->same("$b.ple_key.weight", $self->t("$p.ple.key_proj.weight")),
                 $self->same("$b.ple_value.weight", $self->t("$p.ple.value_proj.weight"));
        push @s, map { $self->_norm1("$b.ple_$_.weight", $self->t("$p.ple.$_.weight")) } qw(norm_conv norm_key norm_query);
        my @sh = $self->_ple_shards;
        my $rows = 0;
        $rows += $_->{shape}[0] for @sh;
        push @s, {name => 'per_layer_token_embd.weight', ne => [$sh[0]{shape}[-1], $rows],
                  parts => [map { [$_, 0, $_->{shape}[0]] } @sh], uses => $self->{const_used}};
    }
    return @s;
}

sub tensors {
    my ($self) = @_;
    my @s = $self->SUPER::tensors;
    my @hc = $self->_hc('model.language_model.hyper_connection_mixer', 'x', 'hc');
    $_->{name} =~ s/^x\.hc_/output_hc_/ for @hc;
    return (@s, @hc);
}

sub mtp_extra {
    my ($self, $b) = @_;
    my $T = $self->st->{tensors};
    my @s;
    if (my $e = $T->{'mtp.fc_embedding.weight'}) {  # eh_proj([e ; h]) = fc_embedding(e) + fc_hidden(h)
        my $h = $self->t('mtp.fc_hidden.weight');
        push @s, {name => "$b.nextn.eh_proj.weight", ne => [$e->{shape}[1] + $h->{shape}[1], $e->{shape}[0]],
                  hcat => [$e, $h]};
    }
    my @hc = $self->_hc('mtp.hyper_connection_mixer', 'x', 'hc');
    $_->{name} =~ s/^x\.hc_/$b.nextn.hc_head_/ for @hc;
    return (@s, @hc);
}

package ROCm::Inference::Convert::Qwen35Moe;
our @ISA = ('ROCm::Inference::Convert::Qwen35');
__PACKAGE__->register(qw(Qwen3_5MoeForConditionalGeneration Qwen3_5MoeForCausalLM));
sub arch { 'qwen35moe' }

package ROCm::Inference::Convert::Glm5Next;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert');
__PACKAGE__->register(qw(Glm5NextForConditionalGeneration));
sub arch { 'glm5-next' }
sub skipped { qr/^model\.visual\./ }  # the vision encoder: a separate (mmproj) file
sub tokenizer_pre { 'glm4' }

# convert_hf_to_gguf.py's Glm5NextModel: the MTP layer after the decoder layers counts as a block; head_dim is not
# the attention's (MLA's widths are)
sub init {
    my ($self) = @_;
    my $hp = $self->hp;
    delete $hp->{head_dim};
    $self->{n_main} = $hp->{num_hidden_layers};
    $self->{n_nextn} = $hp->{num_nextn_predict_layers} // 0;
    $self->{block_count} += $self->{n_nextn};
}

sub vocab {
    my ($self) = @_;
    $self->vocab_gpt2(special => [[bos => '[gMASK]'], [eot => '<|user|>'], [unk => '<|endoftext|>'],
                                  [eom => '<|observation|>']]);
}

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    my @kinds = @{$hp->{layer_types}};
    croak "layer_types has " . @kinds . " layers, num_hidden_layers $self->{n_main}" unless @kinds == $self->{n_main};
    my @kv = map { $_ eq 'linear_attention' ? 0 : 1 } @kinds;
    push @kv, (1) x ($self->{block_count} - @kv);
    $self->arr('attention.head_count_kv', \@kv, ROCm::Inference::Convert::I32);
    $self->u32('vocab_size', $hp->{vocab_size});
    $self->f32('attention.layer_norm_epsilon', 1e-6);
    $self->u32('nextn_predict_layers', $self->{n_nextn}) if $self->{n_nextn};
    my $lin = $hp->{linear_attn_config};
    croak "linear_attn_config.num_heads $lin->{num_heads} != num_attention_heads $hp->{num_attention_heads}"
        unless $lin->{num_heads} == $hp->{num_attention_heads};
    $self->u32('ssm.conv_kernel', $lin->{short_conv_kernel_size});
    $self->u32('kda.head_dim', $lin->{head_dim});
    $self->f32('kda.gate_lower_bound', $lin->{gate_lower_bound}) if defined $lin->{gate_lower_bound};
    croak 'expected NoPE-only MLA' unless $hp->{mla_use_nope} && $hp->{qk_rope_head_dim} == 0;
    my ($kvl, $rope) = ($hp->{kv_lora_rank}, $hp->{qk_rope_head_dim});
    $self->u32('attention.q_lora_rank', $hp->{q_lora_rank});
    $self->u32('attention.kv_lora_rank', $kvl);
    $self->u32('rope.dimension_count', $rope);
    $self->u32('attention.key_length', $kvl + $rope);
    $self->u32('attention.value_length', $kvl);
    $self->u32('attention.key_length_mla', $hp->{qk_nope_head_dim} + $rope);
    $self->u32('attention.value_length_mla', $hp->{v_head_dim});
    $self->u32('attention.indexer.head_count', $hp->{index_n_heads});
    $self->u32('attention.indexer.key_length', $hp->{index_head_dim});
    $self->u32('attention.indexer.top_k', $hp->{index_topk});
    $self->u32('attention.indexer.kpool', $hp->{index_kpool});
    $self->bool('attention.indexer.kpool_select_tail', $hp->{index_kpool_always_select_tail} // 1);
    $self->arr('attention.indexer.types', [map { $_ eq 'full' ? 1 : 0 } @{$hp->{indexer_types}}],
               ROCm::Inference::Convert::BOOL) if $hp->{indexer_types};
    croak 'expected mHC' unless $hp->{mhc} // 1;
    $self->u32('hyper_connection.count', $hp->{hc_mult});
    $self->u32('hyper_connection.sinkhorn_iterations', $hp->{hc_sinkhorn_iters});
    $self->f32('hyper_connection.epsilon', $hp->{hc_eps});
    $self->u32('leading_dense_block_count', $hp->{first_k_dense_replace});
    $self->u32('expert_feed_forward_length', $hp->{moe_intermediate_size});
    $self->u32('expert_shared_count', $hp->{n_shared_experts});
    $self->f32('expert_weights_scale', $hp->{routed_scaling_factor});
    $self->bool('expert_weights_norm', $hp->{norm_topk_prob});
    if (defined(my $lim = $hp->{swiglu_limit})) {
        $self->arr('swiglu_clamp_exp', [($lim) x $self->{block_count}], ROCm::Inference::Convert::F32);
        $self->arr('swiglu_clamp_shexp', [($lim) x $self->{block_count}], ROCm::Inference::Convert::F32);
    }
}

sub tensors {
    my ($self) = @_;
    my $hp = $self->hp;
    my $T = $self->st->{tensors};
    my $P = 'model.language_model';
    my $t = sub { $self->t("$P.$_[0]") };
    my $has = sub { exists $T->{"$P.$_[0]"} };
    my @s = ($self->same('token_embd.weight', $t->('embed_tokens.weight')));
    push @s, $self->same('output.weight', $self->t('lm_head.weight')) unless $hp->{tie_word_embeddings};
    for my $l (0 .. $self->{block_count} - 1) {
        my ($p, $b) = ("layers.$l", "blk.$l");
        my $mtp = $l >= $self->{n_main};
        push @s, $self->same("$b.attn_norm.weight", $t->("$p.input_layernorm.weight"));
        push @s, $self->same("$b.ffn_norm.weight", $t->("$p.post_attention_layernorm.weight"));
        for my $k (qw(attn ffn)) {
            next unless $has->("$p.hc_${k}_fn");
            push @s, map { $self->same("$b.hc_${k}_$_.weight", $t->("$p.hc_${k}_$_"), f32 => 1) } qw(fn base scale);
        }
        my $a = "$p.self_attn";
        if ($has->("$a.q_a_proj.weight")) {  # sparse MLA with the DSA indexer
            push @s, map { $self->same("$b.$_->[0]", $t->("$a.$_->[1]")) } (
                ['attn_q_a.weight', 'q_a_proj.weight'], ['attn_q_a_norm.weight', 'q_a_layernorm.weight'],
                ['attn_q_b.weight', 'q_b_proj.weight'], ['attn_kv_a_mqa.weight', 'kv_a_proj_with_mqa.weight'],
                ['attn_kv_a_norm.weight', 'kv_a_layernorm.weight'], ['attn_output.weight', 'o_proj.weight'],
                ['indexer.attn_q_b.weight', 'indexer.wq_b.weight'], ['indexer.attn_k.weight', 'indexer.wk.weight'],
                ['indexer.k_norm.weight', 'indexer.k_norm.weight'], ['indexer.k_norm.bias', 'indexer.k_norm.bias']);
            push @s, $self->same("$b.indexer.proj.weight", $t->("$a.indexer.weights_proj.weight"), f32 => 1);
            push @s, $self->same("$b.indexer_compressor_ape.weight", $t->("$a.indexer.index_kpool_compress_ape"), f32 => 1);
            push @s, $self->same("$b.indexer_compressor_gate.weight", $t->("$a.indexer.index_kpool_compress_gate"), f32 => 1);
            push @s, $self->_kv_b($b, $t->("$a.kv_b_proj.weight"));
        } else {  # KDA
            push @s, map { $self->same("$b.$_->[0]", $t->("$a.$_->[1]")) } (
                ['attn_q.weight', 'q_proj.weight'], ['attn_k.weight', 'k_proj.weight'], ['attn_v.weight', 'v_proj.weight'],
                ['attn_output.weight', 'o_proj.weight'], ['ssm_f_a.weight', 'f_a_proj.weight'],
                ['ssm_f_b.weight', 'f_b_proj.weight'], ['ssm_g_a.weight', 'g_a_proj.weight'],
                ['ssm_g_b.weight', 'g_b_proj.weight'], ['ssm_beta.weight', 'b_proj.weight'],
                ['ssm_norm.weight', 'o_norm.weight']);
            push @s, $self->same("$b.ssm_dt.bias", $t->("$a.dt_bias"), f32 => 1);
            for my $k (qw(q k v)) {  # [d_inner, 1, d_conv] as [1, d_inner, 1, d_conv]
                my $c = $t->("$a.${k}_conv1d.weight");
                my ($d_inner, $d_conv) = ($c->{shape}[0], $c->{shape}[-1]);
                push @s, {%{$self->same("$b.ssm_conv1d_$k.weight", $c)}, ne => [$d_conv, 1, $d_inner, 1]};
            }
            my $al = $t->("$a.A_log");
            my $nh = $hp->{num_attention_heads};
            push @s, {name => "$b.ssm_a", ne => [$nh], f32 => 1, uses => [$al], data => sub {
                my @v = ($self->floats($al))[0 .. $nh - 1];
                return ROCm::Inference::exp_neg_f32(pack 'f<*', @v);
            }};
        }
        my $m = "$p.mlp";
        if ($has->("$m.gate_proj.weight")) {  # dense
            push @s, map { $self->same("$b.ffn_$_.weight", $t->("$m.${_}_proj.weight")) } qw(gate up down);
        } else {
            my $ne = $hp->{n_routed_experts};
            for my $w (qw(down gate up)) {
                push @s, $self->stack("$b.ffn_${w}_exps.weight", [map { $t->("$m.experts.$_.${w}_proj.weight") } 0 .. $ne - 1]);
            }
            push @s, $self->same("$b.exp_probs_b.bias", $t->("$m.gate.e_score_correction_bias"), f32 => 1);
            push @s, $self->same("$b.ffn_gate_inp.weight", $t->("$m.gate.weight"));
            push @s, map { $self->same("$b.ffn_${_}_shexp.weight", $t->("$m.shared_experts.${_}_proj.weight")) } qw(down gate up);
        }
        if ($mtp) {
            push @s, $self->same("$b.nextn.eh_proj.weight", $t->("$p.eh_proj.weight"));
            push @s, $self->same("$b.nextn.enorm.weight", $t->("$p.enorm.weight"));
            push @s, $self->same("$b.nextn.hnorm.weight", $t->("$p.hnorm.weight"));
            push @s, $self->same("$b.nextn.shared_head_norm.weight", $t->("$p.shared_head.norm.weight"));
        }
    }
    push @s, $self->same('output_norm.weight', $t->('norm.weight'));
    return @s;
}

# MLA absorption: kv_b_proj [n_head * (nope + v), kv_lora] as attn_k_b (each head's nope rows transposed: [n_head,
# kv_lora, nope]) and attn_v_b (each head's v rows: [n_head, v, kv_lora])
sub _kv_b {
    my ($self, $b, $kv) = @_;
    my $hp = $self->hp;
    my ($nh, $nope, $vd) = ($hp->{num_attention_heads}, $hp->{qk_nope_head_dim}, $hp->{v_head_dim});
    my $kvl = $kv->{shape}[1];
    croak "$kv->{name}: [@{$kv->{shape}}], $nh heads of $nope + $vd rows wanted" unless $kv->{shape}[0] == $nh * ($nope + $vd);
    my $k = {name => "$b.attn_k_b.weight", ne => [$nope, $kvl, $nh], uses => [$kv], data => sub {
        my @f = $self->floats($kv);
        my @o;
        for my $h (0 .. $nh - 1) {
            my $base = $h * ($nope + $vd) * $kvl;
            for my $c (0 .. $kvl - 1) { push @o, map { $f[$base + $_ * $kvl + $c] } 0 .. $nope - 1 }
        }
        return pack 'f<*', @o;
    }};
    my $v = {name => "$b.attn_v_b.weight", ne => [$kvl, $vd, $nh],
             parts => [map { [$kv, $_ * ($nope + $vd) + $nope, $vd] } 0 .. $nh - 1]};
    return ($k, $v);
}

package ROCm::Inference::Convert::DeepSeek4;
use Carp qw(croak);
use File::Basename qw(dirname);
use File::Spec;
our @ISA = ('ROCm::Inference::Convert');
__PACKAGE__->register(qw(DeepseekV4ForCausalLM));
sub arch { 'deepseek4' }
sub skipped { qr/^(?:mtp\.|aligner\.|image_)/ }  # the MTP layer (a separate draft file upstream) and vision
sub tokenizer_pre { 'joyai-llm' }
sub ref_ftype { 38 }  # MOSTLY_MXFP4_MOE

# DeepSeek's own checkpoint layout (layers.N.attn.wq_a, ffn.experts.E.w1, mtp.0.*), as convert_hf_to_gguf.py's
# DeepseekV4Model converts it: FP8 weights with E8M0 block scales, routed experts packed FP4 with E8M0 scales
# (MXFP4, moved bit for bit), hash-routing tables as I32. The MTP layer is not converted (upstream's "conversion
# v0" leaves it to a separate draft file).
sub init {
    my ($self) = @_;
    my $rs = $self->{config}{rope_scaling};
    if (ref $rs eq 'HASH' && !defined(($self->{rope}{full_attention} // $self->{rope})->{rope_type})) {
        my %r = %$rs;
        $r{rope_type} //= $r{type} if defined $r{type};
        $self->{rope} = {%{$self->{rope}}, %r};
    }
}

sub vocab {
    my ($self) = @_;
    $self->vocab_gpt2;
    return if defined $self->{kv_at}{'tokenizer.chat_template'};
    # the release has no template: convert_hf_to_gguf.py adds its own (models/templates), as we do ours
    my $name = basename_($self->{dir}) =~ /0731/ ? 'deepseek-v4-flash-0731.jinja' : 'deepseek-v4.jinja';
    my $path = File::Spec->catfile(dirname(__FILE__), 'Convert', 'templates', $name);
    if (-f $path) {
        my $tpl = ROCm::Inference::Convert::_slurp($path);
        utf8::decode($tpl);
        $self->kv('tokenizer.chat_template', ROCm::Inference::Convert::STR, $tpl);
    }
}
sub basename_ { my ($d) = @_; $d =~ s{/+$}{}; return (File::Spec->splitdir($d))[-1] }

sub params {
    my ($self) = @_;
    my $hp = $self->hp;
    $self->u32('rope.dimension_count', $hp->{qk_rope_head_dim});
    $self->u32('attention.q_lora_rank', $hp->{q_lora_rank});
    $self->u32('attention.sliding_window', $hp->{sliding_window});
    $self->u32('expert_feed_forward_length', $hp->{moe_intermediate_size});
    $self->u32('expert_shared_count', $hp->{n_shared_experts});
    $self->f32('expert_weights_scale', $hp->{routed_scaling_factor});
    $self->bool('expert_weights_norm', $hp->{norm_topk_prob});
    $self->arr('swiglu_clamp_exp', [($hp->{swiglu_limit}) x $self->{block_count}], ROCm::Inference::Convert::F32);
    $self->arr('swiglu_clamp_shexp', [($hp->{swiglu_limit}) x $self->{block_count}], ROCm::Inference::Convert::F32);
    $self->u32('attention.indexer.head_count', $hp->{index_n_heads});
    $self->u32('attention.indexer.key_length', $hp->{index_head_dim});
    $self->u32('attention.indexer.top_k', $hp->{index_topk});
    $self->u32('attention.output_group_count', $hp->{o_groups});
    $self->u32('attention.output_lora_rank', $hp->{o_lora_rank});
    $self->arr('attention.compress_ratios', $hp->{compress_ratios}, ROCm::Inference::Convert::I32);
    $self->f32('attention.compress_rope_freq_base', $hp->{compress_rope_theta});
    $self->u32('hyper_connection.count', $hp->{hc_mult});
    $self->u32('hyper_connection.sinkhorn_iterations', $hp->{hc_sinkhorn_iters});
    $self->f32('hyper_connection.epsilon', $hp->{hc_eps});
    $self->u32('hash_layer_count', $hp->{num_hash_layers});
    $self->u32('embedding_length_out', $hp->{hidden_size} * $hp->{hc_mult});
}

# convert_hf_to_gguf.py's types for this architecture: FP8 matrices Q8_0, F32 sources F32, BF16 matrices BF16;
# 1-D tensors and norms F32
sub base_type {
    my ($self, $s) = @_;
    return 'f32' if @{$s->{ne}} <= 1 || $s->{name} =~ /_norm\.weight$/;
    my $t = $s->{parts} && $s->{parts}[0][0];
    return 'q8_0' if $t && $t->{dtype} eq 'F8_E4M3';
    return 'f32' if $t && $t->{dtype} eq 'F32';
    return 'bf16';
}

sub tensors {
    my ($self) = @_;
    my $hp = $self->hp;
    my $T = $self->st->{tensors};
    my @s = ($self->same('token_embd.weight', $self->t('embed.weight')),
             $self->same('output.weight', $self->t('head.weight')),
             $self->same('output_norm.weight', $self->t('norm.weight')),
             map { $self->same("output_hc_$_.weight", $self->t("hc_head_$_")) } qw(base fn scale));
    my %map = ('attn.attn_sink' => 'attn_sinks.weight', 'attn.wq_a.weight' => 'attn_q_a.weight',
               'attn.wq_b.weight' => 'attn_q_b.weight', 'attn.q_norm.weight' => 'attn_q_a_norm.weight',
               'attn.wkv.weight' => 'attn_kv.weight', 'attn.kv_norm.weight' => 'attn_kv_a_norm.weight',
               'attn.wo_a.weight' => 'attn_output_a.weight', 'attn.wo_b.weight' => 'attn_output_b.weight',
               'attn.compressor.ape' => 'attn_compressor_ape.weight',
               'attn.compressor.wkv.weight' => 'attn_compressor_kv.weight',
               'attn.compressor.wgate.weight' => 'attn_compressor_gate.weight',
               'attn.compressor.norm.weight' => 'attn_compressor_norm.weight',
               'attn.indexer.wq_b.weight' => 'indexer.attn_q_b.weight',
               'attn.indexer.weights_proj.weight' => 'indexer.proj.weight',
               'attn.indexer.compressor.ape' => 'indexer_compressor_ape.weight',
               'attn.indexer.compressor.wkv.weight' => 'indexer_compressor_kv.weight',
               'attn.indexer.compressor.wgate.weight' => 'indexer_compressor_gate.weight',
               'attn.indexer.compressor.norm.weight' => 'indexer_compressor_norm.weight',
               'attn_norm.weight' => 'attn_norm.weight', 'ffn_norm.weight' => 'ffn_norm.weight',
               'ffn.gate.weight' => 'ffn_gate_inp.weight', 'ffn.gate.bias' => 'exp_probs_b.bias',
               'ffn.gate.bias_vl' => 'exp_probs_b_vl.bias',
               'ffn.shared_experts.w1.weight' => 'ffn_gate_shexp.weight',
               'ffn.shared_experts.w2.weight' => 'ffn_down_shexp.weight',
               'ffn.shared_experts.w3.weight' => 'ffn_up_shexp.weight',
               map { ("hc_${_}", "hc_$_.weight") } map { ("attn_$_", "ffn_$_") } qw(fn base scale));
    my $ne = $hp->{n_routed_experts};
    for my $l (0 .. $self->{block_count} - 1) {
        my $p = "layers.$l";
        my $hash = $l < $hp->{num_hash_layers};
        for my $k (sort grep { !/\.(?:scale)$/ || /hc_/ } map { s/^\Q$p.\E//r } grep { /^\Q$p.\E/ } keys %$T) {
            next if $k =~ /^ffn\.experts\./ || $k eq 'ffn.gate.tid2eid';
            next if $k =~ /\.scale$/ && $k !~ /^hc_/;
            next if $hash && $k eq 'ffn.gate.bias';  # hash layers route by tid2eid; the bias is unused
            my $g = $map{$k} // croak "$p.$k: no GGUF name";
            push @s, $self->same("blk.$l.$g", $T->{"$p.$k"});
        }
        if ($hash) {
            my $tt = $self->t("$p.ffn.gate.tid2eid");
            croak "$tt->{name}: I64 wanted" unless $tt->{dtype} eq 'I64';
            push @s, {name => "blk.$l.ffn_gate_tid2eid.weight", ne => [reverse @{$tt->{shape}}], type_fixed => 'i32',
                      uses => [$tt],
                      from => 'i64', raw => sub {
                my %fh;
                my $b = ROCm::Inference::SafeTensors::st_read_rows(\%fh, $tt, 0, ROCm::Inference::SafeTensors::st_rows($tt));
                return pack 'l<*', unpack 'q<*', $b;
            }};
        }
        for my $w ([w1 => 'gate'], [w2 => 'down'], [w3 => 'up']) {
            my @pairs = map { [$self->t("$p.ffn.experts.$_.$w->[0].weight"), $self->t("$p.ffn.experts.$_.$w->[0].scale")] }
                0 .. $ne - 1;
            my @sh = @{$pairs[0][0]{shape}};
            push @s, {name => "blk.$l.ffn_$w->[1]_exps.weight", ne => [2 * $sh[1], $sh[0], $ne], base => 'mxfp4',
                      from => 'fp4', mxfp4 => \@pairs};
        }
    }
    return @s;
}

1;

__END__

=head1 NAME

ROCm::Inference::Convert - a Hugging Face checkpoint to a GGUF, in the engine's types directly

=head1 SYNOPSIS

  use ROCm::Inference::Convert;
  use ROCm::Inference::Quant;
  my $m = ROCm::Inference::Convert->open_model('/data/hf/Qwen3-0.6B');
  $m->convert('Qwen3-0.6B-FAST.gguf', quant => ROCm::Inference::Quant->new(target => 'fast'));

=head1 DESCRIPTION

The metadata, vocabulary and tensor names are those llama.cpp's F<convert_hf_to_gguf.py> writes, so llama.cpp and
the engine read the result alike; the tensor data is streamed from the safetensors shards and encoded by the
engine's encoders (C<ROCm::Inference::quantize>, C<quantize_fp8_blocks> for FP8 weights with block scales) as the
policy of L<ROCm::Inference::Quant> says, or as BF16 the way C<convert_hf_to_gguf.py --outtype bf16> writes it
when no policy is given. The architectures are packages under C<ROCm::Inference::Convert::>.

=over

=item open_model($dir, arch => NAME, pre => NAME, name => NAME)

The checkpoint in C<$dir>; C<arch> overrides F<config.json>'s architecture, C<pre> the tokenizer's
pre-tokenizer name, C<name> the model's name.

=item convert($out, quant => $policy, threads => N, progress => sub { my ($spec, $i, $n) = @_ }, dry_run => 1)

Writes the GGUF (to F<$out.part>, renamed when complete); returns its metadata, tensor plans and size.

=item architectures

The Hugging Face architecture names supported.

=back

=cut
