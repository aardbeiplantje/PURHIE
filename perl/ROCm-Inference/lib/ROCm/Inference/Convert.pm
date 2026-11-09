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
        $self->kv('general.sampling.top_k', I32, int $gc->{top_k}) if defined $gc->{top_k} && !ref $gc->{top_k};
        $self->kv('general.sampling.top_p', F32, 0 + $gc->{top_p}) if defined $gc->{top_p} && !ref $gc->{top_p};
        $self->kv('general.sampling.temp', F32, 0 + $gc->{temperature})
            if defined $gc->{temperature} && !ref $gc->{temperature};
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
        if (exists $hp->{is_causal} && is_bool($hp->{is_causal}) && !$hp->{is_causal}) || $hp->{use_bidirectional_attention};
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
        my %g = (softmax => 1, sigmoid => 2, sqrtsoftplus => 3);
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
    for my $a (@{$tj->{added_tokens} // []}) {
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
        my $added = $tj->{added_tokens} // [];
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
        my ($id) = map { $_->{id} } grep { $_->{content} eq $text } @{$tj->{added_tokens} // []};
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

# The weight_scale_inv of an FP8 tensor, if it has one
sub fp8_scale {
    my ($self, $t) = @_;
    return undef unless $t->{dtype} eq 'F8_E4M3';
    (my $sn = $t->{name}) =~ s/\.weight$/.weight_scale_inv/;
    return $self->{st}{tensors}{$sn} // croak "$t->{name}: FP8 without $sn";
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
    $self->kv('general.file_type', U32, $q ? $q->ftype : 32);
    $self->kv('general.quantization_version', U32, 2);
    $self->vocab;
    my @specs = $self->tensors;
    for my $s (@specs) {
        my $base = $self->base_type($s);
        my ($want, $why) = $q ? $q->plan($s->{name}, $s->{ne}, $base) : ($base);
        $s->{why} = $why;
        $s->{type} = gguf_type_id($want);
        my $t0 = $s->{parts} && $s->{parts}[0][0];
        $s->{from} = $t0 ? lc $t0->{dtype} : 'f32';
        $s->{in_bytes} = 0;
        $s->{in_bytes} += $_->[2] * ROCm::Inference::SafeTensors::st_row_bytes($_->[0]) for @{$s->{parts} // []};
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
        $s->{bytes} = ROCm::Inference::quant_row_size($s->{type}, $s->{ne}[0]) * $rows
            or croak "$s->{name}: " . (gguf_type_info($s->{type}))[0] . " cannot hold rows of $s->{ne}[0]";
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
        my ($want, $why) = $q->plan($t->{name}, $t->{ne}, $src);
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
        croak "$sc->{name}: F32 scales wanted, $sc->{dtype}" unless $sc->{dtype} eq 'F32';
        for (my $r = $r0; $r < $r0 + $n; $r += $chunk) {
            my $m = $r + $chunk > $r0 + $n ? $r0 + $n - $r : $chunk;
            my $w = st_read_rows($fh, $t, $r, $m);
            my $sr = st_read_rows($fh, $sc, $r / $br, int(($m + $br - 1) / $br));
            my $b = ROCm::Inference::quantize_fp8_blocks($s->{type}, $w, $sr, $m, $cols, $br, $bc, $ss[1], $threads);
            $wr->($b);
            $written += length $b;
        }
        return $written;
    }
    my $src = $DT{$t->{dtype}} // croak "$t->{name}: dtype $t->{dtype} is not read";
    my $sid = gguf_type_id($src);
    my $chunk = int((64 << 20) / $cols) || 1;
    for (my $r = $r0; $r < $r0 + $n; $r += $chunk) {
        my $m = $r + $chunk > $r0 + $n ? $r0 + $n - $r : $chunk;
        my $b = ROCm::Inference::quantize($s->{type}, $sid, st_read_rows($fh, $t, $r, $m), $m, $cols, $threads);
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

sub vocab { $_[0]->vocab_gpt2 }
sub params { }

# ---------------------------------------------------------------- architectures

package ROCm::Inference::Convert::Qwen3;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert');
__PACKAGE__->register(qw(Qwen3ForCausalLM));
sub arch { 'qwen3' }
sub tokenizer_pre { 'qwen2' }

sub tensors {
    my ($self) = @_;
    my $st = $self->st;
    my @s = ($self->same('token_embd.weight', $self->t('model.embed_tokens.weight')));
    for my $l (0 .. $self->{block_count} - 1) {
        my $p = "model.layers.$l";
        push @s, map { $self->same("blk.$l.$_->[0]", $self->t("$p.$_->[1]")) } (
            ['attn_norm.weight', 'input_layernorm.weight'], ['attn_q.weight', 'self_attn.q_proj.weight'],
            ['attn_k.weight', 'self_attn.k_proj.weight'], ['attn_v.weight', 'self_attn.v_proj.weight'],
            ['attn_output.weight', 'self_attn.o_proj.weight'], ['attn_q_norm.weight', 'self_attn.q_norm.weight'],
            ['attn_k_norm.weight', 'self_attn.k_norm.weight'], ['ffn_norm.weight', 'post_attention_layernorm.weight'],
            ['ffn_gate.weight', 'mlp.gate_proj.weight'], ['ffn_up.weight', 'mlp.up_proj.weight'],
            ['ffn_down.weight', 'mlp.down_proj.weight']);
    }
    push @s, $self->same('output_norm.weight', $self->t('model.norm.weight'));
    push @s, $self->same('output.weight', $st->{tensors}{'lm_head.weight'} // $self->t('model.embed_tokens.weight'))
        if $st->{tensors}{'lm_head.weight'} || !$self->hp->{tie_word_embeddings};
    return @s;
}

package ROCm::Inference::Convert::Glm5Next;
use Carp qw(croak);
our @ISA = ('ROCm::Inference::Convert');
__PACKAGE__->register(qw(Glm5NextForConditionalGeneration));
sub arch { 'glm5-next' }
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
            push @s, {name => "$b.ssm_a", ne => [$nh], f32 => 1, data => sub {
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
    my $k = {name => "$b.attn_k_b.weight", ne => [$nope, $kvl, $nh], data => sub {
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
