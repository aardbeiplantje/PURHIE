package llama;

use strict; use warnings;

use nginx;
use JSON::XS;
use POSIX qw(strftime);
use File::Path qw(make_path);

our $_json;
our $CACHE;

sub fixup {
    my ($r) = @_;
    my $r_ok = eval {
        $r->has_request_body(\&handle_req);
        return OK;
    };
    if($@){
        chomp(my $err = $@);
        print_error("[ERROR] request: $err");
        # the status first: send_http_header sends it (the Lua side then keeps the original body)
        $r->status(500);
        $r->send_http_header("text/plain");
        return OK;
    }
    return $r_ok;
}

sub handle_req {
    my ($r) = @_;
    my $method = $r->request_method;
    my $slot_id = $r->header_in("X-LLamaCPP-Id-slot");
    my $rb = $r->request_body();
    my $rf = $r->request_body_file();
    if(!defined $rb and defined $rf){
        open(my $b_fh, "<", $rf) or do {
            print_error("[ERROR] cant open tmp body file $rf: $!");
            return HTTP_INTERNAL_SERVER_ERROR;
        };
        local $/;
        $rb = <$b_fh>;
        close $b_fh;
    }
    $rb //= "{}";
    print_error("[INFO] REQUEST $method, header X-LLamaCPP-Id-slot: ".( $slot_id//"<no id_slot>"));
    if(defined $slot_id and $slot_id =~ m/^\d+$/){
        eval {
            my $req = JSON::XS::decode_json($rb);
            $_json //= JSON::XS->new->utf8->allow_blessed->allow_unknown->allow_nonref->convert_blessed->canonical;
            print_error("[INFO] REQUEST $method, header X-LLamaCPP-Id-slot: ".( $slot_id//"<no id_slot>").", model:".($req->{model}//"<no model>"));

            do_magic_fixes($r, $req, $slot_id);

            $rb = $_json->encode($req);

            eval {
                my $log_dir = "/tmp/request-logs";
                make_path($log_dir) unless -d $log_dir;
                my $log_file = strftime("$log_dir/requests_%Y%m%d_%H%M00.log", localtime());
                open(my $fh, ">>", $log_file) or do {
                    print_error("[WARN] cannot open log file $log_file: $!");
                    return;
                };
                my $timestamp = localtime();
                my $model = $req->{model} // "unknown";
                print_error("[INFO] Logging request body for model=$model to $log_file");
                (my $safe_rb = $rb) =~ s/[\r\n]+//g;
                print $fh "[$timestamp] method=$method slot_id=" . ($slot_id // "none") . " model=$model $safe_rb\n";
                close $fh;

                my $link = "$log_dir/latest";
                unlink $link if -e $link;
                symlink($log_file, $link) or print_error("[WARN] cannot create symlink $link: $!");
            };
            if($@){
                print_error("[WARN] failed to log request body: $@");
            }
        };
        if($@){
            chomp(my $err = $@);
            print_error("[error] $err");
        }
    }
    $r->send_http_header("application/json");
    $r->print($rb);
    return OK;
}

sub do_magic_fixes {
    my ($r, $llm_req, $slot_id) = @_;

    $llm_req->{id_slot} = 0+$slot_id;

    my $conv_id = $llm_req->{conv_id} // "conv-" . time() . "-" . int(rand(1000000));
    $llm_req->{conv_id} = $conv_id;

    my $msgs = $llm_req->{messages};
    return unless ref $msgs eq 'ARRAY' and @$msgs;

    # a system message only first (chat templates want that): later ones become user messages
    $_->{role} = "user" for grep {($_->{role}//"") eq "system"} @{$msgs}[1..$#$msgs];

    # the boilerplate below is moved out of a system prompt only, not out of a first user message
    $msgs->[0]{role} = "system" if ($msgs->[0]{role}//"") eq "developer";
    return unless ($msgs->[0]{role}//"") eq "system" and defined $msgs->[0]{content} and !ref $msgs->[0]{content};

    my $m = \$llm_req->{messages}[0]{content};
    my $model_env;
    if($$m =~ s/^(You are powered by the model named .*?\. The exact model ID is .*?\n)//gms){
        $model_env = $1;
    }
    my $project_env;
    if($$m =~ s/^(Here is some useful information about the environment you are running in:\n<env>.*?<\/env>)//gms){
        $project_env = $1;
        $project_env =~ s/Today's date: .*?\n//ms;
    }
    my $mcp_instructions;
    if($$m =~ s/^(<mcp_instructions>.*?<\/mcp_instructions>)//gms){
        $mcp_instructions = $1;
    }
    my $skills;
    if($$m =~ s/^(Skills provide specialized instructions and workflows for specific tasks.*?<available_skills>.*?<\/available_skills>)//gms){
        $skills = $1;
    }
    my $agents_instructions;
    if($$m =~ s/^(Instructions from: .*?\/AGENTS\.md.*)//gms){
        $agents_instructions = $1;
    }

    my $fr = shift @{$llm_req->{messages}//[]};
    unshift @{$llm_req->{messages}},
        $fr,
        (length($mcp_instructions//"")?(
            {role => "user", content => $mcp_instructions},
            {role => "assistant", content => "Understood."}
        ):()),
        (length($skills//"")?(
            {role => "user", content => $skills},
            {role => "assistant", content => "Understood."},
        ):()),
        (length($model_env//"")?(
            {role => "user", content => $model_env},
            {role => "assistant", content => "Understood."},
        ):()),
        (length($agents_instructions//"")?(
            {role => "user", content => $agents_instructions},
            {role => "assistant", content => "Understood."},
        ):()),
        (length($project_env//"")?(
            {role => "user", content => $project_env},
            {role => "assistant", content => "Understood."},
        ):());

    return;
}

sub print_error {
    print STDERR @_,"\n";
}

1;
