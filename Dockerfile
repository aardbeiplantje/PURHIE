FROM debian:trixie-slim AS base
WORKDIR /tmp
RUN apt update && apt install -y --no-install-recommends \
    ca-certificates \
    unzip \
    libcurl4 \
    libssl3 \
    libgomp1 \
    libatomic1 \
    make \
    gcc \
    g++ \
    git \
    build-essential \
    clang \
    pkg-config \
    curl \
    perl \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ADD https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-10.1.0.tar.gz /tmp/rocm.tar.gz
# this is on AWS S3 and cached badly
# ADD https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx1151-7.14.0a20260612.tar.gz /tmp/rocm.tar.gz
# hence: curl
RUN --mount=type=cache,target=/var/cache/downloads \
    curl -qsSk -L https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.1.0.tar.gz \
        -z file -o /var/cache/downloads/rocm.tar.gz
RUN --mount=type=cache,target=/var/cache/downloads \
    cp /var/cache/downloads/rocm.tar.gz /rocm.tar.gz

# ROCm::Inference (https://github.com/aardbeiplantje/purhie):
# libinference (kernels embedded), the Perl modules ROCm::Inference, Web::Route and Minijinja (with the Rust
# minijinja C library), and the OpenAI-compatible rocm-inference-server, all under /opt/rocm-hip
FROM base AS rocm-hip-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libperl-dev \
    rustc \
    cargo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/rocm \
    && tar -xzf /rocm.tar.gz -C /opt/rocm
ENV ROCM_PATH=/opt/rocm
# with the perl-minijinja-xs submodule (Minijinja's source); CACHEBUST makes a rebuild take new commits
ARG CACHEBUST=1
RUN git clone --depth=1 --single-branch -b main --recurse-submodules --shallow-submodules \
        https://github.com/aardbeiplantje/purhie.git /src/rocm-hip
# the commit built, for /version (the version stage)
RUN git -C /src/rocm-hip log -1 --date=iso-strict --format='rocm-hip %H %cd %s' > /rocm-hip.version \
    && git -C /src/rocm-hip submodule status | sed 's/^[ +-]*/  submodule /' >> /rocm-hip.version
WORKDIR /src/rocm-hip
# Minijinja first: its C library goes into /opt/rocm-hip/lib, which the install below creates as well
RUN mkdir -p /opt/rocm-hip/lib
RUN cd perl-minijinja-xs \
    && bash scripts/setup-minijinja.sh \
    && cp minijinja/target/release/libminijinja_cabi.so /opt/rocm-hip/lib/ \
    && MINIJINJA_RPATH=/opt/rocm-hip/lib perl Makefile.PL INSTALL_BASE=/opt/rocm-hip/perl5 \
    && make && make install
RUN make -j32 install PREFIX=/opt/rocm-hip
RUN cd perl/ROCm-Inference \
    && ROCM_INFERENCE_PREFIX=/opt/rocm-hip perl Makefile.PL INSTALL_BASE=/opt/rocm-hip/perl5 \
    && make && make install
RUN cd perl/Web-Route \
    && perl Makefile.PL INSTALL_BASE=/opt/rocm-hip/perl5 \
    && make && make install

# The ROCm runtime libraries only: the runtime image gets these, not the tarball (a lower layer keeps a
# file even when a later one deletes it) or the build tools of base
FROM base AS rocm-runtime
RUN mkdir -p /opt/rocm \
    && tar -xzf /rocm.tar.gz -C /opt/rocm \
        --wildcards \
        "*/lib/*.so*" \
        "*/lib/rocblas/*" \
        "*/lib/hipblaslt/*"

# /version: the commits of this project (its .git, the bake file's named context docker-git: .dockerignore keeps
# .git out of the main context) and of rocm-hip. Its own stage, copied last: a new commit here rebuilds only that
FROM base AS version
COPY --from=rocm-hip-build /rocm-hip.version /rocm-hip.version

FROM debian:trixie-slim AS runtime
WORKDIR /app
# nginx and the runtime's packages: their own layer, before the copies from the build stages (a rocm-hip
# change does not reinstall them); --build-arg APT_CACHEBUST=<new value> to refresh them
ARG APT_CACHEBUST=1
# (the echo makes the value part of this RUN's cache key: a new value runs apt again)
RUN echo "apt: $APT_CACHEBUST" && apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    libssl3 \
    libgomp1 \
    libatomic1 \
    perl \
    curl \
    nginx \
    libjson-xs-perl \
    libsys-mmap-perl \
    ca-certificates \
    strace \
    lsof \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY --from=rocm-runtime /opt/rocm /opt/rocm
COPY --from=rocm-hip-build /opt/rocm-hip /opt/rocm-hip

# every shared library ROCm::Inference need is here (the image no longer inherits base's packages)
RUN ! LD_LIBRARY_PATH=/opt/rocm/lib:/app/bin \
        ldd /opt/rocm-hip/lib/*.so* $(find /opt/rocm-hip/perl5 -name '*.so') 2>/dev/null | grep 'not found'
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:/app/bin
COPY nginx.conf /nginx.conf

RUN useradd -N -M -d /app-server/ -u 1000 runtime
RUN mkdir -p /models      && chown -R runtime:users /models
RUN mkdir -p /hf          && chown -R runtime:users /hf
RUN mkdir -p /var/lib/nginx/body  && chown runtime /var/lib/nginx/body
RUN mkdir -p /var/lib/nginx/proxy && chown runtime /var/lib/nginx/proxy
RUN mkdir -p /var/lib/nginx/fastcgi && chown runtime /var/lib/nginx/fastcgi
RUN mkdir -p /var/lib/nginx/uwsgi && chown runtime /var/lib/nginx/uwsgi
RUN mkdir -p /var/lib/nginx/scgi && chown runtime /var/lib/nginx/scgi

COPY app.sh /
WORKDIR /app
COPY lib /app/lib

USER runtime
RUN nginx -t -c /nginx.conf
WORKDIR /models
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_CACHE=/hf/hub
ENV HF_HOME=/hf
ENV PERL5LIB=/opt/rocm-hip/perl5/lib/perl5
ENTRYPOINT ["/app.sh"]
