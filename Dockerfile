FROM davisteague/glyph-rwkv13-v2@sha256:5bcfbca95d64645d73ef1adf4622bbdf917eaa130d4399448e8eb6bcdb1618e5

USER root

RUN printf '%s\n' '730c373fb26b68bcb966d1bb4de6acd4ffedacec22e4ef1358162cba14bb99a4' \
      > /opt/codec/VALIDATION_RECEIPT_COMMITMENT \
 && chmod 0444 /opt/codec/VALIDATION_RECEIPT_COMMITMENT

LABEL io.glyph.validation-receipt-commitment=730c373fb26b68bcb966d1bb4de6acd4ffedacec22e4ef1358162cba14bb99a4


RUN printf '%s\n' \
      '8004313036bd66fd0af00d65c45dc764acf4ff85535257a6989a5173c0691e6b  /opt/codec/coder.py' \
      '496c477e464a4bf03494d5405cbd5ed20ed902eb3d2a3def23394e4dddb8e7f9  /opt/codec/engine.py' \
      '41bccba980d64d42d74b7195dd074630075f0ce5bbd650ba979ceb09525b1684  /opt/codec/mailbox.py' \
      '664b90af924ed227f34c87971e370b01078ffb736931c488f41c3b6cd7509662  /opt/codec/model/model_int8.safetensors' \
      '4ce8cd792c61bc14fa8bee2519e6eb363b88e5fe86acb11fc985f411af87b7fb  /opt/codec/rwkv_tokenizer.py' \
      'e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89  /opt/codec/rwkv_vocab_v20230424.txt' \
      'bec60c9ca9bbb7dff28bf8e92032dec09b12af1577d08a268a237004ed1b3ceb  /opt/codec/server.py' \
    | sha256sum --check --strict - \
 && test "$(stat -c '%s' /opt/codec/model/model_int8.safetensors)" = '13996094520'

COPY --chown=0:0 --chmod=0444 overlay/hier256_core.py /opt/codec/hier256_core.py
COPY --chown=0:0 --chmod=0444 overlay/coder_hier256.py /opt/codec/coder_hier256.py
COPY --chown=0:0 --chmod=0444 overlay/server_hier256.py /opt/codec/server_hier256.py
COPY --chown=0:0 --chmod=0444 overlay/compress.py /opt/codec/compress.py
COPY --chown=0:0 --chmod=0444 overlay/decompress.py /opt/codec/decompress.py
COPY --chown=0:0 --chmod=0444 overlay/mbclient.py /opt/codec/mbclient.py
COPY --chown=0:0 --chmod=0444 overlay/warmup_uid95.py /opt/codec/warmup_uid95.py
COPY --chown=0:0 --chmod=0444 overlay/warmup.py /opt/codec/warmup.py
COPY --chown=0:0 --chmod=0444 overlay/coder_hier256.py /opt/codec/coder.py
COPY --chown=0:0 --chmod=0444 overlay/server_hier256.py /opt/codec/server.py

RUN printf '%s\n' \
      '04ed04c63cd92fb473a154696a38ada381ff05bb70853491d30f308da598a240  /opt/codec/hier256_core.py' \
      'eff638b3197fe99d57bae0d72ec175a0bf0f40ac94444f40178dc3d6fc920f74  /opt/codec/coder_hier256.py' \
      '6bf931882efa8a87665c0c2ff60891ade55e0784422f141160ef2f6799a3e16c  /opt/codec/server_hier256.py' \
      '5cba90fb16d70b5cda0ae6ba448f2bde1fe3b1f0ea0635dac25ced0ab6d56227  /opt/codec/compress.py' \
      'c6db1477ff893df2817fa886b097c602ea6a186f0ef4e9480f62a37210ea51b4  /opt/codec/decompress.py' \
      '5c4f732d4a2d25c5941d0252416c5cb13ff01bc2b4e3b86a3790f9cdfcfc537e  /opt/codec/mbclient.py' \
      '00c9eda140a3c1285f231aafa4ca0f200cfe136902673859a5c6054e39386aa8  /opt/codec/warmup_uid95.py' \
      '6ba28266a1ef70a97997bf04f00a79c7c00c44499f503a83825c58a2c3abc995  /opt/codec/warmup.py' \
      'eff638b3197fe99d57bae0d72ec175a0bf0f40ac94444f40178dc3d6fc920f74  /opt/codec/coder.py' \
      '6bf931882efa8a87665c0c2ff60891ade55e0784422f141160ef2f6799a3e16c  /opt/codec/server.py' \
    | sha256sum --check --strict - \
 && printf 'GLYPH_B_FIXED=192\nGLYPH_HIER256_ALLOWED_B=192\n' \
      > /opt/codec/HIER256_ENVIRONMENT.txt \
 && chmod 0444 /opt/codec/HIER256_ENVIRONMENT.txt

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    GLYPH_B_FIXED=192 \
    GLYPH_DAEMON_LOG=/scratch/rwkv-hier256-daemon.log \
    GLYPH_GEMM_CFG=32,128,64,4,3 \
    GLYPH_HIER256_ALLOWED_B=192 \
    GLYPH_IDLE_TIMEOUT_SECS=1800 \
    GLYPH_K=4096 \
    GLYPH_MAILBOX_DIR=/scratch/rwkv-mailbox-v1 \
    GLYPH_MODEL=/opt/codec/model/model_int8.safetensors \
    GLYPH_REC_BV=32 \
    GLYPH_REC_STAGES=3 \
    GLYPH_REC_WARPS=4 \
    GLYPH_RUNTIME=int8 \
    GLYPH_SERVER_PATH=/opt/codec/server.py \
    GLYPH_TRITON_SEED=/opt/codec/triton_seed \
    GLYPH_VOCAB=/opt/codec/rwkv_vocab_v20230424.txt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_CACHE_DIR=/scratch/.triton

WORKDIR /opt/codec
