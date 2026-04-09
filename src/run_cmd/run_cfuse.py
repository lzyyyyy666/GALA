COMMAND = ("cd {{repo_dir}} && git reset --hard HEAD && git checkout -f {{commit_id}} && "
           "CFUSE_BIN=\"${CFUSE_BIN:-$(command -v pycfuse || command -v cfuse)}\" && [ -n \"$CFUSE_BIN\" ] && "
           "CFUSE_STREAM_FLAG=\"${CFUSE_STREAM_FLAG:---no-stream}\" && "
           "\"$CFUSE_BIN\" --model {{model_name}} --api-key {{api_key}} --base-url {{base_url}} -pp {{prompt_file}} --logs-dir {{log_dir}} --temperature {{temperature}} --yolo ${CFUSE_STREAM_FLAG} && "
           "git -C {{repo_dir}} add . -A && "
           "git -C {{repo_dir}} diff --cached > {{patch_file}}")
