cd "/home/yeah/code/LYTrobot_amp"

export PATH="$HOME/snap/code/current/.local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PROJECT="$PWD"
export PYTHONUNBUFFERED=1

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PROMPT_MODIFIER
export CONDA_SHLVL=0

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

if [[ $- == *i* ]]; then
  force_color_prompt=yes
  alias ls='ls --color=auto'
  alias grep='grep --color=auto'
  PS1='\[\033[01;32m\](lyt) \u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
fi
