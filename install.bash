#!/usr/bin/env bash

# Function to check for "-y" flag to skip prompts
for arg; do
  if [ "$arg" = "-y" ]; then
    # Set an environment variable when -y is present
    export SKIP_PROMPT=true
  else
    export SKIP_PROMPT=false
  fi
done

# Package installation functions
mbd_install_linux_packages() {
  sudo apt-get update
  if dpkg -s python3 python3-pip python3-venv libcairo2-dev libgirepository1.0-dev libffi-dev portaudio19-dev python3-gi python3-gi-cairo gir1.2-gtk-3.0 libgirepository1.0-dev >/dev/null 2>&1; then
    echo "All required linux packages are already installed."
    else
      echo "Installing required packages without prompt..."
      sudo apt-get install -y python3 python3-pip python3-venv
      sudo apt-get install -y libcairo2-dev libgirepository1.0-dev
      sudo apt-get install -y libffi-dev
      sudo apt-get install -y portaudio19-dev
      sudo apt install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
      sudo apt-get install -y libgirepository1.0-dev
      CFLAGS="-march=native" pip install pyaudio
      python3 -m pip install hatch
  fi
}

mbd_install_macos_packages() {
  export PATH=/usr/local/bin:/usr/local/sbin:$PATH
  if command -v brew >/dev/null; then
    echo "Homebrew is already installed."
  else
    echo "Installing Homebrew..."
    NONINTERACTIVE=$SKIP_PROMPT /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  if brew list hatch cairo pygobject3 libffi >/dev/null 2>&1; then
    echo "All required macos packages are already installed."
  else
    echo "Installing required packages..."
    brew update
    brew install hatch
    brew install cairo
    brew install pygobject3
    brew install libffi
    brew install portaudio

    PKG_CONFIG_PATH="$(brew --prefix libffi)/lib/pkgconfig:$PKG_CONFIG_PATH"
    export PKG_CONFIG_PATH
    LDFLAGS="-L$(brew --prefix libffi)/lib"
    export LDFLAGS
    CPPFLAGS="-I$(brew --prefix libffi)/include"
    export CPPFLAGS
  fi
}

# Detect the operating system
OS=$(uname -s)

if [[ $OS == "Linux" ]]; then
  if [[ "$SKIP_PROMPT" = false ]]; then
     read -rp "Install required Linux packages? (y/n) " response
   if [[ $response =~ ^[Yy]$ ]]; then
    mbd_install_linux_packages
    else
      echo "Skipping linux package installation."
    fi
  else
    mbd_install_linux_packages
  fi
 
elif [[ $OS == "Darwin" ]]; then
    if [[ "$SKIP_PROMPT" = false ]]; then
      read -rp "Install required macOS packages? (y/n) " response
      if [[ $response =~ ^[Yy]$ ]]; then
        mbd_install_macos_packages
      else
        echo "Skipping package installation."
      fi
    else
      mbd_install_macos_packages
    fi
else
  # Unsupported operating system
  echo "Unsupported operating system: $OS"
  exit 1
fi

if [[ -z "$CI" ]]; then
  # shellcheck disable=SC1091
   hatch run echo "Sourcing environment..."
  source .mbodied/envs/mbodied/bin/activate
else
  echo "CI environment detected. Skipping sourcing of environment."
fi
