#!/usr/bin/env /bin/bash

setup_c_ares_dns_passthrough() {
	local OS
	OS="$(uname)"

	echo "Detected OS: $OS"

	if [ "$OS" = "Darwin" ]; then
		# macOS
		export PYTHONPATH="$(pwd -P)" # -P resolves symlinks
		export GRPC_DNS_RESOLVER="native"
		echo "macOS detected: Set PYTHONPATH to '$PYTHONPATH'"
		echo "macOS detected: Set GRPC_DNS_RESOLVER to $GRPC_DNS_RESOLVER"
	elif [ "$OS" = "Linux" ]; then
		# Linux - don't set variables
		echo "Linux detected: No environment variables set"
		echo "NOTE: This assumes server environment. If Linux is used as laptop OS, might need different configuration."
	elif [[ "$OS" = *"NT"* || "$OS" == "MINGW"* ]]; then
		# Windows
		export GRPC_DNS_RESOLVER="native"
		export PYTHONPATH="$(pwd)"
		echo "Windows detected: Set PYTHONPATH to '$(pwd)'"
		echo "Windows detected: Set GRPC_DNS_RESOLVER to 'native'"
	else
		# Unknown OS
		echo "Unknown OS: $OS - using default configuration"
		export PYTHONPATH="$(pwd)"
		echo "Set PYTHONPATH to '$(pwd)'"
	fi
}

setup_c_ares_dns_passthrough
