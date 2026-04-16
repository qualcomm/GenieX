BRIDGE_VERSION ?= v1.0.45-rc1

ifeq ($(OS), Windows_NT)
	OS      := windows
	ARCH    ?= $(shell powershell -NoProfile -NonInteractive "switch ((Get-CimInstance Win32_Processor).Architecture) { 0 {'x86'} 5 {'arm'} 9 {'x86_64'} 12 {'arm64'} Default {'unknown'} }")
	EXE     := .exe
	RM      := powershell -NoProfile -NonInteractive "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -Path"
	MKDIR   := powershell -NoProfile -NonInteractive "New-Item -ItemType Directory -Force -Path"
	MKLINK  := powershell -NoProfile -NonInteractive "New-Item -ItemType Junction -Path 'build' -Target '..\..\nexasdk-bridge\build\out'"
	UNZIP   := powershell -NoProfile -NonInteractive -Command "Expand-Archive -Path build/nexasdk-bridge.zip -DestinationPath build -Force"
else
	OS      := $(shell echo $(shell uname -s) | tr '[:upper:]' '[:lower:]')
	ifeq ($(OS), darwin)
		OS := macos
	endif
	ARCH    := $(shell uname -m)
	ifeq ($(ARCH), aarch64)
		ARCH := arm64
	endif
	EXE     :=
	RM      := rm -rf
	MKDIR   := mkdir -p
	MKLINK  := ln -s ../../nexasdk-bridge/build/out build
	UNZIP   := unzip -o build/nexasdk-bridge.zip -d build
endif

.PHONY: build link download clean docker docker-cuda

build:
	# go build -ldflags "-s -w" -o build/geniex$(EXE) ./cmd/geniex-launcher
	go build -tags="sonic avx" -ldflags "-X 'main.Version=$(VERSION)' -s -w" -o build/geniex$(EXE) ./cmd/geniex-cli

link: clean
	$(MKLINK)

download: clean
	@echo "====> Download runtime"
	@echo "OS: $(OS)"
	@echo "BRIDGE_VERSION: $(BRIDGE_VERSION)"
	$(MKDIR) build
	curl -L -o build/nexasdk-bridge.zip \
		https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/nexasdk/$(BRIDGE_VERSION)/$(OS)_$(ARCH)/nexasdk-bridge.zip
	$(UNZIP)
	$(RM) build/nexasdk-bridge.zip
	$(RM) build/nexa_bridge.lib

docker: build
	cp release/linux/Dockerfile build/
	docker build --platform arm64 -t qcom-it-nexa-ai/geniex-cli:latest build

clean:
	-$(RM) build
