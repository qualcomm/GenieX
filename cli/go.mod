module github.com/qcom-it-nexa-ai/GenieX/geniex-cli

go 1.24.1

toolchain go1.24.3

replace github.com/qcom-it-nexa-ai/GenieX/geniex-sdk-bindings/go => ../geniex-sdk-bindings/go

require (
	github.com/qcom-it-nexa-ai/GenieX/geniex-sdk-bindings/go v0.0.0-00010101000000-000000000000
	github.com/spf13/cobra v1.10.2
)

require (
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/spf13/pflag v1.0.9 // indirect
)
