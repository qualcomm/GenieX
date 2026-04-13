package main

import (
	"fmt"

	geniex_sdk "github.com/qcom-it-nexa-ai/GenieX/geniex-sdk-bindings/go"
	"github.com/spf13/cobra"

	"github.com/qcom-it-nexa-ai/GenieX/geniex-cli/internal/config"
)

func main() {
	rootCmd := &cobra.Command{
		Use: "geniex",
		Run: func(cmd *cobra.Command, args []string) {

			fmt.Printf("GenieX CLI - Version: %s\n", geniex_sdk.Version())
			c := config.Config{}
			fmt.Printf("test config: %+#v\n", c)
		},
	}
	if err := rootCmd.Execute(); err != nil {
		fmt.Println("GenieX CLI placeholder")
	}
}
