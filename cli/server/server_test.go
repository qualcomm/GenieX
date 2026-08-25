// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package server

import (
	"strings"
	"testing"
)

func TestHostBindingHint(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		wantHint bool
		wantPort string // if wantHint, the port that should appear in the message
	}{
		{"ipv4 loopback", "127.0.0.1:18181", true, "18181"},
		{"ipv4 loopback other subnet", "127.0.1.5:8080", true, "8080"},
		{"localhost", "localhost:18181", true, "18181"},
		{"localhost uppercase", "LOCALHOST:18181", true, "18181"},
		{"ipv6 loopback", "[::1]:18181", true, "18181"},
		{"any ipv4", "0.0.0.0:18181", false, ""},
		{"any ipv6", "[::]:18181", false, ""},
		{"lan ipv4", "192.168.1.10:18181", false, ""},
		{"malformed no port", "127.0.0.1", false, ""},
		{"empty", "", false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hostBindingHint(tt.host)
			if tt.wantHint && got == "" {
				t.Errorf("hostBindingHint(%q) = %q, want non-empty hint", tt.host, got)
			}
			if !tt.wantHint && got != "" {
				t.Errorf("hostBindingHint(%q) = %q, want empty", tt.host, got)
			}
			if tt.wantHint && !strings.Contains(got, "--host 0.0.0.0:"+tt.wantPort) {
				t.Errorf("hostBindingHint(%q) = %q, want it to suggest --host 0.0.0.0:%s", tt.host, got, tt.wantPort)
			}
		})
	}
}
