## 目录结构

- 所有对外发布的库和工具都放在根目录下
- 依赖关系通过构建工具而不是目录来管理
- 统一构建工具, 避免不同子项目使用不同的构建系统导致的复杂性

## Bazel

- 优点
  - 支持 cpp/go/python/android, 支持 docker 镜像打包
  - 整合 lint/test/build, ci 只需要调用 bazel 命令, 本地开发也可以直接使用 bazel, 保持一致性
  - 使用自己维护的编译器, 不受系统环境影响
  - 相比 cmake 更易于集成自定义构建规则
- 缺点
  - 对于小型项目可能过于复杂 (但我们项目涉及多语言多平台)
  - 学习曲线较陡峭 (资料多,AI可以帮助解决)

### 常用命令

- `bazel query //...` list all targets in the workspace
- `bazel run //geniex-cli/cmd/geniex-cli` build and run the geniex-cli target, will not build full project.

#### go tips

- `bazel run //gazelle` update BUILD files from `go.mod`
- `# gazelle:resolve go ://example.com //local/foo:go_default_library` resolve external go module to local bazel target

## files

```
.
├── .github                   #
│   ├── actions               # reusable actions, like env setup and s3 config
│   │   ├── env.yml           #
│   │   └── s3.yml            #
│   ├── scripts               # reusable scripts used by workflows, like create github release
│   │   └── release.js        #
│   └── workflows             # github actions workflows, it's simple because we use bazel for everything
│       ├── build.yml         #
│       ├── lint.yml          #
│       └── test.yml          #
│                             #
├── assets                    # assets files for release, doc, etc
│   └── favicon.ico           #
│                             #
├── geniex                    # core library
│   └── BUILD.bazel           #
│                             #
├── geniex-cli                # command line tool
│   ├── cmd                   #
│   │   └── geniex-cli        #
│   │       ├── BUILD.bazel   #
│   │       └── main.go       #
│   ├── internal              #
│   ├── release               #
│   │   ├── docker.bazel      #
│   │   └── msi.bazel         #
│   ├── server                #
│   ├── BUILD.bazel           #
│   ├── go.mod                #
│   └── go.sum                #
│                             #
├── geniex-proc               # pre/post processing library
│   └── BUILD.bazel           #
│                             #
├── geniex-sdk                # sdk for developers
│   ├── build                 #
│   │   ├── llama.cpp.bazel   #
│   │   └── opencl.bazel      #
│   ├── BUILD.bazel           #
│   ├── include               #
│   ├── libs                  #
│   ├── src                   #
│   │   ├── ml.cpp            #
│   │   └── plugins           #
│   │       ├── geniex        #
│   │       └── llama.cpp     #
│   └── third-party           #
│       ├── llama.cpp         #
│       └── opencl            #
│                             #
├── geniex-sdk-bindings       # language bindings for geniex-sdk
│   ├── android               #
│   │   └── BUILD.bazel       #
│   ├── BUILD.bazel           #
│   ├── go                    #
│   │   ├── BUILD.bazel       #
│   │   ├── go.mod            #
│   │   └── ml.go             #
│   └── python                #
│       └── BUILD.bazel       #
│                             #
├── BUILD.bazel               # root BUILD file
├── MODULE.bazel              # root MODULE file
├── MODULE.bazel.lock         #
└── README.md                 #
```
