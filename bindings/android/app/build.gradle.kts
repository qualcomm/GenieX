plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.geniex.sdk"
    compileSdk = 35
    ndkVersion = "29.0.14206865"

    defaultConfig {
        minSdk = 27

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    packaging {
        jniLibs.useLegacyPackaging = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// Copy the prebuilt geniex libraries from sdk/pkg-geniex so they get packaged
// into jniLibs (and therefore into the final APK of any consumer).
val pkgGenieXDir = file("$projectDir/../../../sdk/pkg-geniex")
val jniOutDir = file("$projectDir/src/main/jniLibs/arm64-v8a")

val copyBridgeLibs = tasks.register<Copy>("copyBridgeLibs") {
    require(pkgGenieXDir.exists()) {
        "SDK package not found at ${pkgGenieXDir.absolutePath}"
    }
    val libDir = File(pkgGenieXDir, "lib")
    from(libDir) { include("libgeniex.so") }
    from(File(libDir, "llama_cpp")) {
        include("*.so")
        exclude("*.a")
        // The SDK package ships multiple libgeniex_plugin.so variants, one per backend.
        // They must be renamed to coexist in the flat APK jniLibs directory.
        rename("libgeniex_plugin\\.so", "libgeniex_plugin_llama_cpp.so")
    }
    from(File(libDir, "qairt")) {
        // Pull every top-level .so so new transitive deps in libgeniex_plugin.so
        // (e.g. libgeniex_vlm.so, libgeniex-proc.so) get packaged automatically.
        include("*.so")
        rename("libgeniex_plugin\\.so", "libgeniex_plugin_qairt.so")
    }
    // QNN runtime libs (libQnnHtp.so, libQnnSystem.so, V79/V81 stubs/skels,
    // libCalculator_skel.so, FastRPC bits) — installed by the SDK build via
    // third-party/geniex-qairt/CMakeLists.txt's install(DIRECTORY) of its
    // platform htp-files/ folder. Same single source the CLI/python SDK
    // packages consume; no submodule needed at AAR build time.
    //
    // The Windows CLI package's htp-files/ ships Hexagon DSP (32-bit)
    // binaries whose basenames collide with the Android CPU-side libs and
    // would crash on dlopen — but the Android sdk-android-arm64 artifact
    // only ever contains ARM64 + non-colliding DSP skel names, so a flat
    // copy into jniLibs/arm64-v8a/ is safe here.
    from(File(libDir, "qairt/htp-files")) {
        include("*.so")
    }
    from(File(projectDir, "extLibs/arm64-v8a")) { include("*.so") }
    into(jniOutDir)
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

tasks.matching { it.name == "preBuild" }.configureEach {
    dependsOn(copyBridgeLibs)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
}
